#!/usr/bin/env python3
r"""
rpm_ihog_round7.py — end-to-end KL distillation through the whole compressed model
==================================================================================
What changed vs round 6, and why
  Rounds 3-6 fit each layer to its own teacher output (closed loop). Per-layer R² is a poor proxy for
  PPL, and on the 3B the closed-loop R² at 2 bit is only ~40-57 % while the open-loop one is ~78 %:
  errors compound and no single layer is told which of its errors matter for the next-token
  distribution. Round 7 keeps the round-6 per-layer stage (GPTQ codes, damp 1.0) and then trains ALL
  continuous parameters of ALL compressed layers jointly on
        KL( p_teacher(.|x) || p_student(.|x) )       on calibration tokens only,
  with the residual codes FIXED (no repacks). Trained: Kronecker factors A,B, log row/col scales,
  residual gains, out_gain. Storage is unchanged, so bpw is identical to round 6.

  Kronecker factors are trained with INT{--qat_bits} fake quantisation (straight-through estimator)
  using exactly the quantiser that rpm_ihog_round4.quantize_factors_ applies at scoring time
  (one symmetric scale per factor matrix), so the INT4-factor row is the trained one.

  Reported in addition: top-1 agreement with the FP16 model on the held-out text (eval only; nothing
  is chosen on it), and the calibration-internal KL curve (train on calib seqs [0, n-8), monitor
  [n-8, n)) written to e2e_<config>.json.

Memory: per-layer activation checkpointing (weights are re-synthesised in backward), fp32 master
params, fp16 activations with dynamic loss scaling. Teacher log-probs on calibration text are
computed once with the original FFNs and kept on the GPU.

  CUDA_VISIBLE_DEVICES=1 python rpm_ihog_round7.py --configs kron13_2bit_gptq --out_dir ./rpm_ihog_round7_3b_<tag>
"""

import json
import math
import os
import time
import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import rpm_ihog_15bpw_hadamard_distill as V1
import rpm_ihog_round4 as R4
import rpm_ihog_round5 as R5
import rpm_ihog_round6 as R6

E2E = {"teacher_lp": None, "calib": None, "evalt": None, "args": None, "log": {}}


# ------------------------------------------------------------------------------
# STE fake quantisation of Kronecker factors (mirrors R4.quantize_factors_)
# ------------------------------------------------------------------------------
def fq(P, bits):
    qmax = 2 ** (bits - 1) - 1
    s = P.detach().abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12) / qmax
    q = torch.round(P / s).clamp(-qmax - 1, qmax) * s
    return P + (q - P).detach()


def _kron_qat(self, dtype):
    if self.rank == 0:
        return None
    A, B = self.A, self.B
    if getattr(self, "_qat_bits", 0):
        A, B = fq(A.float(), self._qat_bits), fq(B.float(), self._qat_bits)
    A = A.to(dtype) * self.rank_mask.to(dtype)[:, :, None, None]
    return torch.einsum("erij,erkl->eikjl", A, B.to(dtype)).reshape(self.E, self.d_out, self.d_in)


def _ckpt_forward(self, hidden_states, *a, **k):
    shp = hidden_states.shape
    x = hidden_states.reshape(-1, self.d_model)
    y = checkpoint(self.forward_moe, x, use_reentrant=False).reshape(shp)
    return (y, None) if self._return_tuple else y


# ------------------------------------------------------------------------------
# Teacher log-probs (top-k sparse would save memory; 48x256x49k fp16 = 1.2 GB fits on the V100)
# ------------------------------------------------------------------------------
@torch.no_grad()
def teacher_logprobs(model, tokens, dev, bs=4):
    out = []
    for i in range(0, tokens.shape[0], bs):
        lg = model(tokens[i : i + bs].to(dev), use_cache=False).logits.float()
        out.append(F.log_softmax(lg, -1).half())
    return torch.cat(out)


@torch.no_grad()
def top1_agreement(model, tokens, ref_top1, dev, bs=4):
    agree = []
    for i in range(0, tokens.shape[0], bs):
        lg = model(tokens[i : i + bs].to(dev), use_cache=False).logits
        agree.append((lg.argmax(-1) == ref_top1[i : i + bs]).float().mean(1))
    return torch.cat(agree)


@torch.no_grad()
def ref_top1(model, tokens, dev, bs=4):
    return torch.cat([model(tokens[i : i + bs].to(dev), use_cache=False).logits.argmax(-1)
                      for i in range(0, tokens.shape[0], bs)])


def kl_loss(model, toks, tlp):
    lg = model(toks, use_cache=False).logits.float()
    lp = F.log_softmax(lg, -1)
    p = tlp.float().exp()
    return (p * (tlp.float() - lp)).sum(-1).mean()


# ------------------------------------------------------------------------------
# End-to-end stage
# ------------------------------------------------------------------------------
def e2e_train(name, model, layers, dev, args):
    calib, tlp = E2E["calib"], E2E["teacher_lp"]
    n = calib.shape[0]
    n_mon = min(8, n // 4)
    tr_idx, mon_idx = torch.arange(n - n_mon), torch.arange(n - n_mon, n)
    blocks = [R4.get_ffn(model, l) for l in layers]
    banks = [b for blk in blocks for b in blk.banks()]
    for blk in blocks:
        blk.forward = types.MethodType(_ckpt_forward, blk)
        for b in blk.banks():
            b.kron = types.MethodType(_kron_qat, b)
            b._qat_bits = args.qat_bits
    params, scale_params = [], []
    for blk in blocks:
        for nm, p in blk.named_parameters():
            p.data = p.data.float()
            p.requires_grad_(True)
            (scale_params if nm.endswith(("log_r", "log_c")) else params).append(p)
    opt = torch.optim.AdamW([{"params": params, "lr": args.e2e_lr},
                             {"params": scale_params, "lr": args.e2e_lr * args.e2e_scale_lr_mult}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    steps = args.e2e_steps
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12)
    gen = torch.Generator().manual_seed(getattr(args, "seed", 42))

    @torch.no_grad()
    def monitor():
        model.eval()
        v = [float(kl_loss(model, calib[mon_idx[i : i + 4]].to(dev), tlp[mon_idx[i : i + 4]]))
             for i in range(0, n_mon, 4)]
        return sum(v) / len(v)

    log = {"train": [], "monitor": [[0, monitor()]], "skipped_steps": 0}
    print(f"  [e2e {name}] {sum(p.numel() for p in params + scale_params)/1e6:.2f}M trainable params, "
          f"QAT INT{args.qat_bits} factors, {steps} steps x {args.e2e_bs} seqs; calib-monitor KL {log['monitor'][0][1]:.4f}",
          flush=True)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        pick = tr_idx[torch.randint(0, tr_idx.numel(), (args.e2e_bs,), generator=gen)]
        loss = kl_loss(model, calib[pick].to(dev), tlp[pick])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params + scale_params, 1.0)
        s0 = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < s0:
            log["skipped_steps"] += 1
        sched.step()
        if step % 10 == 0 or step == 1:
            log["train"].append([step, float(loss)])
        if step % args.e2e_mon_every == 0 or step == steps:
            log["monitor"].append([step, monitor()])
            print(f"    step {step:4d}  train KL {float(loss):.4f}  calib-monitor KL {log['monitor'][-1][1]:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        for p in params + scale_params:
            p.requires_grad_(False)
            p.data = p.data.half()
        for b in banks:
            b._qat_bits = 0          # scoring applies R4.quantize_factors_ explicitly, like rounds 4-6
    log["sec"] = round(time.perf_counter() - t0, 1)
    return log


def run_chain7(name, spec, model, orig, layers, calib, evalt, dev, args):
    per = R6.run_chain6(name, spec, model, orig, layers, calib, evalt, dev, args)
    # pre-e2e score (held-out, reporting only) so the e2e effect is visible inside this run
    nll_pre = R4.seq_nll(model, evalt, dev)
    t1_pre = top1_agreement(model, evalt, E2E["ref_top1"], dev)
    log = e2e_train(name, model, layers, dev, args)
    for l in layers:
        torch.save(R4.get_ffn(model, l).state_dict(), os.path.join(args.out_dir, f"ckpt_{name}_L{l}_e2e.pt"))
    log["pre_e2e_fp16factors"] = {**R4.ppl_stats(nll_pre, E2E["nll_fp16"]), "top1": round(100 * float(t1_pre.mean()), 2)}
    for fb in args.factor_bits:
        blocks = [R4.get_ffn(model, l) for l in layers]
        saved = R4.quantize_factors_(blocks, fb) if fb < 16 else []
        log[f"top1_f{fb}"] = round(100 * float(top1_agreement(model, evalt, E2E["ref_top1"], dev).mean()), 2)
        R4.restore_(saved)
    print(f"  [e2e {name}] pre-e2e ΔPPL {log['pre_e2e_fp16factors']['dppl']:+} (top1 {log['pre_e2e_fp16factors']['top1']}%)"
          f"; post top1 " + ", ".join(f"f{fb}={log[f'top1_f{fb}']}%" for fb in args.factor_bits), flush=True)
    with open(os.path.join(args.out_dir, f"e2e_{name}.json"), "w") as f:
        json.dump(log, f, indent=1)
    E2E["log"][name] = log
    return per


def install(model, calib, evalt, dev):
    R6.install()
    R5.run_chain5 = run_chain7
    E2E["calib"], E2E["evalt"] = calib, evalt
    E2E["teacher_lp"] = teacher_logprobs(model, calib, dev)
    E2E["ref_top1"] = ref_top1(model, evalt, dev)
    E2E["nll_fp16"] = R4.seq_nll(model, evalt, dev)


def build_parser():
    ap = R6.build_parser()
    ap.add_argument("--e2e_steps", type=int, default=400)
    ap.add_argument("--e2e_bs", type=int, default=4)
    ap.add_argument("--e2e_lr", type=float, default=2e-4)
    ap.add_argument("--e2e_scale_lr_mult", type=float, default=0.5)
    ap.add_argument("--e2e_mon_every", type=int, default=50)
    ap.add_argument("--qat_bits", type=int, default=4, help="0 = no factor QAT")
    ap.set_defaults(configs=["kron13_2bit_gptq"], damp=1.0)
    return ap


def main():
    args = build_parser().parse_args()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"{args.out_dir} exists and is not empty; pick a fresh --out_dir")
    R6.GPTQ["damp"] = args.damp
    print("[round 7] vs round 6: after the per-layer stage, joint end-to-end KL distillation of all continuous "
          f"params (codes fixed), INT{args.qat_bits} factor QAT, {args.e2e_steps} steps, lr {args.e2e_lr}.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    install(model, calib, evalt, dev)
    R5.run_round5(model, calib, evalt, args, dev)


if __name__ == "__main__":
    main()
