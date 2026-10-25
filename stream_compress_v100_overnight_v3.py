#!/usr/bin/env python3
"""RPM-IHOG streaming research codec, format v3 (not an inference engine).

Fixes the dense FP16 scale bug using separable trainable row/column scales.
Stores bounded chunks, verifies serialized decoding, and falls back to exact raw
bytes whenever the actual serialized candidate is no smaller than its input.
Use --help. Only PyTorch is required. Old layer_XX.pt files are never rewritten.
Synthetic linear calibration is a diagnostic, NOT real model distillation.
"""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = 3
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
          "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}

def log(message):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)

def pack_bits(values, bits):
    if bits not in (1, 2, 4):
        raise ValueError("bits must be 1, 2 or 4")
    if ((values < 0) | (values >= 2**bits) | (values != values.round())).any():
        raise ValueError("out-of-range packed code")
    flat = values.reshape(-1).to(torch.uint8)
    per = 8 // bits
    flat = F.pad(flat, (0, (-flat.numel()) % per)).reshape(-1, per)
    out = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
    for i in range(per):
        out |= flat[:, i] << (8 - bits * (i + 1))
    return out.contiguous().clone()

def unpack_bits(packed, bits, count):
    per = 8 // bits
    if packed.dtype != torch.uint8 or packed.numel() != (count + per - 1) // per:
        raise ValueError("invalid packed size/dtype")
    return torch.stack([(packed >> (8-bits*(i+1))) & (2**bits-1)
                        for i in range(per)], dim=-1).reshape(-1)[:count]

def pack_2bit(x): return pack_bits(x, 2)
def pack_1bit_sign(x): return pack_bits((x > 0).to(torch.uint8), 1)
def pack_int4(x): return pack_bits(x, 4)

def build_hadamard_matrix(n, device, dtype=torch.float32):
    if n < 1 or n & (n-1): raise ValueError("Hadamard size must be a power of two")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return (h / math.sqrt(n)).to(dtype)

def apply_block_hadamard(x, h, block_size, transpose=False):
    e, m, n = x.shape
    v = x.reshape(e, m//block_size, block_size, n//block_size, block_size)
    h = h.T if transpose else h
    return torch.einsum("ij,ebjkl,lm->ebikm", h, v, h.T).reshape(e,m,n)

@torch.no_grad()
def decode(payload, device="cpu"):
    """Reference decoder: materializes one chunk as FP32, not a fused kernel."""
    shape = payload["shape"]
    count = math.prod(shape)
    p = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in payload.items()}
    if p["codec"] == "int4":
        q = unpack_bits(p["packed"], 4, p["padded_count"]).float().reshape(-1,p["group_size"])
        return (q*p["scales"] + p["mins"]).flatten()[:count].reshape(shape)
    if p["codec"] != "kron3": raise ValueError("unknown codec")
    q = unpack_bits(p["res_rot_q1_packed2b"], 2, count).long().reshape(shape)
    signs = unpack_bits(p["res_rot_q2_sign_packed1b"], 1, count).float().reshape(shape)*2-1
    levels = torch.tensor([-1.510,-0.453,0.453,1.510], device=device)
    rot = levels[q]*p["res_rot_std"].float() + signs*p["alpha_1b"].float()
    h = build_hadamard_matrix(p["block_size"],device)
    residual = apply_block_hadamard(rot,h,p["block_size"],True)
    kron = torch.einsum("erij,erkl->eikjl",p["A"].float(),p["B"].float()).reshape(shape)
    return (kron+p["res_gain"].float()*residual)*p["row_scale"].float()*p["col_scale"].float()

@torch.no_grad()
def quality(w, restored):
    w, restored = w.double(), restored.double()
    if not torch.isfinite(restored).all(): raise ValueError("non-finite decode")
    signal = w.square().sum().item()
    error = (w-restored).square().sum().item()
    return {"signal_sum_squares": signal, "error_sum_squares": error,
            "decoded_weight_sqnr_db": 10*math.log10(signal/error) if signal>0 and error>0 else None,
            "max_abs_error": (w-restored).abs().max().item()}

@torch.no_grad()
def quantize_int4(w, group_size=128):
    shape = list(w.shape)
    flat = w.float().flatten()
    padded = F.pad(flat,(0,(-flat.numel())%group_size)).reshape(-1,group_size)
    low = padded.amin(1,keepdim=True)
    scale = (padded.amax(1,keepdim=True)-low)/15
    scale = torch.where(scale>0,scale,torch.ones_like(scale))
    q = ((padded-low)/scale).round().clamp(0,15).to(torch.uint8)
    return {"codec":"int4", "shape":shape, "group_size":group_size,
            "padded_count":padded.numel(), "packed":pack_int4(q).cpu(),
            "scales":scale.cpu().clone(), "mins":low.cpu().clone()}

def distill_expert_chunk(
    W_chunk: torch.Tensor,
    rank_k: int = 16,
    p1: int = 16,
    q1: int = 16,
    steps: int = 40,
    device: torch.device = None,
    seed: int = 42,
    factor_dtype: str = "f16",
) -> Tuple[dict, dict]:
    """
    Compresses a chunk of experts [E_chunk, d_out, d_in] using Activation-Weighted
    Kronecker SVD + Sub-Block Walsh-Hadamard 2-bit + 1-bit residual.
    Fits synthetic linear outputs; returns packed tensors and proxy diagnostics.
    """
    E_chunk, d_out, d_in = W_chunk.shape
    p1 = min(p1, max(1, d_out // 2))
    q1 = min(q1, max(1, d_in // 2))
    while d_out % p1 != 0 and p1 > 1:
        p1 -= 1
    while d_in % q1 != 0 and q1 > 1:
        q1 -= 1
    p2, q2 = d_out // p1, d_in // q1

    block_size = 128
    while (d_out % block_size != 0 or d_in % block_size != 0) and block_size > 1:
        block_size //= 2

    H_block = build_hadamard_matrix(block_size, device, torch.float32)

    # Synthetic heavy-tailed calibration & held-out evaluation activations
    rng = torch.Generator(device=device).manual_seed(seed)
    X_calib = torch.randn(128, d_in, device=device, generator=rng) * 0.2
    X_calib[:, ::32] *= 4.0
    X_eval = torch.randn(64, d_in, device=device, generator=rng) * 0.2
    X_eval[:, ::32] *= 4.0

    with torch.no_grad():
        act_rms = torch.sqrt(torch.mean(X_calib ** 2, dim=0, keepdim=True)).unsqueeze(0) + 1e-8
        W_act = W_chunk * act_rms

        row_s = torch.sqrt(torch.mean(W_act ** 2, dim=2, keepdim=True)) + 1e-8
        col_s = torch.sqrt(torch.mean((W_act / row_s) ** 2, dim=1, keepdim=True)) + 1e-8
        W_bal = W_act / (row_s * col_s)

        K_mat = W_bal.reshape(E_chunk, p1, p2, q1, q2).permute(0, 1, 3, 2, 4).reshape(E_chunk, p1 * q1, p2 * q2)
        uk, sk, vtk = torch.linalg.svd(K_mat, full_matrices=False)

        sk_sq = sk ** 2
        base_kron_srank = float(torch.mean(torch.sum(sk_sq, dim=-1) / (sk_sq[:, 0] + 1e-12)).item())

        rk = min(rank_k, sk.shape[-1])
        sqrt_sk = torch.sqrt(sk[:, :rk])
        A_init = (uk[:, :, :rk] * sqrt_sk.unsqueeze(1)).transpose(1, 2).reshape(E_chunk, rk, p1, q1)
        B_init = (vtk[:, :rk, :] * sqrt_sk.unsqueeze(2)).reshape(E_chunk, rk, p2, q2)

        W_k0 = torch.einsum("erij,erkl->eikjl", A_init, B_init).reshape(E_chunk, d_out, d_in)
        res0 = W_bal - W_k0
        res_rot = apply_block_hadamard(res0, H_block, block_size, transpose=False)

        levels_2bit = torch.tensor([-1.510, -0.453, 0.453, 1.510], device=device, dtype=torch.float32)
        r_std = torch.std(res_rot, dim=2, keepdim=True, unbiased=False) + 1e-12
        idx2 = torch.argmin(torch.abs(res_rot.unsqueeze(-1) / r_std.unsqueeze(-1) - levels_2bit), dim=-1)
        q1_rot = levels_2bit[idx2] * r_std

        res_rem = res_rot - q1_rot
        alpha_1b = torch.mean(torch.abs(res_rem), dim=2, keepdim=True)
        q2_sign = torch.sign(res_rem)
        q2_sign_fixed = torch.where(q2_sign == 0, torch.ones_like(q2_sign), q2_sign)
        q2_rot = q2_sign_fixed * alpha_1b

        W_res_fixed = apply_block_hadamard(q1_rot + q2_rot, H_block, block_size, transpose=True)
        eff_scale = (row_s * col_s) / act_rms

        # One-shot reconstruction & held-out accuracy before distillation
        W_oneshot = (W_k0 + W_res_fixed) * eff_scale
        Y_eval_true = torch.einsum("nd,eod->eno", X_eval, W_chunk)
        Y_eval_oneshot = torch.einsum("nd,eod->eno", X_eval, W_oneshot)
        var_true = torch.var(Y_eval_true, unbiased=False) + 1e-12
        oneshot_r2_pct = float((1.0 - torch.mean((Y_eval_true - Y_eval_oneshot) ** 2) / var_true).item() * 100.0)

        Y_calib_target = torch.einsum("nd,eod->eno", X_calib, W_chunk)
        del W_oneshot, Y_eval_oneshot, K_mat, uk, sk, vtk, res0, res_rot, res_rem

    # Synthetic linear fitting: actual gate/up/down composition requires model integration
    A_param = nn.Parameter(A_init.clone())
    B_param = nn.Parameter(B_init.clone())
    row_mult = nn.Parameter(torch.ones_like(row_s))
    col_mult = nn.Parameter(torch.ones_like(col_s))
    res_gain = nn.Parameter(torch.ones(E_chunk, d_out, 1, device=device))

    opt = torch.optim.AdamW([A_param, B_param, row_mult, col_mult, res_gain], lr=4e-3, weight_decay=1e-4)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        W_k = torch.einsum("erij,erkl->eikjl", A_param, B_param).reshape(E_chunk, d_out, d_in)
        W_synth = (W_k + res_gain * W_res_fixed) * (row_s * row_mult * (col_s / act_rms) * col_mult)
        Y_pred = torch.einsum("nd,eod->eno", X_calib, W_synth)
        loss = F.mse_loss(Y_pred, Y_calib_target)
        loss.backward()
        opt.step()

    with torch.no_grad():
        W_k_final = torch.einsum("erij,erkl->eikjl", A_param, B_param).reshape(E_chunk, d_out, d_in)
        final_scale = row_s * row_mult * (col_s / act_rms) * col_mult
        W_distilled = (W_k_final + res_gain * W_res_fixed) * final_scale

        # Weight-space SQNR (dB)
        sig_pwr = torch.mean(W_chunk ** 2).item() + 1e-12
        mse_dist = torch.mean((W_chunk - W_distilled) ** 2).item() + 1e-12
        sqnr_db = 10.0 * math.log10(sig_pwr / mse_dist)

        # Functional held-out R^2 and Cosine Similarity
        Y_eval_dist = torch.einsum("nd,eod->eno", X_eval, W_distilled)
        distilled_r2_pct = float((1.0 - torch.mean((Y_eval_true - Y_eval_dist) ** 2) / var_true).item() * 100.0)
        distilled_cos_sim = float(
            F.cosine_similarity(Y_eval_true.flatten(), Y_eval_dist.flatten(), dim=0).item()
        )

        # Bit-pack 2-bit base grid and 1-bit sign residual before moving to CPU
        packed_q1 = pack_2bit(idx2).cpu()
        packed_q2_sign = pack_1bit_sign(q2_sign_fixed).cpu()

        comp_chunk = {
            "A": A_param.detach().to(torch.float16 if factor_dtype=="f16" else torch.float32).cpu().clone(),
            "B": B_param.detach().to(torch.float16 if factor_dtype=="f16" else torch.float32).cpu().clone(),
            "row_scale": (row_s * row_mult).detach().cpu().clone(),
            "col_scale": ((col_s / act_rms) * col_mult).detach().cpu().clone(),
            "res_gain": res_gain.detach().cpu().clone(),
            "res_rot_q1_packed2b": packed_q1,
            "res_rot_q2_sign_packed1b": packed_q2_sign,
            "res_rot_std": r_std.cpu().clone(),
            "alpha_1b": alpha_1b.cpu().clone(),
            "p1": p1,
            "q1": q1,
            "p2": p2,
            "q2": q2,
            "d_out": d_out,
            "d_in": d_in,
            "block_size": block_size,
            "bit_packed": True,
            "codec": "kron3",
            "shape": list(W_chunk.shape),
        }

        W_loaded = decode(comp_chunk, device)
        if not torch.isfinite(W_loaded).all():
            raise ValueError("Non-finite serialized reconstruction; reduce learning rate/steps")
        chunk_metrics = {
            **quality(W_chunk, W_loaded),
            "metric_scope": "synthetic_linear_projection_not_model_accuracy",
            "base_kron_srank": base_kron_srank,
            "oneshot_r2_pct": oneshot_r2_pct,
            "distilled_r2_pct": distilled_r2_pct,
            "distilled_cos_sim": distilled_cos_sim,
            "weight_sqnr_db": sqnr_db,
        }

    del A_param, B_param, row_mult, col_mult, res_gain, opt, W_k_final, W_distilled, W_res_fixed
    return comp_chunk, chunk_metrics



def fetch_byte_range(url, start, end, retries=4):
    """Require exact 206 semantics; never read a whole shard on ignored Range."""
    if start < 0 or end < start: raise ValueError("invalid range")
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    token = os.environ.get("HF_TOKEN")
    if token: headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=180) as resp:
                expected = f"bytes {start}-{end}/"
                if resp.status != 206 or not resp.headers.get("Content-Range","").startswith(expected):
                    raise ValueError("Server ignored or mismatched HTTP Range; aborting")
                data = resp.read(end-start+2)
                if len(data) != end-start+1: raise IOError("short/oversized HTTP range")
                return data
        except (OSError, urllib.error.URLError):
            if attempt+1 == retries: raise
            time.sleep(2**attempt)


def get_json(url):
    headers = {"Accept-Encoding":"identity"}
    if os.environ.get("HF_TOKEN"): headers["Authorization"] = "Bearer "+os.environ["HF_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as r:
        return json.load(r)


class HFShardStreamer:
    """Contiguous expert/row/byte slices; no full-bank download or prefetch."""
    def __init__(self, model_id, revision="main", local_dir=None):
        self.local = Path(local_dir).resolve() if local_dir else None
        self.model_id = model_id
        self.headers = {}
        if self.local:
            self.revision = "local"
            index = self.local / "model.safetensors.index.json"
            if index.exists(): self.weight_map = json.loads(index.read_text())["weight_map"]
            else: self.weight_map = {k:"model.safetensors" for k in self.header("model.safetensors")[0] if k != "__metadata__"}
        else:
            quoted = urllib.parse.quote(revision,safe="")
            meta = get_json(f"https://huggingface.co/api/models/{model_id}/revision/{quoted}")
            self.revision = meta["sha"]
            self.base_url = f"https://huggingface.co/{model_id}/resolve/{self.revision}"
            try: self.weight_map = get_json(self.base_url+"/model.safetensors.index.json")["weight_map"]
            except urllib.error.HTTPError as e:
                if e.code != 404: raise
                self.weight_map = {k:"model.safetensors" for k in self.header("model.safetensors")[0] if k != "__metadata__"}

    def local_path(self, shard):
        p = (self.local / shard).resolve()
        if not p.is_relative_to(self.local): raise ValueError("unsafe shard path")
        return p

    def read(self, shard, start, end):
        if end == start-1: return b""
        if self.local:
            with self.local_path(shard).open("rb") as f:
                f.seek(start)
                data = f.read(end-start+1)
            if len(data) != end-start+1: raise IOError("truncated local shard")
            return data
        return fetch_byte_range(self.base_url+"/"+urllib.parse.quote(shard,safe="/"),start,end)

    def header(self, shard):
        if shard not in self.headers:
            length = struct.unpack("<Q",self.read(shard,0,7))[0]
            if not 2 <= length <= 100_000_000: raise ValueError("invalid safetensors header size")
            header = json.loads(self.read(shard,8,7+length))
            self.headers[shard] = (header,8+length)
        return self.headers[shard]

    def info(self, key):
        h, offset = self.header(self.weight_map[key])
        info = h[key]
        if any(not isinstance(d,int) or d<0 for d in info["shape"]): raise ValueError("invalid shape")
        start,end = info["data_offsets"]
        if start<0 or end<start: raise ValueError("invalid offsets")
        if info["dtype"] in DTYPES:
            element = torch.empty((),dtype=DTYPES[info["dtype"]]).element_size()
            if end-start != math.prod(info["shape"])*element: raise ValueError("shape/dtype/byte mismatch")
        return info,offset

    def source_identity(self):
        signature = {"model_id":self.model_id,"revision":self.revision}
        if self.local:
            signature["local_dir"] = str(self.local)
            signature["shards"] = {}
            for shard in sorted(set(self.weight_map.values())):
                p = self.local_path(shard)
                signature["shards"][shard] = {"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns,
                                             "header":self.header(shard)[0]}
        return signature


def atomic_json(path, obj):
    tmp = Path(str(path)+".tmp")
    with tmp.open("w") as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def save_chunk(out, name, raw, payload, metadata, min_sqnr=None):
    """Compare actual files; raw fallback has zero per-file framing overhead."""
    stats = {}
    selected = "raw"
    candidate = out / (name+".pt.tmp")
    if payload is not None:
        torch.save(payload,candidate)
        # Roundtrip actual file, not only the in-memory float32 reconstruction.
        loaded = torch.load(candidate,map_location="cpu",weights_only=True)
        decoded = decode(loaded)
        original = torch.frombuffer(bytearray(raw),dtype=DTYPES[metadata["dtype"]]).reshape(metadata["chunk_shape"])
        stats = quality(original,decoded.reshape_as(original).to(original.dtype).float())
        sqnr = stats["decoded_weight_sqnr_db"]
        quality_ok = min_sqnr is None or stats["error_sum_squares"]==0 or (sqnr is not None and sqnr >= min_sqnr)
        if candidate.stat().st_size < len(raw) and quality_ok:
            selected = payload["codec"]
            path = out / (name+".pt")
            os.replace(candidate,path)
        else:
            candidate.unlink()
    if selected == "raw":
        path = out / (name+".bin")
        tmp = Path(str(path)+".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp,path)
    return {**metadata,"codec":selected,"file":path.name,"source_bytes":len(raw),
            "stored_bytes":path.stat().st_size,"sha256":file_digest(path),
            "source_sha256":hashlib.sha256(raw).hexdigest(),
            "candidate_quality":stats,
            "decoded_error_sum_squares":0.0 if selected=="raw" else stats["error_sum_squares"]}


def classify(key, info, quantize_ngram):
    k = key.lower()
    shape = info["shape"]
    if info["dtype"] not in ("BF16","F16","F32"): return "raw"
    if "experts." in k and "shared_expert" not in k and len(shape) in (2,3) and (k.endswith("weight") or k.rsplit(".",1)[-1] in ("gate_up_proj","gate_proj","up_proj","down_proj")):
        return "kron3"
    if quantize_ngram and len(shape)==2 and any(s in k for s in ("ngram","n_gram","ple","memory_table","lookup","embed_tokens")) and not any(s in k for s in ("norm","conv","experts")):
        return "int4"
    return "raw"


def chunk_ranges(info, codec, chunk_size, row_batch, max_bytes):
    shape = info["shape"]
    size = info["data_offsets"][1]-info["data_offsets"][0]
    if codec=="raw":
        for start in range(0,size,max_bytes): yield start,min(size,start+max_bytes),None
        if size==0: yield 0,0,None
        return
    rows = shape[0] if len(shape)==3 or codec=="int4" else 1
    row_bytes = size//rows if rows else 0
    if row_bytes==0:
        yield 0,size,None
        return
    if row_bytes>max_bytes:
        raise ValueError(f"One expert/row needs {row_bytes} source bytes; increase --max_chunk_mb or use --codec int4")
    batch = min(chunk_size if codec=="kron3" else row_batch,max(1,max_bytes//row_bytes))
    for first in range(0,rows,batch):
        count = min(batch,rows-first)
        chunk_shape = [count]+shape[1:] if len(shape)==3 or codec=="int4" else [1]+shape
        yield first*row_bytes,(first+count)*row_bytes,chunk_shape


def run(args):
    source = HFShardStreamer(args.model_id,args.revision,args.local_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    manifest_path = out/"manifest.json"
    settings = {k:v for k,v in vars(args).items() if k not in ("out_dir","out_stats","inspect","device")}
    identity = {"version":VERSION,"script_sha256":file_digest(__file__),"source":source.source_identity(),"settings":settings}
    run_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    selected = []
    skipped = []
    for key in sorted(source.weight_map):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.",key)
        excluded = args.total_layers is not None and match and int(match[1])>=args.total_layers
        if excluded or (args.key_regex and not re.search(args.key_regex,key)):
            skipped.append(key)
            continue
        info,_ = source.info(key)
        codec = classify(key,info,args.quantize_ngram)
        if codec=="raw" and not args.include_other:
            skipped.append(key)
            continue
        if args.codec=="int4" and codec=="kron3":
            # Flatten each expert bank into rows for bounded groupwise baseline.
            codec="int4"
        selected.append((key,info,codec))
    if not selected: raise ValueError("No tensors matched. Inspect model keys or use --include-other.")
    log(f"Pinned revision: {source.revision}; selected {len(selected)} / {len(source.weight_map)} tensors")
    if args.inspect:
        for key,info,codec in selected: log(f"{codec}: {key} {info['dtype']} {info['shape']}")
        return
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest["run_id"]!=run_id: raise ValueError("Resume source/settings mismatch; use a new output directory")
    else:
        if any(out.iterdir()): raise ValueError("Output directory is not empty and has no v3 manifest; use a new directory")
        manifest={"run_id":run_id,"identity":identity,"chunks":{},"skipped_keys":skipped,"complete":False,
                  "scope":"all_safetensors_weights" if not skipped else "selected_weights_only",
                  "warning":"Research archive; requires decoder integration, config/tokenizer, and model evaluation"}
        atomic_json(manifest_path,manifest)
    device=torch.device(args.device)
    expected=[]
    for key,original_info,codec in selected:
        info=dict(original_info)
        if codec=="int4" and len(info["shape"])==3:
            e,m,n=info["shape"]
            info["shape"]=[e*m,n]
        for start,end,shape in chunk_ranges(info,codec,args.chunk_size,args.row_batch,int(args.max_chunk_mb*1024**2)):
            name=hashlib.sha256(f"{key}:{start}:{end}".encode()).hexdigest()[:24]
            expected.append(name)
            old=manifest["chunks"].get(name)
            if old:
                p=out/old["file"]
                if p.is_file() and p.stat().st_size==old["stored_bytes"] and file_digest(p)==old["sha256"]:
                    continue
                log(f"Rebuilding missing/corrupt chunk {name}")
            _,base=source.info(key)
            raw=source.read(source.weight_map[key],base+info["data_offsets"][0]+start,base+info["data_offsets"][0]+end-1)
            payload=None
            t0=time.perf_counter()
            if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
            if shape is not None:
                w=torch.frombuffer(bytearray(raw),dtype=DTYPES[info["dtype"]]).reshape(shape).to(device).float()
                if not torch.isfinite(w).all(): raise ValueError(f"Non-finite input in {key}")
                if codec=="kron3":
                    payload,_=distill_expert_chunk(w,rank_k=args.rank,steps=args.steps,device=device,seed=args.seed,factor_dtype=args.factor_dtype)
                else: payload=quantize_int4(w,args.group_size)
                del w
            meta={"key":key,"dtype":info["dtype"],"original_shape":original_info["shape"],
                  "chunk_shape":shape,"byte_start":start,"byte_end":end}
            record=save_chunk(out,name,raw,payload,meta,args.min_sqnr_db)
            record["encode_verify_write_seconds"]=time.perf_counter()-t0
            record["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
            manifest["chunks"][name]=record
            manifest["complete"]=False
            atomic_json(manifest_path,manifest)
            log(f"{key} [{start}:{end}]: {len(raw)} -> {record['stored_bytes']} bytes ({record['codec']})")
            del raw,payload
            gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
    manifest["complete"]=True
    manifest["expected_chunks"]=expected
    records=[manifest["chunks"][k] for k in expected]
    source_bytes=sum(r["source_bytes"] for r in records)
    stored_bytes=sum(r["stored_bytes"] for r in records)
    elements=sum(math.prod(info["shape"]) for _,info,_ in selected)
    manifest["summary"]={"source_bytes":source_bytes,"payload_disk_bytes":stored_bytes,
                         "selected_elements":elements,
                         "payload_effective_bpw":8*stored_bytes/elements if elements else None,
                         "payload_ratio":source_bytes/stored_bytes if stored_bytes else None,
                         "raw_fallback_chunks":sum(r["codec"]=="raw" for r in records)}
    atomic_json(manifest_path,manifest)
    archive_bytes=stored_bytes+manifest_path.stat().st_size
    stats={**manifest["summary"],"archive_bytes_including_manifest":archive_bytes,
           "archive_ratio":source_bytes/archive_bytes if archive_bytes else None,
           "archive_effective_bpw":8*archive_bytes/elements if elements else None,
           "scope":manifest["scope"],"skipped_tensor_count":len(skipped)}
    if args.out_stats: atomic_json(Path(args.out_stats),stats)
    log(json.dumps(stats))



def restore_archive(out_dir, restore_dir):
    """Write selected tensors to model.safetensors using one decoded chunk at a time.

    Config, tokenizer, non-tensor assets and inference integration are external.
    A partial archive restores ONLY its selected keys. No original model loading.
    """
    out=Path(out_dir)
    manifest=json.loads((out/"manifest.json").read_text())
    if not manifest.get("complete") or manifest["identity"]["version"]!=VERSION:
        raise ValueError("Incomplete or incompatible archive")
    grouped={}
    for name in manifest["expected_chunks"]:
        r=manifest["chunks"][name]
        path=out/r["file"]
        if path.parent.resolve()!=out.resolve() or file_digest(path)!=r["sha256"]:
            raise ValueError("Archive integrity failure")
        grouped.setdefault(r["key"],[]).append(r)
    header={}
    offset=0
    for key,records in grouped.items():
        records.sort(key=lambda r:r["byte_start"])
        cursor=0
        for r in records:
            if r["byte_start"]!=cursor or r["byte_end"]-cursor!=r["source_bytes"]:
                raise ValueError("Missing/overlapping source ranges")
            cursor=r["byte_end"]
        first=records[0]
        if first["dtype"] in DTYPES:
            expected=math.prod(first["original_shape"])*torch.empty((),dtype=DTYPES[first["dtype"]]).element_size()
            if cursor!=expected: raise ValueError("Incomplete tensor coverage")
        header[key]={"dtype":first["dtype"],"shape":first["original_shape"],"data_offsets":[offset,offset+cursor]}
        offset+=cursor
    destination=Path(restore_dir)
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/"model.safetensors"
    if path.exists(): raise ValueError("Restored model.safetensors already exists")
    encoded=json.dumps(header,separators=(",",":")).encode()
    encoded+=b" "*((-len(encoded))%8)
    tmp=destination/"model.safetensors.tmp"
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q",len(encoded)))
        f.write(encoded)
        for records in grouped.values():
            for r in records:
                if r["codec"]=="raw": data=(out/r["file"]).read_bytes()
                else:
                    decoded=decode(torch.load(out/r["file"],map_location="cpu",weights_only=True))
                    tensor=decoded.to(DTYPES[r["dtype"]]).contiguous()
                    if not torch.isfinite(tensor).all(): raise ValueError("Non-finite dtype conversion")
                    # NumPy is not required; clone keeps storage exactly the chunk's bytes.
                    data=bytes(tensor.view(torch.uint8).flatten().untyped_storage())
                if len(data)!=r["source_bytes"]: raise ValueError("Decoded byte length mismatch")
                f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)
    log(f"Restored {len(grouped)} tensors to {path}; scope={manifest['scope']}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id",default="Qwen/Qwen3.8-Flash-Next")
    p.add_argument("--revision",default="main",help="Resolved to immutable HF commit before reading")
    p.add_argument("--local_dir",help="Use local safetensors checkpoint, no network")
    p.add_argument("--out_dir",default="compressed_v3")
    p.add_argument("--out_stats")
    p.add_argument("--restore_dir",help="Decode an existing --out_dir archive to this directory")
    p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total_layers",type=int,help="Optional layer-index upper bound; default discovers all")
    p.add_argument("--chunk_size",type=int,default=1,help="Experts per chunk; start at 1 on V100")
    p.add_argument("--max_chunk_mb",type=float,default=16,help="Source bytes per chunk, not peak compute RAM")
    p.add_argument("--row_batch",type=int,default=1024)
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--rank",type=int,default=16)
    p.add_argument("--factor_dtype",choices=("f16","f32"),default="f16",help="Factor storage; f32 is a larger precision-control baseline")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--group_size",type=int,default=128)
    p.add_argument("--codec",choices=("kron3","int4"),default="kron3")
    p.add_argument("--quantize_ngram",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--include-other",action="store_true",help="Copy all other tensor bytes unchanged")
    p.add_argument("--key_regex",help="Restrict to matching tensor keys; reported as partial archive")
    p.add_argument("--min_sqnr_db",type=float,help="Raw fallback below decoded weight SQNR threshold")
    p.add_argument("--inspect",action="store_true",help="Read headers and print selection only")
    args=p.parse_args()
    for name in ("chunk_size","row_batch","rank","group_size"):
        if getattr(args,name)<1: p.error(f"{name} must be positive")
    if args.steps<0 or not math.isfinite(args.max_chunk_mb) or args.max_chunk_mb<=0: p.error("invalid steps/chunk budget")
    if args.total_layers is not None and args.total_layers<1: p.error("total_layers must be positive")
    if args.restore_dir:
        restore_archive(args.out_dir,args.restore_dir)
    else:
        run(args)

if __name__=="__main__": main()
