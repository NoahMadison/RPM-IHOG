#!/usr/bin/env python3
r"""
rpm_ihog_15bpw_hadamard_distill.py
========================================================================================
Rigorous Multi-Layer & Single-Layer Benchmark for RPM-IHOG with Packed Hadamard Residuals
on Real Granite MoE Activations (`ibm-granite/granite-3.1-1b-a400m-base`).

Directly compares 6 representations with physically bit-packed `uint8` checkpoints:
  1. `zero_moe_control`        (0.00 bpw) : MoE output = 0 (measures residual skip path alone)
  2. `pure_kronecker_0.5bpw`   (0.51 bpw) : Rank-13 Per-Expert Kronecker Core (no residual)
  3. `rpm_ihog_1.5bpw`         (1.51 bpw) : 0.51 bpw Kronecker + 1.00 bpw 1-Bit Sign Hadamard
                                            (Packed 8 weights per uint8 byte -> TRUE 1.5 bpw!)
  4. `rpm_ihog_ternary_2.1bpw` (2.11 bpw) : 0.51 bpw Kronecker + 1.60 bpw Ternary Hadamard
                                            (Packed 5 trits per uint8 byte, 3^5 = 243 <= 256)
  5. `v8_2b1b_hadamard_3.5bpw` (3.51 bpw) : Your previous v8 formula (0.51 bpw Kronecker +
                                            2-bit Hadamard + 1-bit Sign Hadamard packed in uint8)
  6. `int4_group64_4.25bpw`    (4.25 bpw) : Standard INT4 group-64 weight quantization control
                                            (Packed 2 weights per uint8 byte + FP16 group scales)

Every representation is:
  - Initialized & distilled on real calibration tokens (matching complete routed gate+up+down MoE)
  - Closed-loop compensated across multiple layers when `--layers` has >1 layer
  - Physically bit-packed into `uint8` tensors and saved to disk (`.pt`)
  - Reloaded from disk into a fresh module before measuring Held-Out R^2 and Full-Model Perplexity.
"""

import argparse
import json
import math
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# 1. FAST WALSH-HADAMARD TRANSFORM & PHYSICAL BIT-PACKING UTILITIES
# ==============================================================================
_HADAMARD_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def get_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    key = (n, device, dtype)
    if key not in _HADAMARD_CACHE:
        assert n > 0 and (n & (n - 1)) == 0, f"Dimension {n} must be a power of 2 for Walsh-Hadamard."
        H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        _HADAMARD_CACHE[key] = (H / math.sqrt(n)).to(dtype)
    return _HADAMARD_CACHE[key]


def pack_1bit_signs_to_uint8(sign_tensor: torch.Tensor) -> torch.Tensor:
    """Packs {-1, +1} tensor into uint8 at 8 weights per byte (1.000 bpw exact)."""
    bits = (sign_tensor > 0).to(torch.uint8).reshape(-1, 8)
    shifts = torch.arange(8, device=sign_tensor.device, dtype=torch.uint8).unsqueeze(0)
    packed = torch.sum(bits << shifts, dim=-1, dtype=torch.uint8)
    return packed


def unpack_1bit_signs_from_uint8(packed: torch.Tensor, target_shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Unpacks uint8 bytes back into {-1.0, +1.0} tensor of `target_shape`."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8).unsqueeze(0)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(dtype)
    return (bits * 2.0 - 1.0).reshape(target_shape)


def pack_2bit_indices_to_uint8(idx_tensor: torch.Tensor) -> torch.Tensor:
    """Packs {0, 1, 2, 3} indices into uint8 at 4 weights per byte (2.000 bpw exact)."""
    idx = idx_tensor.to(torch.uint8).reshape(-1, 4)
    shifts = torch.tensor([0, 2, 4, 6], device=idx_tensor.device, dtype=torch.uint8).unsqueeze(0)
    return torch.sum(idx << shifts, dim=-1, dtype=torch.uint8)


def unpack_2bit_indices_from_uint8(packed: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8).unsqueeze(0)
    idx = (packed.unsqueeze(-1) >> shifts) & 3
    return idx.reshape(target_shape).long()


def pack_ternary_indices_to_uint8(idx_tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs ternary indices {0, 1, 2} into uint8 at 5 trits per byte (3^5 = 243 <= 255),
    achieving 8 / 5 = 1.600 bpw physical storage.
    """
    flat = idx_tensor.to(torch.int32).reshape(-1)
    pad = (5 - (flat.numel() % 5)) % 5
    if pad > 0:
        flat = F.pad(flat, (0, pad), value=1)
    trits = flat.reshape(-1, 5)
    powers = torch.tensor([1, 3, 9, 27, 81], device=idx_tensor.device, dtype=torch.int32).unsqueeze(0)
    packed = torch.sum(trits * powers, dim=-1).to(torch.uint8)
    return packed


def unpack_ternary_indices_from_uint8(packed: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    total_el = math.prod(target_shape)
    val = packed.to(torch.int32).unsqueeze(-1)
    powers = torch.tensor([1, 3, 9, 27, 81], device=packed.device, dtype=torch.int32).unsqueeze(0)
    trits = (val // powers) % 3
    return trits.reshape(-1)[:total_el].reshape(target_shape).long()


def pack_int4_to_uint8(q_uint4: torch.Tensor) -> torch.Tensor:
    """Packs {0..15} 4-bit integers into uint8 at 2 weights per byte (4.000 bpw)."""
    q = q_uint4.to(torch.uint8).reshape(-1, 2)
    return (q[:, 0] | (q[:, 1] << 4)).contiguous()


def unpack_int4_from_uint8(packed: torch.Tensor, target_shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=-1).reshape(target_shape).to(dtype)
    return q - 8.0


# ==============================================================================
# 2. REAL CALIBRATION & HELD-OUT TEXT LOADER
# ==============================================================================
def load_disjoint_calib_and_eval_tokens(
    tokenizer: Any,
    seq_len: int = 256,
    num_calib_seqs: int = 48,
    num_eval_seqs: int = 24,
    cache_dir: str = "./rpm_ihog_15bpw_results",
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    os.makedirs(cache_dir, exist_ok=True)
    raw_txt_path = os.path.join(cache_dir, "real_corpus.txt")

    if not os.path.exists(raw_txt_path):
        urls = [
            (
                "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
                "WikiText-2 Validation",
            ),
            (
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                "TinyShakespeare",
            ),
        ]
        downloaded = False
        for url, name in urls:
            try:
                print(f"[*] Downloading real calibration/evaluation text ({name})...")
                with urllib.request.urlopen(url, timeout=15) as resp:
                    text = resp.read().decode("utf-8", errors="ignore")
                if len(text) > 100_000:
                    with open(raw_txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    downloaded = True
                    break
            except Exception as exc:
                print(f"[!] Could not fetch {url}: {exc}")

        if not downloaded:
            raise RuntimeError(f"Unable to download corpus. Place a text file at {raw_txt_path}.")

    with open(raw_txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    total_needed = (num_calib_seqs + num_eval_seqs + 4) * seq_len
    if enc.numel() < total_needed:
        enc = enc.repeat(math.ceil(total_needed / enc.numel()))

    calib_end = num_calib_seqs * seq_len
    eval_start = calib_end + (4 * seq_len)
    eval_end = eval_start + (num_eval_seqs * seq_len)

    calib_tokens = enc[:calib_end].reshape(num_calib_seqs, seq_len).contiguous()
    eval_tokens = enc[eval_start:eval_end].reshape(num_eval_seqs, seq_len).contiguous()
    return calib_tokens, eval_tokens, raw_txt_path


def extract_granite_moe_weights(moe_block: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if hasattr(moe_block, "router"):
        router_mod = moe_block.router
        router_w = router_mod.layer.weight.detach() if hasattr(router_mod, "layer") else router_mod.weight.detach()
    elif hasattr(moe_block, "gate"):
        router_w = moe_block.gate.weight.detach()
    else:
        raise ValueError(f"Cannot locate router weight in MoE block: {type(moe_block)}")

    if hasattr(moe_block, "input_linear") and hasattr(moe_block, "output_linear"):
        w_in = moe_block.input_linear.weight.detach()
        w_out = moe_block.output_linear.weight.detach()
    elif hasattr(moe_block, "experts"):
        exp = moe_block.experts
        if hasattr(exp, "gate_up_proj") and hasattr(exp, "down_proj"):
            w_in = exp.gate_up_proj.detach()
            w_out = exp.down_proj.detach()
        else:
            w_in = torch.stack(
                [torch.cat([e.gate_proj.weight.detach(), e.up_proj.weight.detach()], dim=0) for e in exp], dim=0
            )
            w_out = torch.stack([e.down_proj.weight.detach() for e in exp], dim=0)
    else:
        raise ValueError(f"Cannot locate expert weights in MoE block: {type(moe_block)}")

    top_k = getattr(moe_block, "top_k", getattr(moe_block, "num_experts_per_tok", 8))
    return router_w.float(), w_in.float(), w_out.float(), int(top_k)


# ==============================================================================
# 3. PACKED KRONECKER + HADAMARD RESIDUAL EXPERT TENSOR MODULE
# ==============================================================================
class PackedRPMIHOGExpertTensor(nn.Module):
    r"""
    Represents a 3D expert weight bank W of shape [E, d_out, d_in] with physical `uint8`
    bit-packed residuals in `state_dict()`:
      - `pure_kronecker`   : W_e = ( \sum_r A_{e,r} \otimes B_{e,r} ) * row_s * col_s  (0.51 bpw)
      - `rpm_ihog_1bit`    : Kronecker + 1-Bit Sign Walsh-Hadamard Residual            (1.51 bpw)
      - `rpm_ihog_ternary` : Kronecker + Base-3 Packed Ternary Walsh-Hadamard Residual (2.11 bpw)
      - `v8_2b1b_hadamard` : Kronecker + 2-Bit + 1-Bit Sign Walsh-Hadamard Residual    (3.51 bpw)
      - `int4_group64`     : Packed 4-bit Group-64 Quantization Control                (4.25 bpw)
    """

    def __init__(
        self,
        num_experts: int,
        d_out: int,
        d_in: int,
        mode: str,
        rank: int = 13,
        p1: int = 32,
        q1: int = 32,
        int4_group_size: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_out = d_out
        self.d_in = d_in
        self.mode = mode
        self.rank = rank
        self.p1 = p1
        self.q1 = q1
        self.p2 = d_out // p1
        self.q2 = d_in // q1
        self.int4_group_size = int4_group_size
        total_el = num_experts * d_out * d_in

        if mode == "int4_group64":
            self.register_buffer("packed_q4", torch.zeros(total_el // 2, dtype=torch.uint8))
            num_groups = d_in // int4_group_size
            self.scales = nn.Parameter(torch.ones(num_experts, d_out, num_groups, dtype=torch.float16))
            return

        # Kronecker core parameters (stored in FP16)
        self.A = nn.Parameter(torch.empty(num_experts, rank, p1, q1, dtype=torch.float16))
        self.B = nn.Parameter(torch.empty(num_experts, rank, self.p2, self.q2, dtype=torch.float16))
        self.row_s = nn.Parameter(torch.ones(num_experts, d_out, 1, dtype=torch.float16))
        self.col_s = nn.Parameter(torch.ones(num_experts, 1, d_in, dtype=torch.float16))

        if mode == "rpm_ihog_1bit":
            # 1.000 bpw packed sign bits (8 weights per uint8 byte) + per-row FP16 residual gain
            self.register_buffer("packed_res_1b", torch.zeros(total_el // 8, dtype=torch.uint8))
            self.res_gain = nn.Parameter(torch.ones(num_experts, d_out, 1, dtype=torch.float16))
        elif mode == "rpm_ihog_ternary":
            # 1.600 bpw packed ternary trits (5 weights per uint8 byte) + per-row FP16 residual gain
            packed_len = math.ceil(total_el / 5)
            self.register_buffer("packed_res_tern", torch.zeros(packed_len, dtype=torch.uint8))
            self.res_gain = nn.Parameter(torch.ones(num_experts, d_out, 1, dtype=torch.float16))
        elif mode == "v8_2b1b_hadamard":
            # v8 formula: 2.000 bpw 2-bit grid + 1.000 bpw 1-bit sign residual + per-row gains
            self.register_buffer("packed_res_2b", torch.zeros(total_el // 4, dtype=torch.uint8))
            self.register_buffer("packed_res_1b", torch.zeros(total_el // 8, dtype=torch.uint8))
            self.res_gain_2b = nn.Parameter(torch.ones(num_experts, d_out, 1, dtype=torch.float16))
            self.res_gain_1b = nn.Parameter(torch.ones(num_experts, d_out, 1, dtype=torch.float16))
        elif mode != "pure_kronecker":
            raise ValueError(f"Unknown mode: {mode}")

        # Transient runtime cache for the unpacked/rotated fixed residual during distillation steps
        # (Not registered in state_dict, so never saved to disk; recomputed from packed uint8 on reload!)
        self._cached_w_res_fixed: Optional[torch.Tensor] = None

    @torch.no_grad()
    def init_from_teacher(self, W_teacher: torch.Tensor, act_rms: Optional[torch.Tensor] = None) -> None:
        device = W_teacher.device
        E, d_out, d_in = W_teacher.shape
        if act_rms is None:
            act_rms_v = torch.ones(1, 1, d_in, device=device, dtype=torch.float32)
        elif act_rms.ndim == 1:
            act_rms_v = act_rms.view(1, 1, d_in).float() + 1e-6
        else:
            act_rms_v = act_rms.view(E, 1, d_in).float() + 1e-6

        if self.mode == "int4_group64":
            W_g = W_teacher.float().reshape(E, d_out, d_in // self.int4_group_size, self.int4_group_size)
            max_abs = torch.amax(torch.abs(W_g), dim=-1, keepdim=True) + 1e-8
            scale = max_abs / 7.0
            q = torch.clamp(torch.round(W_g / scale) + 8.0, 0.0, 15.0).to(torch.uint8)
            self.packed_q4.copy_(pack_int4_to_uint8(q))
            self.scales.data.copy_(scale.squeeze(-1).half())
            return

        # 1. Activation-aware Sinkhorn balancing
        W_act = W_teacher.float() * act_rms_v
        row_s = torch.sqrt(torch.mean(W_act ** 2, dim=2, keepdim=True)) + 1e-6
        col_s = torch.sqrt(torch.mean((W_act / row_s) ** 2, dim=1, keepdim=True)) + 1e-6
        W_bal = W_act / (row_s * col_s)

        # 2. Per-expert Kronecker SVD initialization
        K_mat = (
            W_bal.reshape(E, self.p1, self.p2, self.q1, self.q2)
            .permute(0, 1, 3, 2, 4)
            .reshape(E, self.p1 * self.q1, self.p2 * self.q2)
        )
        A_list, B_list = [], []
        for e in range(E):
            uk, sk, vtk = torch.linalg.svd(K_mat[e], full_matrices=False)
            r = self.rank
            sqrt_sk = torch.sqrt(sk[:r])
            a_e = (uk[:, :r] * sqrt_sk.unsqueeze(0)).T.reshape(r, self.p1, self.q1)
            b_e = (vtk[:r, :] * sqrt_sk.unsqueeze(1)).reshape(r, self.p2, self.q2)
            A_list.append(a_e.half())
            B_list.append(b_e.half())

        self.A.data.copy_(torch.stack(A_list, dim=0))
        self.B.data.copy_(torch.stack(B_list, dim=0))
        self.row_s.data.copy_(row_s.half())
        self.col_s.data.copy_((col_s / act_rms_v).half())

        if self.mode == "pure_kronecker":
            return

        # 3. Compute Kronecker residual and rotate via Fast Walsh-Hadamard
        W_k0 = torch.einsum("erij,erkl->eikjl", self.A.float(), self.B.float()).reshape(E, d_out, d_in)
        res0 = W_bal - W_k0

        H_out = get_hadamard_matrix(d_out, device, torch.float32)
        H_in = get_hadamard_matrix(d_in, device, torch.float32)
        res_rot = torch.einsum("oi,eij,pj->eop", H_out, res0, H_in)

        if self.mode == "rpm_ihog_1bit":
            # Optimal Lloyd-Max 1-bit Gaussian quantizer: alpha = E[|res_rot|], signs in {-1, +1}
            alpha_1b = torch.mean(torch.abs(res_rot), dim=2, keepdim=True) + 1e-8
            signs = torch.where(res_rot >= 0, 1.0, -1.0)
            self.packed_res_1b.copy_(pack_1bit_signs_to_uint8(signs))
            self.res_gain.data.copy_(alpha_1b.half())

        elif self.mode == "rpm_ihog_ternary":
            # Optimal Lloyd-Max Ternary Gaussian quantizer: levels {-1.224, 0, +1.224} * std
            r_std = torch.std(res_rot, dim=2, keepdim=True) + 1e-8
            levels_tern = torch.tensor([-1.224, 0.0, 1.224], device=device, dtype=torch.float32)
            idx_tern = torch.argmin(torch.abs(res_rot.unsqueeze(-1) / r_std.unsqueeze(-1) - levels_tern), dim=-1)
            self.packed_res_tern.copy_(pack_ternary_indices_to_uint8(idx_tern))
            self.res_gain.data.copy_(r_std.half())

        elif self.mode == "v8_2b1b_hadamard":
            # v8 2-bit grid + 1-bit sign residual
            r_std = torch.std(res_rot, dim=2, keepdim=True) + 1e-8
            levels_2b = torch.tensor([-1.510, -0.453, 0.453, 1.510], device=device, dtype=torch.float32)
            idx2 = torch.argmin(torch.abs(res_rot.unsqueeze(-1) / r_std.unsqueeze(-1) - levels_2b), dim=-1)
            q1_rot = levels_2b[idx2] * r_std
            res_rem = res_rot - q1_rot
            alpha_1b = torch.mean(torch.abs(res_rem), dim=2, keepdim=True) + 1e-8
            signs_rem = torch.where(res_rem >= 0, 1.0, -1.0)

            self.packed_res_2b.copy_(pack_2bit_indices_to_uint8(idx2))
            self.packed_res_1b.copy_(pack_1bit_signs_to_uint8(signs_rem))
            self.res_gain_2b.data.copy_(r_std.half())
            self.res_gain_1b.data.copy_(alpha_1b.half())

        self._cached_w_res_fixed = None

    @torch.no_grad()
    def _unpack_base_residual(self, device: torch.device) -> torch.Tensor:
        """
        Unpacks the `uint8` bit-packed Hadamard residual from `state_dict()` buffers
        and rotates back to weight space via inverse Walsh-Hadamard transform.
        """
        shape = (self.num_experts, self.d_out, self.d_in)
        H_out = get_hadamard_matrix(self.d_out, device, torch.float32)
        H_in = get_hadamard_matrix(self.d_in, device, torch.float32)

        if self.mode == "rpm_ihog_1bit":
            q_rot = unpack_1bit_signs_from_uint8(self.packed_res_1b, shape, torch.float32)
            return torch.einsum("io,eij,jp->eop", H_out, q_rot, H_in).half()

        elif self.mode == "rpm_ihog_ternary":
            levels_tern = torch.tensor([-1.224, 0.0, 1.224], device=device, dtype=torch.float32)
            idx = unpack_ternary_indices_from_uint8(self.packed_res_tern, shape)
            q_rot = levels_tern[idx]
            return torch.einsum("io,eij,jp->eop", H_out, q_rot, H_in).half()

        raise RuntimeError(f"Unexpected mode in _unpack_base_residual: {self.mode}")

    def synthesize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        device = self.scales.device if self.mode == "int4_group64" else self.A.device
        shape = (self.num_experts, self.d_out, self.d_in)

        if self.mode == "int4_group64":
            q = unpack_int4_from_uint8(self.packed_q4, shape, dtype)
            q_g = q.reshape(self.num_experts, self.d_out, self.d_in // self.int4_group_size, self.int4_group_size)
            w = (q_g * self.scales.to(dtype).unsqueeze(-1)).reshape(shape)
            return w

        W_k = torch.einsum("erij,erkl->eikjl", self.A.to(dtype), self.B.to(dtype)).reshape(shape)
        if self.mode == "pure_kronecker":
            return W_k * self.row_s.to(dtype) * self.col_s.to(dtype)

        if self.mode in ("rpm_ihog_1bit", "rpm_ihog_ternary"):
            if self._cached_w_res_fixed is None:
                self._cached_w_res_fixed = self._unpack_base_residual(device)
            W_res = self.res_gain.to(dtype) * self._cached_w_res_fixed.to(dtype)
            return (W_k + W_res) * self.row_s.to(dtype) * self.col_s.to(dtype)

        if self.mode == "v8_2b1b_hadamard":
            if self._cached_w_res_fixed is None:
                H_out = get_hadamard_matrix(self.d_out, device, torch.float32)
                H_in = get_hadamard_matrix(self.d_in, device, torch.float32)
                levels_2b = torch.tensor([-1.510, -0.453, 0.453, 1.510], device=device, dtype=torch.float32)
                idx2 = unpack_2bit_indices_from_uint8(self.packed_res_2b, shape)
                q2_rot = levels_2b[idx2]
                q1_rot = unpack_1bit_signs_from_uint8(self.packed_res_1b, shape, torch.float32)
                w2 = torch.einsum("io,eij,jp->eop", H_out, q2_rot, H_in).half()
                w1 = torch.einsum("io,eij,jp->eop", H_out, q1_rot, H_in).half()
                self._cached_w_res_fixed = (w2, w1)
            w2, w1 = self._cached_w_res_fixed
            W_res = self.res_gain_2b.to(dtype) * w2.to(dtype) + self.res_gain_1b.to(dtype) * w1.to(dtype)
            return (W_k + W_res) * self.row_s.to(dtype) * self.col_s.to(dtype)

        raise RuntimeError(f"Invalid mode {self.mode}")


class ReplacedGraniteMoEBlock(nn.Module):
    """
    Complete routed GraniteMoE block (`gate` + `up` + `down` + `router`) backed by
    physically packed sub-1-bpw / 1.5-bpw / 2.1-bpw / 3.5-bpw / INT4 representations.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ffn: int,
        top_k: int,
        mode: str,
        rank: int = 13,
        p1: int = 32,
        q1: int = 32,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.top_k = top_k
        self.mode = mode
        self.rank = rank

        self.router_weight = nn.Parameter(torch.empty(num_experts, d_model, dtype=torch.float16), requires_grad=False)

        if mode != "zero_moe_control":
            self.gate_factors = PackedRPMIHOGExpertTensor(num_experts, d_ffn, d_model, mode, rank, p1, q1)
            self.up_factors = PackedRPMIHOGExpertTensor(num_experts, d_ffn, d_model, mode, rank, p1, q1)
            self.down_factors = PackedRPMIHOGExpertTensor(num_experts, d_model, d_ffn, mode, rank, p1, q1)

    @torch.no_grad()
    def init_all_from_teacher(
        self,
        router_w: torch.Tensor,
        w_in_teacher: torch.Tensor,
        w_out_teacher: torch.Tensor,
        X_calib: torch.Tensor,
        Y_calib_teacher: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.router_weight.data.copy_(router_w.half().to(device))
        if self.mode == "zero_moe_control":
            return

        w_gate_t, w_up_t = w_in_teacher.to(device).chunk(2, dim=1)
        w_down_t = w_out_teacher.to(device)

        x_sub = X_calib[: min(4096, X_calib.shape[0])].to(device, dtype=torch.float32)
        y_sub = Y_calib_teacher[: min(4096, Y_calib_teacher.shape[0])].to(device, dtype=torch.float32)
        act_rms_in = torch.sqrt(torch.mean(x_sub ** 2, dim=0)) + 1e-6

        logits = F.linear(x_sub, self.router_weight.float())
        probs = F.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        h_rms_list = []
        for e in range(self.num_experts):
            token_idx, _ = (topk_idx == e).nonzero(as_tuple=True)
            x_e = x_sub.index_select(0, token_idx) if token_idx.numel() > 16 else x_sub[:256]
            h_e = F.silu(F.linear(x_e, w_gate_t[e])) * F.linear(x_e, w_up_t[e])
            h_rms_list.append(torch.sqrt(torch.mean(h_e ** 2, dim=0)) + 1e-6)
        act_rms_mid = torch.stack(h_rms_list, dim=0)

        self.gate_factors.init_from_teacher(w_gate_t, act_rms=act_rms_in)
        self.up_factors.init_from_teacher(w_up_t, act_rms=act_rms_in)
        self.down_factors.init_from_teacher(w_down_t, act_rms=act_rms_mid)

        # Calibrate global output gain on calibration tokens so One-Shot variance is aligned
        y_pred_init = self.forward_routed_moe(x_sub, compute_dtype=torch.float32)
        num = torch.sum(y_pred_init * y_sub).item()
        den = torch.sum(y_pred_init ** 2).item() + 1e-8
        alpha_opt = max(0.2, min(num / den, 5.0))

        if self.mode == "int4_group64":
            self.down_factors.scales.data.copy_((self.down_factors.scales.float() * alpha_opt).half())
        else:
            self.down_factors.row_s.data.copy_((self.down_factors.row_s.float() * alpha_opt).half())

    def count_expert_storage(self) -> Dict[str, float]:
        total_expert_weights = self.num_experts * (3 * self.d_ffn * self.d_model)
        fp16_expert_bytes = total_expert_weights * 2

        if self.mode == "zero_moe_control":
            return {
                "total_dense_expert_weights": int(total_expert_weights),
                "fp16_expert_bytes": int(fp16_expert_bytes),
                "saved_expert_tensors_count": 0,
                "saved_expert_bytes": 0,
                "effective_expert_bpw": 0.0,
                "compression_ratio_vs_fp16": float("inf"),
            }

        saved_bytes = 0
        tensor_count = 0
        for k, tensor in self.state_dict().items():
            if k == "router_weight":
                continue
            tensor_count += 1
            saved_bytes += tensor.numel() * tensor.element_size()

        eff_bpw = (saved_bytes * 8.0) / total_expert_weights
        return {
            "total_dense_expert_weights": int(total_expert_weights),
            "fp16_expert_bytes": int(fp16_expert_bytes),
            "saved_expert_tensors_count": int(tensor_count),
            "saved_expert_bytes": int(saved_bytes),
            "effective_expert_bpw": round(eff_bpw, 4),
            "compression_ratio_vs_fp16": round(16.0 / eff_bpw, 2),
        }

    def forward_routed_moe(self, x_flat: torch.Tensor, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.mode == "zero_moe_control":
            return torch.zeros_like(x_flat)

        x_work = x_flat.to(compute_dtype)
        logits = F.linear(x_work, self.router_weight.to(compute_dtype))
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = (topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)).to(compute_dtype)

        W_gate = self.gate_factors.synthesize(dtype=compute_dtype)
        W_up = self.up_factors.synthesize(dtype=compute_dtype)
        W_down = self.down_factors.synthesize(dtype=compute_dtype)

        out = torch.zeros_like(x_work)
        for e in range(self.num_experts):
            token_idx, k_pos = (topk_idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            x_e = x_work.index_select(0, token_idx)
            h_e = F.silu(F.linear(x_e, W_gate[e])) * F.linear(x_e, W_up[e])
            y_e = F.linear(h_e, W_down[e])
            w_e = topk_probs[token_idx, k_pos].unsqueeze(-1)
            out.index_add_(0, token_idx, w_e * y_e)

        return out.to(x_flat.dtype)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        orig_shape = hidden_states.shape
        x_flat = hidden_states.reshape(-1, self.d_model)
        compute_dtype = torch.float32 if self.training else hidden_states.dtype
        y_flat = self.forward_routed_moe(x_flat, compute_dtype=compute_dtype)
        y = y_flat.reshape(orig_shape)
        if getattr(self, "_return_tuple", False):
            return y, None
        return y


# ==============================================================================
# 4. ACTIVATION CAPTURE, DISTILLATION & RELOAD VERIFICATION
# ==============================================================================
@torch.no_grad()
def capture_real_moe_io(
    model: nn.Module,
    moe_module: nn.Module,
    token_batches: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    capture_outputs: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    inputs_list: List[torch.Tensor] = []
    outputs_list: List[torch.Tensor] = []
    returns_tuple = False

    def hook_fn(module, inp, out):
        nonlocal returns_tuple
        x_in = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu().half()
        inputs_list.append(x_in)
        if isinstance(out, tuple):
            returns_tuple = True
            if capture_outputs:
                outputs_list.append(out[0].detach().reshape(-1, out[0].shape[-1]).cpu().half())
        else:
            if capture_outputs:
                outputs_list.append(out.detach().reshape(-1, out.shape[-1]).cpu().half())

    handle = moe_module.register_forward_hook(hook_fn)
    model.eval()
    try:
        for i in range(0, token_batches.shape[0], batch_size):
            batch = token_batches[i : i + batch_size].to(device)
            _ = model(batch, use_cache=False)
    finally:
        handle.remove()

    X_cat = torch.cat(inputs_list, dim=0)
    Y_cat = torch.cat(outputs_list, dim=0) if capture_outputs else None
    return X_cat, Y_cat, returns_tuple


@torch.no_grad()
def evaluate_moe_layer_on_heldout(
    student_moe: ReplacedGraniteMoEBlock,
    X_eval: torch.Tensor,
    Y_eval_teacher: torch.Tensor,
    device: torch.device,
    eval_chunk_size: int = 2048,
) -> Dict[str, float]:
    student_moe.eval()
    preds = []
    for i in range(0, X_eval.shape[0], eval_chunk_size):
        x_b = X_eval[i : i + eval_chunk_size].to(device, dtype=torch.float32)
        y_b = student_moe.forward_routed_moe(x_b, compute_dtype=torch.float32)
        preds.append(y_b.cpu())

    Y_pred = torch.cat(preds, dim=0).float()
    Y_true = Y_eval_teacher.float()

    mse = F.mse_loss(Y_pred, Y_true).item()
    target_var = torch.var(Y_true).item() + 1e-12
    nmse = mse / target_var
    r2_pct = (1.0 - nmse) * 100.0
    sqnr_db = 10.0 * math.log10((torch.mean(Y_true ** 2).item() + 1e-12) / (mse + 1e-12))
    cos_sim = F.cosine_similarity(Y_pred.flatten(), Y_true.flatten(), dim=0).item()

    return {
        "heldout_mse": round(mse, 6),
        "heldout_nmse": round(nmse, 5),
        "heldout_r2_pct": round(r2_pct, 2),
        "heldout_sqnr_db": round(sqnr_db, 2),
        "heldout_cosine_sim": round(cos_sim, 5),
    }


@torch.no_grad()
def evaluate_full_model_heldout(
    model: nn.Module,
    eval_tokens: torch.Tensor,
    device: torch.device,
    baseline_top1: Optional[torch.Tensor] = None,
    batch_size: int = 4,
) -> Tuple[Dict[str, float], torch.Tensor]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    top1_list = []

    for i in range(0, eval_tokens.shape[0], batch_size):
        batch = eval_tokens[i : i + batch_size].to(device)
        logits = model(batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += shift_labels.numel()
        top1_list.append(shift_logits.argmax(dim=-1).cpu())

    all_top1 = torch.cat(top1_list, dim=0)
    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(mean_nll, 20.0))

    res = {
        "heldout_nll": round(mean_nll, 4),
        "heldout_ppl": round(ppl, 3),
    }
    if baseline_top1 is not None:
        match_pct = (all_top1 == baseline_top1).float().mean().item() * 100.0
        res["heldout_top1_agreement_pct"] = round(match_pct, 2)
    else:
        res["heldout_top1_agreement_pct"] = 100.0

    return res, all_top1


def train_and_verify_student_moe(
    mode: str,
    rank: int,
    router_w: torch.Tensor,
    w_in_teacher: torch.Tensor,
    w_out_teacher: torch.Tensor,
    top_k: int,
    X_calib_student_in: torch.Tensor,
    Y_calib_teacher: torch.Tensor,
    X_eval_student_in: torch.Tensor,
    Y_eval_teacher: torch.Tensor,
    returns_tuple: bool,
    ckpt_path: str,
    device: torch.device,
    steps: int = 250,
    batch_tokens: int = 1024,
    lr: float = 4e-3,
    p1: int = 32,
    q1: int = 32,
) -> Tuple[ReplacedGraniteMoEBlock, Dict[str, Any]]:
    E, two_dffn, d_model = w_in_teacher.shape
    d_ffn = two_dffn // 2

    student = ReplacedGraniteMoEBlock(
        num_experts=E,
        d_model=d_model,
        d_ffn=d_ffn,
        top_k=top_k,
        mode=mode,
        rank=rank,
        p1=p1,
        q1=q1,
    ).to(device)
    student._return_tuple = returns_tuple

    student.init_all_from_teacher(
        router_w=router_w,
        w_in_teacher=w_in_teacher,
        w_out_teacher=w_out_teacher,
        X_calib=X_calib_student_in,
        Y_calib_teacher=Y_calib_teacher,
        device=device,
    )

    oneshot_metrics = evaluate_moe_layer_on_heldout(student, X_eval_student_in, Y_eval_teacher, device)

    t0 = time.perf_counter()
    final_calib_mse = oneshot_metrics["heldout_mse"]

    # Distill continuous factors (A, B, row_s, col_s, res_gain / group scales) on calibration activations
    if mode != "zero_moe_control" and steps > 0:
        student.train()
        for p in student.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

        num_calib_tokens = X_calib_student_in.shape[0]
        target_var = torch.var(Y_calib_teacher.float()).item() + 1e-12
        torch.manual_seed(42)

        for step in range(steps):
            idx = torch.randint(0, num_calib_tokens, (batch_tokens,))
            x_b = X_calib_student_in.index_select(0, idx).to(device, dtype=torch.float32)
            y_t = Y_calib_teacher.index_select(0, idx).to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            y_pred = student.forward_routed_moe(x_b, compute_dtype=torch.float32)
            loss = F.mse_loss(y_pred, y_t) / target_var
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            final_calib_mse = loss.item() * target_var

        with torch.no_grad():
            for _, p in student.named_parameters():
                p.data = p.data.half()

    distill_time_sec = time.perf_counter() - t0

    # Save physical checkpoint (`uint8` packed residuals + `float16` Kronecker factors) to disk
    torch.save(student.state_dict(), ckpt_path)
    disk_file_bytes = os.path.getsize(ckpt_path)

    # Instantiate a brand-new module and reload strictly from the saved disk checkpoint
    # This proves `_cached_w_res_fixed` is NOT needed in the checkpoint and is reconstructed from `uint8`!
    reloaded_student = ReplacedGraniteMoEBlock(
        num_experts=E,
        d_model=d_model,
        d_ffn=d_ffn,
        top_k=top_k,
        mode=mode,
        rank=rank,
        p1=p1,
        q1=q1,
    ).to(device)
    reloaded_student._return_tuple = returns_tuple
    reloaded_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    reloaded_student.load_state_dict(reloaded_state, strict=True)
    reloaded_student.eval()

    storage_audit = reloaded_student.count_expert_storage()
    storage_audit["serialized_ckpt_file_bytes"] = int(disk_file_bytes)

    post_reload_metrics = evaluate_moe_layer_on_heldout(reloaded_student, X_eval_student_in, Y_eval_teacher, device)

    metrics = {
        "mode": mode,
        "rank": rank,
        **storage_audit,
        "oneshot_heldout_r2_pct": oneshot_metrics["heldout_r2_pct"],
        "oneshot_heldout_sqnr_db": oneshot_metrics["heldout_sqnr_db"],
        "final_calib_mse": round(float(final_calib_mse), 6),
        **post_reload_metrics,
        "distill_time_sec": round(distill_time_sec, 2),
    }
    return reloaded_student, metrics


# ==============================================================================
# 5. MAIN RUNNER (SINGLE-LAYER OR MULTI-LAYER CLOSED-LOOP CASCADE)
# ==============================================================================
def run_rpm_ihog_15bpw_suite(
    model_id: str = "ibm-granite/granite-3.1-1b-a400m-base",
    target_layers: List[int] = [4, 8, 12, 16, 20],
    steps: int = 250,
    seq_len: int = 256,
    num_calib_seqs: int = 48,
    num_eval_seqs: int = 24,
    out_dir: str = "./rpm_ihog_15bpw_results",
    force_rerun: bool = False,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    recovery_json_path = os.path.join(out_dir, "recovery_state.json")
    final_md_path = os.path.join(out_dir, "RPM_IHOG_15BPW_REPORT.md")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running RPM-IHOG 1.5-BPW Hadamard Residual Benchmark Suite on {device}")
    print(f"[*] Target Model : {model_id}")
    print(f"[*] Target Layers: {target_layers} | Steps per layer: {steps}")

    if os.path.exists(recovery_json_path) and not force_rerun:
        with open(recovery_json_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("target_layers") == target_layers:
            print(f"[*] Resuming from existing recovery checkpoint: {recovery_json_path}")
        else:
            state = {
                "model_id": model_id,
                "target_layers": target_layers,
                "steps": steps,
                "experiments": {},
            }
    else:
        state = {
            "model_id": model_id,
            "target_layers": target_layers,
            "steps": steps,
            "experiments": {},
        }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    calib_tokens, eval_tokens, corpus_path = load_disjoint_calib_and_eval_tokens(
        tokenizer,
        seq_len=seq_len,
        num_calib_seqs=num_calib_seqs,
        num_eval_seqs=num_eval_seqs,
        cache_dir=out_dir,
    )
    print(
        f"[*] Tokenized real text ({corpus_path}): "
        f"Calib={tuple(calib_tokens.shape)} ({calib_tokens.numel():,} tokens), "
        f"Held-Out Eval={tuple(eval_tokens.shape)} ({eval_tokens.numel():,} tokens)"
    )

    print(f"[*] Loading pretrained model '{model_id}' in FP16...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[*] Evaluating uncompressed FP16 baseline model on held-out text...")
    base_eval, base_top1 = evaluate_full_model_heldout(model, eval_tokens, device, baseline_top1=None)
    state["fp16_baseline"] = base_eval
    print(f"    -> Uncompressed FP16 Held-Out Perplexity: {base_eval['heldout_ppl']} (NLL: {base_eval['heldout_nll']})")

    # The 6 representations comparing Pure 0.5 bpw, TRUE 1.5 bpw Hadamard, 2.1 bpw Ternary, v8 3.5 bpw, and INT4
    experiment_specs = [
        ("zero_moe_control", "zero_moe_control", 0, "0.00_bpw"),
        ("pure_kronecker_0.5bpw", "pure_kronecker", 13, "0.51_bpw"),
        ("rpm_ihog_1.5bpw_1bit_hadamard", "rpm_ihog_1bit", 12, "1.51_bpw (TRUE 1.5 BPW)"),
        ("rpm_ihog_2.1bpw_ternary_hadamard", "rpm_ihog_ternary", 12, "2.11_bpw (Ternary Had)"),
        ("v8_3.5bpw_2bit_plus_1bit_hadamard", "v8_2b1b_hadamard", 13, "3.51_bpw (v8 Formula)"),
        ("int4_group64_control_4.25bpw", "int4_group64", 0, "4.25_bpw (INT4 Control)"),
    ]

    orig_moe_modules = {l_idx: model.model.layers[l_idx].block_sparse_moe for l_idx in target_layers}

    teacher_cache = {}
    for l_idx in target_layers:
        print(f"[*] Caching teacher ground-truth MoE outputs at Layer {l_idx}...")
        X_cal_t, Y_cal_t, ret_tuple = capture_real_moe_io(
            model, orig_moe_modules[l_idx], calib_tokens, device, capture_outputs=True
        )
        X_ev_t, Y_ev_t, _ = capture_real_moe_io(
            model, orig_moe_modules[l_idx], eval_tokens, device, capture_outputs=True
        )
        r_w, w_in, w_out, tk = extract_granite_moe_weights(orig_moe_modules[l_idx])
        teacher_cache[l_idx] = {
            "X_calib_clean": X_cal_t,
            "Y_calib_teacher": Y_cal_t,
            "X_eval_clean": X_ev_t,
            "Y_eval_teacher": Y_ev_t,
            "returns_tuple": ret_tuple,
            "router_w": r_w,
            "w_in": w_in,
            "w_out": w_out,
            "top_k": tk,
        }

    for exp_key, mode_name, rank_val, budget_tag in experiment_specs:
        print(f"\n" + "=" * 82)
        print(f"EXPERIMENT: {exp_key} | Mode={mode_name} | Target={budget_tag}")
        print("=" * 82)

        if exp_key in state["experiments"] and not force_rerun:
            print(f"[*] Found completed result in recovery_state.json for '{exp_key}'; skipping rerun.")
            continue

        for l_idx in target_layers:
            model.model.layers[l_idx].block_sparse_moe = orig_moe_modules[l_idx]

        per_layer_results = {}
        total_saved_expert_bytes = 0
        total_fp16_expert_bytes = 0

        for pos_idx, l_idx in enumerate(target_layers):
            t_data = teacher_cache[l_idx]
            ckpt_file = os.path.join(out_dir, f"ckpt_{exp_key}_layer{l_idx}.pt")

            # Closed-loop multi-layer activation capture:
            # Layer l_idx receives the actual hidden states produced by the already-compressed earlier layers!
            if pos_idx == 0:
                X_cal_in = t_data["X_calib_clean"]
                X_ev_in = t_data["X_eval_clean"]
            else:
                X_cal_in, _, _ = capture_real_moe_io(
                    model, orig_moe_modules[l_idx], calib_tokens, device, capture_outputs=False
                )
                X_ev_in, _, _ = capture_real_moe_io(
                    model, orig_moe_modules[l_idx], eval_tokens, device, capture_outputs=False
                )

            reloaded_block, layer_metrics = train_and_verify_student_moe(
                mode=mode_name,
                rank=rank_val,
                router_w=t_data["router_w"],
                w_in_teacher=t_data["w_in"],
                w_out_teacher=t_data["w_out"],
                top_k=t_data["top_k"],
                X_calib_student_in=X_cal_in,
                Y_calib_teacher=t_data["Y_calib_teacher"],
                X_eval_student_in=X_ev_in,
                Y_eval_teacher=t_data["Y_eval_teacher"],
                returns_tuple=t_data["returns_tuple"],
                ckpt_path=ckpt_file,
                device=device,
                steps=steps,
            )

            model.model.layers[l_idx].block_sparse_moe = reloaded_block
            per_layer_results[f"layer_{l_idx}"] = layer_metrics
            total_saved_expert_bytes += layer_metrics["saved_expert_bytes"]
            total_fp16_expert_bytes += layer_metrics["fp16_expert_bytes"]

            print(
                f"  [Layer {l_idx:2d}] Exact BPW: {layer_metrics['effective_expert_bpw']:.4f} "
                f"({layer_metrics['saved_expert_bytes'] / 1e6:.2f} MB vs {layer_metrics['fp16_expert_bytes'] / 1e6:.2f} MB FP16) | "
                f"One-Shot R2: {layer_metrics['oneshot_heldout_r2_pct']:6.2f}% -> "
                f"Reloaded Distilled R2: {layer_metrics['heldout_r2_pct']:6.2f}% "
                f"(CosSim: {layer_metrics['heldout_cosine_sim']:.5f})"
            )

        full_model_eval, _ = evaluate_full_model_heldout(
            model, eval_tokens, device, baseline_top1=base_top1
        )
        ppl_delta = round(full_model_eval["heldout_ppl"] - base_eval["heldout_ppl"], 3)

        first_layer_m = per_layer_results[f"layer_{target_layers[0]}"]
        exp_summary = {
            "experiment": exp_key,
            "budget_group": budget_tag,
            "mode": mode_name,
            "rank": rank_val,
            "layers_replaced": target_layers,
            "effective_expert_bpw": first_layer_m["effective_expert_bpw"],
            "compression_ratio_vs_fp16": first_layer_m["compression_ratio_vs_fp16"],
            "total_saved_expert_bytes": total_saved_expert_bytes,
            "total_fp16_expert_bytes": total_fp16_expert_bytes,
            "avg_oneshot_heldout_r2_pct": round(
                float(np.mean([m["oneshot_heldout_r2_pct"] for m in per_layer_results.values()])), 2
            ),
            "avg_reloaded_heldout_r2_pct": round(
                float(np.mean([m["heldout_r2_pct"] for m in per_layer_results.values()])), 2
            ),
            "avg_reloaded_heldout_sqnr_db": round(
                float(np.mean([m["heldout_sqnr_db"] for m in per_layer_results.values()])), 2
            ),
            "avg_reloaded_heldout_cosine_sim": round(
                float(np.mean([m["heldout_cosine_sim"] for m in per_layer_results.values()])), 5
            ),
            "full_model_heldout_nll": full_model_eval["heldout_nll"],
            "full_model_heldout_ppl": full_model_eval["heldout_ppl"],
            "full_model_ppl_delta_vs_fp16": ppl_delta,
            "full_model_top1_agreement_pct": full_model_eval["heldout_top1_agreement_pct"],
            "per_layer_details": per_layer_results,
        }

        state["experiments"][exp_key] = exp_summary
        with open(recovery_json_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(
            f"  [+] Full-Model Reloaded PPL ({len(target_layers)} layers replaced): "
            f"{full_model_eval['heldout_ppl']} (Delta: {ppl_delta:+}) | "
            f"Top-1 Agreement: {full_model_eval['heldout_top1_agreement_pct']}%"
        )

    for l_idx in target_layers:
        model.model.layers[l_idx].block_sparse_moe = orig_moe_modules[l_idx]

    write_markdown_summary(state, final_md_path)
    return state


def write_markdown_summary(state: Dict[str, Any], md_path: str) -> None:
    base_ppl = state["fp16_baseline"]["heldout_ppl"]
    base_nll = state["fp16_baseline"]["heldout_nll"]

    lines = [
        "# RPM-IHOG 1.5-BPW Packed Hadamard Residual vs. v8 (3.5-BPW) & INT4 Report",
        "",
        f"- **Model**: `{state['model_id']}`",
        f"- **Replaced Routed MoE Layers**: `{state['target_layers']}` (Complete `gate` + `up` + `down` SwiGLU block)",
        f"- **Uncompressed FP16 Baseline Held-Out Perplexity**: **{base_ppl}** (NLL: `{base_nll}`)",
        f"- **Distillation Steps per Layer**: `{state['steps']}` (Closed-loop sequential multi-layer calibration)",
        "",
        "## Summary Comparison Table (All Checkpoints Physically Bit-Packed in `uint8` & Reloaded from Disk)",
        "",
        "| Representation | Bitrate Group | Exact Expert BPW | Compression vs FP16 | Saved Bytes (Per Layer) | One-Shot Held-Out $R^2$ | Distilled Reloaded $R^2$ | Held-Out CosSim | Full-Model PPL ($\\Delta$) | Top-1 Token Match |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    num_l = max(1, len(state["target_layers"]))
    for exp_key, exp in state["experiments"].items():
        per_layer_mb = (exp["total_saved_expert_bytes"] / num_l) / 1e6
        lines.append(
            f"| `{exp_key}` | `{exp['budget_group']}` | "
            f"**{exp['effective_expert_bpw']} bpw** | "
            f"`{exp['compression_ratio_vs_fp16']}x` | "
            f"`{per_layer_mb:.2f} MB` | "
            f"{exp['avg_oneshot_heldout_r2_pct']}% | "
            f"**{exp['avg_reloaded_heldout_r2_pct']}%** | "
            f"`{exp['avg_reloaded_heldout_cosine_sim']}` | "
            f"**{exp['full_model_heldout_ppl']}** (`{exp['full_model_ppl_delta_vs_fp16']:+}`) | "
            f"**{exp['full_model_top1_agreement_pct']}%** |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[+] Wrote Markdown report to: {os.path.abspath(md_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rigorous 1.5-BPW Hadamard Residual vs. 2.1-BPW Ternary vs. 3.5-BPW v8 vs. INT4 Benchmark"
    )
    parser.add_argument("--model_id", type=str, default="ibm-granite/granite-3.1-1b-a400m-base")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20],
        help="MoE layer indices to replace (default: 5 layers [4, 8, 12, 16, 20] to test multi-layer cascade)",
    )
    parser.add_argument("--steps", type=int, default=250, help="Distillation optimization steps per layer")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--calib_seqs", type=int, default=48)
    parser.add_argument("--eval_seqs", type=int, default=24)
    parser.add_argument("--out_dir", type=str, default="./rpm_ihog_15bpw_results")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    run_rpm_ihog_15bpw_suite(
        model_id=args.model_id,
        target_layers=args.layers,
        steps=args.steps,
        seq_len=args.seq_len,
        num_calib_seqs=args.calib_seqs,
        num_eval_seqs=args.eval_seqs,
        out_dir=args.out_dir,
        force_rerun=args.force_rerun,
    )