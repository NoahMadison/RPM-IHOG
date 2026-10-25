#!/usr/bin/env python3
"""
Full-Scale 48-Layer x 512-Expert + 51B N-Gram Overnight Compression Pipeline (v2.1)
===================================================================================
Target Hardware: 32GB Tesla V100 eGPU (use CUDA_VISIBLE_DEVICES=1)
Model: Qwen/Qwen3.8-Flash-Next (180B Total: 125B MoE + 51B N-gram + 4B Shared/MTP)

Fixes in v2.1:
  - Skips 1D norm vectors (e.g. ple.norm_conv.weight) and tiny conv1d weights when
    scanning for N-gram / PLE lookup tables; only INT4-quantizes true 2D embedding
    and projection tables (>= 50 MB, row_dim >= 128).
  - Flattens tensors safely in pack_int4 so odd shapes never trigger AssertionError.
  - Auto-repacks any existing 7.9 GB unpacked layer_XX.pt files down to ~550-650 MB.
"""

import argparse
import concurrent.futures
import datetime
import gc
import json
import math
import os
import struct
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 0. TIMESTAMPED LOGGER & MEMORY HYGIENE
# ==============================================================================
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def clear_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


# ==============================================================================
# 1. BIT-PACKING UTILITIES (4x Compression for 2-bit, 8x for 1-bit Sign)
# ==============================================================================
def pack_2bit(idx: torch.Tensor) -> torch.Tensor:
    """Packs uint8/int8 tensor with values in {0, 1, 2, 3} into uint8 (4 values/byte)."""
    assert idx.shape[-1] % 4 == 0, f"Last dim {idx.shape[-1]} must be divisible by 4"
    orig_shape = idx.shape
    flat = idx.reshape(-1, 4).to(torch.uint8)
    packed = (flat[:, 0] << 6) | (flat[:, 1] << 4) | (flat[:, 2] << 2) | (flat[:, 3] & 0x03)
    new_shape = list(orig_shape[:-1]) + [orig_shape[-1] // 4]
    return packed.reshape(new_shape).contiguous().clone()


def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks uint8 bytes back into indices in {0, 1, 2, 3}."""
    orig_shape = packed.shape
    p_flat = packed.reshape(-1, 1).to(torch.uint8)
    u0 = (p_flat >> 6) & 0x03
    u1 = (p_flat >> 4) & 0x03
    u2 = (p_flat >> 2) & 0x03
    u3 = p_flat & 0x03
    unpacked = torch.cat([u0, u1, u2, u3], dim=-1)
    target_shape = list(orig_shape[:-1]) + [orig_shape[-1] * 4]
    return unpacked.reshape(target_shape)


def pack_1bit_sign(sign_tensor: torch.Tensor) -> torch.Tensor:
    """Packs sign tensor (>0 -> 1, <=0 -> 0) into uint8 (8 values/byte)."""
    assert sign_tensor.shape[-1] % 8 == 0, f"Last dim {sign_tensor.shape[-1]} must be divisible by 8"
    orig_shape = sign_tensor.shape
    bits = (sign_tensor > 0).to(torch.uint8).reshape(-1, 8)
    packed = (
        (bits[:, 0] << 7)
        | (bits[:, 1] << 6)
        | (bits[:, 2] << 5)
        | (bits[:, 3] << 4)
        | (bits[:, 4] << 3)
        | (bits[:, 5] << 2)
        | (bits[:, 6] << 1)
        | bits[:, 7]
    )
    new_shape = list(orig_shape[:-1]) + [orig_shape[-1] // 8]
    return packed.reshape(new_shape).contiguous().clone()


def unpack_1bit_sign(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks uint8 bytes back into {-1.0, +1.0} float32 signs."""
    orig_shape = packed.shape
    p_flat = packed.reshape(-1, 1).to(torch.uint8)
    bits = torch.cat([(p_flat >> (7 - i)) & 0x01 for i in range(8)], dim=-1)
    signs = bits.float() * 2.0 - 1.0
    target_shape = list(orig_shape[:-1]) + [orig_shape[-1] * 8]
    return signs.reshape(target_shape)


def pack_int4(q_uint8: torch.Tensor) -> torch.Tensor:
    """Packs 4-bit unsigned integers [0..15] into uint8 (2 values/byte)."""
    orig_shape = q_uint8.shape
    if orig_shape[-1] % 2 == 0:
        flat = q_uint8.reshape(-1, 2).to(torch.uint8)
        packed = ((flat[:, 0] & 0x0F) << 4) | (flat[:, 1] & 0x0F)
        new_shape = list(orig_shape[:-1]) + [orig_shape[-1] // 2]
        return packed.reshape(new_shape).contiguous().clone()
    else:
        flat_1d = q_uint8.reshape(-1).to(torch.uint8)
        if flat_1d.numel() % 2 != 0:
            flat_1d = torch.cat([flat_1d, torch.zeros(1, dtype=torch.uint8, device=flat_1d.device)])
        pairs = flat_1d.reshape(-1, 2)
        packed = ((pairs[:, 0] & 0x0F) << 4) | (pairs[:, 1] & 0x0F)
        return packed.contiguous().clone()


def tensor_dict_bytes(d: dict) -> int:
    total = 0
    for v in d.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            total += tensor_dict_bytes(v)
    return total


# ==============================================================================
# 2. SUB-BLOCK WALSH-HADAMARD TRANSFORM
# ==============================================================================
def build_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert n > 0 and (n & (n - 1)) == 0, f"Dimension {n} must be a power of 2."
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return (H / math.sqrt(n)).to(dtype)


def apply_block_hadamard(
    X: torch.Tensor, H_b: torch.Tensor, block_size: int, transpose: bool = False
) -> torch.Tensor:
    E, d_out, d_in = X.shape
    n_b_out = d_out // block_size
    n_b_in = d_in // block_size
    x_blk = X.reshape(E, n_b_out, block_size, n_b_in, block_size)
    H = H_b.T if transpose else H_b
    rot = torch.einsum("ij,ebjkl,lm->ebikm", H, x_blk, H.T)
    return rot.reshape(E, d_out, d_in)


# ==============================================================================
# 3. RESILIENT HTTP RANGE-REQUEST SAFETENSORS STREAMER
# ==============================================================================
def fetch_byte_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait_s = 2 ** attempt
            log(f"[!] HTTP Range retry {attempt + 1}/{retries} after error ({e}). Waiting {wait_s}s...")
            time.sleep(wait_s)
    return b""


class HFShardStreamer:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.base_url = f"https://huggingface.co/{model_id}/resolve/main"
        log(f"[*] Fetching remote Safetensors index from {self.base_url}...")
        idx_url = f"{self.base_url}/model.safetensors.index.json"
        req = urllib.request.Request(idx_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.index_data = json.loads(resp.read().decode("utf-8"))
        self.weight_map: Dict[str, str] = self.index_data["weight_map"]
        self.header_cache: Dict[str, Tuple[dict, int]] = {}
        log(f"[*] Index loaded: {len(self.weight_map)} total tensors across shards.")

    def get_header(self, shard_file: str) -> Tuple[dict, int]:
        if shard_file in self.header_cache:
            return self.header_cache[shard_file]
        raw_len = fetch_byte_range(f"{self.base_url}/{shard_file}", 0, 7)
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_hdr = fetch_byte_range(f"{self.base_url}/{shard_file}", 8, 8 + header_len - 1)
        header = json.loads(raw_hdr.decode("utf-8"))
        self.header_cache[shard_file] = (header, header_len)
        return header, header_len

    def find_layer_routed_expert_key(self, layer_idx: int, proj_name: str) -> Tuple[str, str]:
        for k, shard in self.weight_map.items():
            if (
                f"layers.{layer_idx}." in k
                and "shared_expert" not in k
                and "gate.weight" not in k
                and ("experts" in k or "mlp" in k)
                and proj_name in k
            ):
                return k, shard
        raise KeyError(f"Could not locate routed expert tensor for Layer {layer_idx} ({proj_name})")

    def stream_tensor_by_key(self, target_key: str) -> Tuple[torch.Tensor, str, int]:
        shard_file = self.weight_map[target_key]
        header, header_len = self.get_header(shard_file)
        info = header[target_key]
        shape = info["shape"]
        dtype_str = info["dtype"]
        start_off, end_off = info["data_offsets"]
        data_start = 8 + header_len + start_off
        data_end = 8 + header_len + end_off - 1
        raw_bytes_len = data_end - data_start + 1

        raw_bytes = fetch_byte_range(f"{self.base_url}/{shard_file}", data_start, data_end)
        torch_dtype = (
            torch.bfloat16
            if dtype_str == "BF16"
            else (torch.float16 if dtype_str == "F16" else torch.float32)
        )
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype).reshape(shape).float().clone()
        del raw_bytes
        return tensor, target_key, raw_bytes_len

    def stream_layer_routed_expert(self, layer_idx: int, proj_name: str) -> Tuple[torch.Tensor, str, int]:
        target_key, _ = self.find_layer_routed_expert_key(layer_idx, proj_name)
        return self.stream_tensor_by_key(target_key)

    def stream_and_quantize_ngram_tables(
        self, out_dir: str, device: torch.device, group_size: int = 128, row_batch: int = 16384
    ) -> List[dict]:
        """
        Locates genuine 2D N-gram / PLE / Embedding lookup tables (>= 25 MB, 2D shape with
        row_dim >= 128, excluding 1D norms and conv1d kernels), streams them in row batches
        via HTTP Range Requests, and quantizes them on the V100 to packed INT4.
        """
        candidate_keys = [
            k
            for k in self.weight_map.keys()
            if (
                any(tok in k.lower() for tok in ("ngram", "n_gram", "ple", "memory_table", "lookup", "embed_tokens"))
                and "norm" not in k.lower()
                and "conv1d" not in k.lower()
                and "experts" not in k.lower()
            )
        ]

        # Filter to only substantial 2D matrices (>= 25 MB and row_dim >= 128, divisible by 2)
        valid_ngram_keys = []
        for k in candidate_keys:
            shard_file = self.weight_map[k]
            header, _ = self.get_header(shard_file)
            if k not in header:
                continue
            info = header[k]
            shape = info["shape"]
            s_off, e_off = info["data_offsets"]
            raw_mb = (e_off - s_off) / (1024 ** 2)
            if len(shape) >= 2:
                row_dim = int(np.prod(shape[1:]))
                if row_dim >= 128 and row_dim % 2 == 0 and raw_mb >= 25.0:
                    valid_ngram_keys.append(k)

        if not valid_ngram_keys:
            log("[*] No qualifying >=25 MB 2D N-gram/PLE lookup tables matched; proceeding to MoE layers.")
            return []

        log(f"[*] Found {len(valid_ngram_keys)} qualifying N-gram/PLE/Embedding lookup tables to INT4-quantize.")
        ngram_reports = []

        for key in valid_ngram_keys:
            safe_k = key.replace(".", "_").replace("/", "_")
            out_file = os.path.join(out_dir, f"quantized_int4_{safe_k}.pt")
            if os.path.exists(out_file):
                sz_mb = os.path.getsize(out_file) / (1024 ** 2)
                log(f"[*] Quantized N-gram/PLE table '{out_file}' already exists ({sz_mb:.1f} MB). Skipping.")
                ngram_reports.append({"key": key, "file": out_file, "compressed_mb": round(sz_mb, 2)})
                continue

            shard_file = self.weight_map[key]
            header, header_len = self.get_header(shard_file)
            info = header[key]
            shape = info["shape"]
            dtype_str = info["dtype"]
            start_off, end_off = info["data_offsets"]
            total_raw_bytes = end_off - start_off
            raw_mb = total_raw_bytes / (1024 ** 2)

            log(f"[*] Streaming & INT4-Quantizing Lookup Table '{key}' shape={shape} ({raw_mb:.1f} MB)...")
            bytes_per_elem = 2 if dtype_str in ("BF16", "F16") else 4
            torch_dtype = (
                torch.bfloat16
                if dtype_str == "BF16"
                else (torch.float16 if dtype_str == "F16" else torch.float32)
            )

            num_rows = shape[0]
            row_dim = int(np.prod(shape[1:]))
            row_bytes = row_dim * bytes_per_elem
            base_data_start = 8 + header_len + start_off

            packed_chunks = []
            scale_chunks = []
            min_chunks = []
            sqnr_list = []

            t0 = time.perf_counter()
            for r_start in range(0, num_rows, row_batch):
                r_end = min(r_start + row_batch, num_rows)
                b_start = base_data_start + r_start * row_bytes
                b_end = base_data_start + r_end * row_bytes - 1

                raw_chunk = fetch_byte_range(f"{self.base_url}/{shard_file}", b_start, b_end)
                w_chunk = (
                    torch.frombuffer(bytearray(raw_chunk), dtype=torch_dtype)
                    .reshape(r_end - r_start, row_dim)
                    .to(device)
                    .float()
                )
                del raw_chunk

                # Group-wise asymmetric INT4 quantization [0..15]
                g = min(group_size, row_dim)
                while row_dim % g != 0 and g > 2:
                    g //= 2
                w_grp = w_chunk.reshape(-1, g)
                w_min = w_grp.min(dim=1, keepdim=True).values
                w_max = w_grp.max(dim=1, keepdim=True).values
                scale = (w_max - w_min) / 15.0 + 1e-10
                q_int4 = torch.clamp(torch.round((w_grp - w_min) / scale), 0, 15).to(torch.uint8)

                # Dequantize on GPU to check SQNR
                w_deq = (q_int4.float() * scale + w_min).reshape_as(w_chunk)
                sig_p = torch.mean(w_chunk ** 2).item() + 1e-12
                mse_p = torch.mean((w_chunk - w_deq) ** 2).item() + 1e-12
                sqnr_list.append(10.0 * math.log10(sig_p / mse_p))

                packed_chunks.append(pack_int4(q_int4.reshape(r_end - r_start, row_dim)).cpu())
                scale_chunks.append(scale.half().reshape(r_end - r_start, -1).cpu())
                min_chunks.append(w_min.half().reshape(r_end - r_start, -1).cpu())

                del w_chunk, w_grp, q_int4, w_deq
                clear_memory(device)

            ngram_payload = {
                "key": key,
                "orig_shape": shape,
                "group_size": g,
                "packed_int4": torch.cat(packed_chunks, dim=0).contiguous(),
                "scales_fp16": torch.cat(scale_chunks, dim=0).contiguous(),
                "mins_fp16": torch.cat(min_chunks, dim=0).contiguous(),
            }
            tmp_file = out_file + ".tmp"
            torch.save(ngram_payload, tmp_file)
            os.replace(tmp_file, out_file)

            comp_mb = os.path.getsize(out_file) / (1024 ** 2)
            mean_sqnr = float(np.mean(sqnr_list))
            dur = time.perf_counter() - t0
            log(
                f"[+] Saved INT4 Lookup Table '{out_file}': {raw_mb:.1f} MB -> {comp_mb:.1f} MB "
                f"(SQNR={mean_sqnr:.2f} dB) in {dur:.1f}s"
            )
            del ngram_payload, packed_chunks, scale_chunks, min_chunks
            clear_memory(device)

            ngram_reports.append(
                {
                    "key": key,
                    "file": out_file,
                    "raw_mb": round(raw_mb, 2),
                    "compressed_mb": round(comp_mb, 2),
                    "sqnr_db": round(mean_sqnr, 2),
                    "elapsed_sec": round(dur, 2),
                }
            )
        return ngram_reports


# ==============================================================================
# 4. AWQ-KRONECKER + 2-STAGE HADAMARD DISTILLATION WITH ACCURACY STATS
# ==============================================================================
def distill_expert_chunk(
    W_chunk: torch.Tensor,
    rank_k: int = 16,
    p1: int = 16,
    q1: int = 16,
    steps: int = 40,
    device: torch.device = None,
) -> Tuple[dict, dict]:
    """
    Compresses a chunk of experts [E_chunk, d_out, d_in] using Activation-Weighted
    Kronecker SVD + Sub-Block Walsh-Hadamard 2-bit + 1-bit residual + Closed-Loop SiLU
    distillation. Returns bit-packed CPU tensors and accuracy diagnostics.
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
    while (d_out % block_size != 0 or d_in % block_size != 0) and block_size > 16:
        block_size //= 2

    H_block = build_hadamard_matrix(block_size, device, torch.float32)

    # Synthetic heavy-tailed calibration & held-out evaluation activations
    torch.manual_seed(42)
    X_calib = torch.randn(128, d_in, device=device) * 0.2
    X_calib[:, ::32] *= 4.0
    X_eval = torch.randn(64, d_in, device=device) * 0.2
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
        r_std = torch.std(res_rot, dim=2, keepdim=True) + 1e-12
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
        Y_eval_true = F.silu(torch.einsum("nd,eod->eno", X_eval, W_chunk))
        Y_eval_oneshot = F.silu(torch.einsum("nd,eod->eno", X_eval, W_oneshot))
        var_true = torch.var(Y_eval_true) + 1e-12
        oneshot_r2_pct = float((1.0 - torch.mean((Y_eval_true - Y_eval_oneshot) ** 2) / var_true).item() * 100.0)

        Y_calib_target = F.silu(torch.einsum("nd,eod->eno", X_calib, W_chunk))
        del W_oneshot, Y_eval_oneshot, K_mat, uk, sk, vtk, res0, res_rot, res_rem

    # Closed-loop functional SiLU distillation
    A_param = nn.Parameter(A_init.clone())
    B_param = nn.Parameter(B_init.clone())
    scale_mult = nn.Parameter(torch.ones_like(eff_scale))
    res_gain = nn.Parameter(torch.ones(E_chunk, d_out, 1, device=device))

    opt = torch.optim.AdamW([A_param, B_param, scale_mult, res_gain], lr=4e-3, weight_decay=1e-4)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        W_k = torch.einsum("erij,erkl->eikjl", A_param, B_param).reshape(E_chunk, d_out, d_in)
        W_synth = (W_k + res_gain * W_res_fixed) * (eff_scale * scale_mult)
        Y_pred = F.silu(torch.einsum("nd,eod->eno", X_calib, W_synth))
        loss = F.mse_loss(Y_pred, Y_calib_target)
        loss.backward()
        opt.step()

    with torch.no_grad():
        W_k_final = torch.einsum("erij,erkl->eikjl", A_param, B_param).reshape(E_chunk, d_out, d_in)
        final_scale = eff_scale * scale_mult
        W_distilled = (W_k_final + res_gain * W_res_fixed) * final_scale

        # Weight-space SQNR (dB)
        sig_pwr = torch.mean(W_chunk ** 2).item() + 1e-12
        mse_dist = torch.mean((W_chunk - W_distilled) ** 2).item() + 1e-12
        sqnr_db = 10.0 * math.log10(sig_pwr / mse_dist)

        # Functional held-out R^2 and Cosine Similarity
        Y_eval_dist = F.silu(torch.einsum("nd,eod->eno", X_eval, W_distilled))
        distilled_r2_pct = float((1.0 - torch.mean((Y_eval_true - Y_eval_dist) ** 2) / var_true).item() * 100.0)
        distilled_cos_sim = float(
            F.cosine_similarity(Y_eval_true.flatten(), Y_eval_dist.flatten(), dim=0).item()
        )

        # Bit-pack 2-bit base grid and 1-bit sign residual before moving to CPU
        packed_q1 = pack_2bit(idx2).cpu()
        packed_q2_sign = pack_1bit_sign(q2_sign_fixed).cpu()

        comp_chunk = {
            "A": A_param.detach().half().cpu().clone(),
            "B": B_param.detach().half().cpu().clone(),
            "scale": final_scale.detach().half().cpu().clone(),
            "res_gain": res_gain.detach().half().cpu().clone(),
            "res_rot_q1_packed2b": packed_q1,
            "res_rot_q2_sign_packed1b": packed_q2_sign,
            "res_rot_std": r_std.half().cpu().clone(),
            "alpha_1b": alpha_1b.half().cpu().clone(),
            "p1": p1,
            "q1": q1,
            "p2": p2,
            "q2": q2,
            "d_out": d_out,
            "d_in": d_in,
            "block_size": block_size,
            "bit_packed": True,
        }

        chunk_metrics = {
            "base_kron_srank": base_kron_srank,
            "oneshot_r2_pct": oneshot_r2_pct,
            "distilled_r2_pct": distilled_r2_pct,
            "distilled_cos_sim": distilled_cos_sim,
            "weight_sqnr_db": sqnr_db,
        }

    del A_param, B_param, scale_mult, res_gain, opt, W_k_final, W_distilled, W_res_fixed
    return comp_chunk, chunk_metrics


def compress_projection(
    W_raw: torch.Tensor, proj_name: str, device: torch.device, chunk_size: int = 32, steps: int = 40
) -> Tuple[dict, dict]:
    if W_raw.ndim == 2:
        W_raw = W_raw.unsqueeze(0)
    E_total, d1, d2 = W_raw.shape
    raw_bytes = W_raw.numel() * 2  # BF16 reference size (2 bytes/weight)

    chunks = []
    metrics_accum = []

    for c_start in range(0, E_total, chunk_size):
        c_end = min(c_start + chunk_size, E_total)
        W_chunk = W_raw[c_start:c_end].to(device, non_blocking=True).float()
        comp, m = distill_expert_chunk(W_chunk, rank_k=16, p1=16, q1=16, steps=steps, device=device)
        chunks.append(comp)
        metrics_accum.append(m)
        del W_chunk
        clear_memory(device)

    packed_proj = {
        "A": torch.cat([c["A"] for c in chunks], dim=0).contiguous(),
        "B": torch.cat([c["B"] for c in chunks], dim=0).contiguous(),
        "scale": torch.cat([c["scale"] for c in chunks], dim=0).contiguous(),
        "res_gain": torch.cat([c["res_gain"] for c in chunks], dim=0).contiguous(),
        "res_rot_q1_packed2b": torch.cat([c["res_rot_q1_packed2b"] for c in chunks], dim=0).contiguous(),
        "res_rot_q2_sign_packed1b": torch.cat([c["res_rot_q2_sign_packed1b"] for c in chunks], dim=0).contiguous(),
        "res_rot_std": torch.cat([c["res_rot_std"] for c in chunks], dim=0).contiguous(),
        "alpha_1b": torch.cat([c["alpha_1b"] for c in chunks], dim=0).contiguous(),
        "d_out": d1,
        "d_in": d2,
        "p1": chunks[0]["p1"],
        "q1": chunks[0]["q1"],
        "p2": chunks[0]["p2"],
        "q2": chunks[0]["q2"],
        "block_size": chunks[0]["block_size"],
        "bit_packed": True,
    }
    del chunks

    comp_bytes = tensor_dict_bytes(packed_proj)
    total_weights = E_total * d1 * d2
    eff_bpw = (comp_bytes * 8.0) / total_weights

    proj_stats = {
        "projection": proj_name,
        "shape": [E_total, d1, d2],
        "raw_bf16_mb": round(raw_bytes / (1024 ** 2), 2),
        "compressed_mb": round(comp_bytes / (1024 ** 2), 2),
        "compression_ratio": round(raw_bytes / max(comp_bytes, 1), 2),
        "effective_bpw": round(eff_bpw, 3),
        "base_kron_srank": round(float(np.mean([m["base_kron_srank"] for m in metrics_accum])), 2),
        "oneshot_r2_pct": round(float(np.mean([m["oneshot_r2_pct"] for m in metrics_accum])), 2),
        "distilled_r2_pct": round(float(np.mean([m["distilled_r2_pct"] for m in metrics_accum])), 2),
        "distilled_cos_sim": round(float(np.mean([m["distilled_cos_sim"] for m in metrics_accum])), 5),
        "weight_sqnr_db": round(float(np.mean([m["weight_sqnr_db"] for m in metrics_accum])), 2),
    }
    return packed_proj, proj_stats


# ==============================================================================
# 5. AUTO-REPACKER FOR ANY EXISTING 7.9 GB UNPACKED LAYER FILES
# ==============================================================================
def repack_existing_layer_if_needed(layer_file: str) -> Optional[dict]:
    """
    If an existing layer_XX.pt file was saved by the older script with unpacked int8
    'res_rot_q1' (~7.9 GB), converts it in-place to 2-bit packed uint8 and saves it.
    """
    size_mb = os.path.getsize(layer_file) / (1024 ** 2)
    if size_mb < 1500.0:
        return None  # Already compact

    log(f"[*] Detected oversized existing layer file '{layer_file}' ({size_mb:.1f} MB). Bit-packing in place...")
    data = torch.load(layer_file, map_location="cpu", weights_only=False)
    clean_data = {}

    for p_name, comp in data.items():
        if isinstance(comp, dict):
            new_comp = {}
            for k, v in comp.items():
                if k == "res_rot_q1" and isinstance(v, torch.Tensor) and v.dtype == torch.int8:
                    new_comp["res_rot_q1_packed2b"] = pack_2bit(v)
                elif isinstance(v, torch.Tensor):
                    new_comp[k] = v.detach().cpu().contiguous().clone()
                else:
                    new_comp[k] = v
            new_comp["bit_packed"] = True
            clean_data[p_name] = new_comp
        else:
            clean_data[p_name] = comp

    del data
    gc.collect()

    tmp_file = layer_file + ".tmp"
    torch.save(clean_data, tmp_file)
    os.replace(tmp_file, layer_file)
    del clean_data
    gc.collect()

    new_size_mb = os.path.getsize(layer_file) / (1024 ** 2)
    log(f"[+] Repacked '{layer_file}': {size_mb:.1f} MB -> {new_size_mb:.1f} MB!")
    return {"repacked_from_mb": round(size_mb, 2), "repacked_to_mb": round(new_size_mb, 2)}


# ==============================================================================
# 6. MAIN OVERNIGHT ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Overnight Full-Model Stream & Compress for Qwen3.8-Flash-Next")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--total_layers", type=int, default=48)
    parser.add_argument("--chunk_size", type=int, default=32, help="Experts per GPU distillation batch")
    parser.add_argument("--steps", type=int, default=40, help="Closed-loop AdamW distillation steps per chunk")
    parser.add_argument("--out_dir", type=str, default="./compressed_layers")
    parser.add_argument("--out_stats", type=str, default="./flash_next_full48_stats.json")
    parser.add_argument("--quantize_ngram", action="store_true", default=True, help="Also stream & INT4-quantize N-gram table")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required! Run with CUDA_VISIBLE_DEVICES=1 to target your 32GB V100.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log(f"[*] Bound to GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    log(f"[*] Output directory for compressed layer checkpoints: '{args.out_dir}/'")
    log(f"[*] Live JSON stats file: '{args.out_stats}'")

    # Load previous stats if resuming
    master_stats = {
        "model_id": args.model_id,
        "gpu_device": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "ngram_tables": [],
        "layers": {},
        "running_summary": {},
    }
    if os.path.exists(args.out_stats):
        try:
            with open(args.out_stats, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "layers" in loaded:
                    master_stats = loaded
        except Exception:
            pass

    streamer = HFShardStreamer(args.model_id)

    # Step 1: Stream & Quantize qualifying >=25MB 2D N-gram/PLE/Embedding lookup tables
    if args.quantize_ngram:
        ngram_info = streamer.stream_and_quantize_ngram_tables(args.out_dir, device=device)
        master_stats["ngram_tables"] = ngram_info
        with open(args.out_stats, "w") as f:
            json.dump(master_stats, f, indent=2)

    # Step 2: Stream & Compress all 48 MoE Layers
    projections = ["gate_up_proj", "down_proj"]
    download_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    t_pipeline_start = time.perf_counter()

    for l_idx in range(args.total_layers):
        layer_file = os.path.join(args.out_dir, f"layer_{l_idx:02d}.pt")

        if os.path.exists(layer_file):
            repack_info = repack_existing_layer_if_needed(layer_file)
            sz_mb = os.path.getsize(layer_file) / (1024 ** 2)
            log(f"[*] Layer {l_idx:02d} already saved on disk ('{layer_file}', {sz_mb:.1f} MB). Skipping.")
            if str(l_idx) not in master_stats["layers"]:
                master_stats["layers"][str(l_idx)] = {
                    "layer": l_idx,
                    "status": "loaded_from_disk",
                    "saved_disk_mb": round(sz_mb, 2),
                    "repack_info": repack_info,
                }
            with open(args.out_stats, "w") as f:
                json.dump(master_stats, f, indent=2)
            continue

        log("======================================================================")
        log(f"[*] Starting Layer {l_idx:02d}/{args.total_layers - 1}: Streaming routed experts...")
        t_layer_0 = time.perf_counter()

        # Prefetch both projections concurrently over HTTP Range Requests
        download_futures = {
            p: download_pool.submit(streamer.stream_layer_routed_expert, l_idx, p)
            for p in projections
        }

        layer_compressed_payload = {}
        layer_proj_stats = {}
        layer_raw_mb = 0.0
        layer_comp_mb = 0.0

        for p_name in projections:
            t_dl0 = time.perf_counter()
            W_raw, key_name, raw_bytes_len = download_futures[p_name].result()
            dl_sec = time.perf_counter() - t_dl0
            raw_mb = raw_bytes_len / (1024 ** 2)
            log(
                f"  -> Streamed '{key_name}' shape={tuple(W_raw.shape)} "
                f"({raw_mb:.1f} MB in {dl_sec:.1f}s). Distilling on V100..."
            )

            t_c0 = time.perf_counter()
            comp_dict, p_stat = compress_projection(
                W_raw, proj_name=p_name, device=device, chunk_size=args.chunk_size, steps=args.steps
            )
            c_sec = time.perf_counter() - t_c0
            p_stat["tensor_key"] = key_name
            p_stat["download_sec"] = round(dl_sec, 2)
            p_stat["distill_sec"] = round(c_sec, 2)

            layer_compressed_payload[p_name] = comp_dict
            layer_proj_stats[p_name] = p_stat
            layer_raw_mb += p_stat["raw_bf16_mb"]
            layer_comp_mb += p_stat["compressed_mb"]

            log(
                f"  [+] '{p_name}' compressed: {p_stat['raw_bf16_mb']:.1f} MB -> {p_stat['compressed_mb']:.1f} MB "
                f"({p_stat['compression_ratio']}x, {p_stat['effective_bpw']} bpw) | "
                f"OneShot R^2={p_stat['oneshot_r2_pct']:.2f}% -> Distilled R^2={p_stat['distilled_r2_pct']:.2f}% "
                f"(CosSim={p_stat['distilled_cos_sim']:.5f}, SQNR={p_stat['weight_sqnr_db']:.2f} dB) in {c_sec:.1f}s"
            )

            # Immediately purge uncompressed projection tensor from host RAM & GPU VRAM
            del W_raw, comp_dict
            clear_memory(device)

        # Write only the compressed layer dictionary to disk (atomic write via .tmp)
        tmp_layer_file = layer_file + ".tmp"
        torch.save(layer_compressed_payload, tmp_layer_file)
        os.replace(tmp_layer_file, layer_file)
        disk_mb = os.path.getsize(layer_file) / (1024 ** 2)

        # Free compressed payload from host RAM immediately after disk flush
        del layer_compressed_payload
        clear_memory(device)

        layer_dur = time.perf_counter() - t_layer_0
        mean_r2 = float(np.mean([s["distilled_r2_pct"] for s in layer_proj_stats.values()]))
        mean_cos = float(np.mean([s["distilled_cos_sim"] for s in layer_proj_stats.values()]))
        mean_sqnr = float(np.mean([s["weight_sqnr_db"] for s in layer_proj_stats.values()]))
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        layer_summary = {
            "layer": l_idx,
            "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_layer_sec": round(layer_dur, 2),
            "raw_bf16_mb": round(layer_raw_mb, 2),
            "compressed_ram_mb": round(layer_comp_mb, 2),
            "saved_disk_mb": round(disk_mb, 2),
            "compression_ratio": round(layer_raw_mb / max(disk_mb, 1e-6), 2),
            "mean_distilled_r2_pct": round(mean_r2, 2),
            "mean_distilled_cos_sim": round(mean_cos, 5),
            "mean_weight_sqnr_db": round(mean_sqnr, 2),
            "vram_peak_mb": round(vram_peak_mb, 1),
            "projections": layer_proj_stats,
        }
        master_stats["layers"][str(l_idx)] = layer_summary

        # Update running global summary across all completed layers
        done_layers = [v for v in master_stats["layers"].values() if "mean_distilled_r2_pct" in v]
        total_disk_gb = sum(
            os.path.getsize(os.path.join(args.out_dir, f))
            for f in os.listdir(args.out_dir)
            if f.endswith(".pt") and f.startswith("layer_")
        ) / (1024 ** 3)

        if done_layers:
            master_stats["running_summary"] = {
                "layers_completed": len(master_stats["layers"]),
                "total_target_layers": args.total_layers,
                "total_moe_disk_footprint_gb": round(total_disk_gb, 3),
                "fits_in_32gb_v100": total_disk_gb <= 29.5,
                "overall_mean_distilled_r2_pct": round(
                    float(np.mean([x["mean_distilled_r2_pct"] for x in done_layers])), 3
                ),
                "overall_mean_cosine_sim": round(
                    float(np.mean([x["mean_distilled_cos_sim"] for x in done_layers])), 5
                ),
                "overall_mean_sqnr_db": round(
                    float(np.mean([x["mean_weight_sqnr_db"] for x in done_layers])), 2
                ),
                "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        with open(args.out_stats, "w") as f:
            json.dump(master_stats, f, indent=2)

        log(
            f"[+] Layer {l_idx:02d} saved to '{layer_file}' ({disk_mb:.1f} MB on disk, "
            f"{layer_summary['compression_ratio']}x compression) | "
            f"Layer R^2={mean_r2:.2f}%, CosSim={mean_cos:.5f} | "
            f"Total MoE Disk So Far={total_disk_gb:.2f} GB | Elapsed={layer_dur:.1f}s"
        )

    download_pool.shutdown(wait=True)
    total_min = (time.perf_counter() - t_pipeline_start) / 60.0
    log("======================================================================")
    log(f"[+] ALL {args.total_layers} LAYERS STREAMED & COMPRESSED IN {total_min:.2f} MINUTES!")
    log(f"[+] Final Summary Stats written to '{args.out_stats}'")
    log("======================================================================")


if __name__ == "__main__":
    main()
