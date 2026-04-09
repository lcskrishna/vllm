# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MoE scatter / gather for FlyDSL grouped GEMM (PyTorch, no Triton scatter).

Replicates the layout produced by ``deep_gemm_utils.ep_scatter`` / ``ep_gather``
so ``flydsl_grouped_gemm_contiguous`` and unpermute see identical tensors,
without ``_fwd_kernel_ep_scatter_2`` (which can fault on some ROCm setups).

Layout (same as DeepGEMM contiguous path):
  - Per-expert token counts are rounded up to 128; expert e occupies
    ``starts[e] : starts[e] + cap[e]`` rows in ``M_sum``.
  - Rows ``starts[e] : starts[e] + n[e]`` hold real tokens; padding stays -1
    in ``expert_ids``.
  - ``(t, k)`` assignments follow row-major order over ``topk_ids`` when
    sorting by (expert, flat_index).
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.utils.math_utils import round_up
from vllm.utils.flydsl_grouped_gemm import flydsl_contiguous_mk_alignment


def expert_num_tokens_round_up_and_sum(
    expert_num_tokens: torch.Tensor, alignment: int
) -> int:
    ent = (expert_num_tokens.to(torch.int64) + (alignment - 1)) // alignment * alignment
    return int(torch.sum(ent).item())


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
) -> int:
    if (expert_tokens_meta is not None) and (
        expert_tokens_meta.expert_num_tokens_cpu is not None
    ):
        return expert_num_tokens_round_up_and_sum(
            expert_tokens_meta.expert_num_tokens_cpu, alignment=alignment
        )
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum


def _apply_expert_map_topk(
    topk_ids: torch.Tensor, expert_map: torch.Tensor | None
) -> torch.Tensor:
    if expert_map is None:
        return topk_ids
    out = topk_ids.clone()
    valid = topk_ids >= 0
    out[valid] = expert_map[topk_ids[valid]].to(dtype=out.dtype)
    return out


def _count_expert_num_tokens_torch(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Device-local histogram; avoids Triton count kernel."""
    mapped = _apply_expert_map_topk(topk_ids, expert_map)
    flat = mapped.view(-1)
    valid = (flat >= 0) & (flat < num_local_experts)
    if not valid.any():
        return torch.zeros(num_local_experts, device=topk_ids.device, dtype=torch.int32)
    counts = torch.bincount(flat[valid].long(), minlength=num_local_experts)
    if counts.numel() > num_local_experts:
        counts = counts[:num_local_experts]
    return counts.to(torch.int32)


def flydsl_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    aq_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute FP8 activations into expert-grouped contiguous layout (PyTorch).

    Returns ``(aq_out, aq_scale_out, expert_ids, inv_perm)`` with the same
    contract as ``deepgemm_moe_permute``.
    """
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    device = aq.device
    H = aq.size(1)
    block_m, block_k = flydsl_contiguous_mk_alignment()

    M_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_m,
        expert_tokens_meta=expert_tokens_meta,
    )

    if expert_tokens_meta is not None:
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
    else:
        expert_num_tokens = _count_expert_num_tokens_torch(
            topk_ids, local_num_experts, expert_map
        )

    caps = ((expert_num_tokens.to(torch.int64) + (block_m - 1)) // block_m * block_m).to(
        torch.int32
    )
    total_slots = int(caps.sum().item())
    if total_slots != M_sum:
        raise RuntimeError(
            f"FlyDSL permute: cap sum {total_slots} != M_sum {M_sum}. "
            "Expert token metadata may be inconsistent with topk_ids."
        )

    caps_64 = caps.to(torch.int64)
    starts = (caps_64.cumsum(0) - caps_64).to(torch.int32)

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)
    aq_scale_out = torch.empty(
        (M_sum, H // block_k), device=device, dtype=torch.float32
    )
    expert_ids = torch.full((M_sum,), -1, device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int64)

    mapped = _apply_expert_map_topk(topk_ids, expert_map)
    T, topk = topk_ids.shape
    flat_e = mapped.view(-1)
    p = torch.arange(T * topk, device=device, dtype=torch.long)
    valid = flat_e >= 0

    max_e = flat_e.clamp(min=0).max().long().item() if valid.any() else 0
    big = (max_e + 2) * (T * topk + 1)
    keys = torch.where(
        valid,
        flat_e.long() * (T * topk + 1) + p,
        torch.full_like(p, big, dtype=torch.long),
    )
    order = torch.argsort(keys)
    flat_indices_order = order.long()
    e_ord = flat_e[flat_indices_order]
    good = valid[flat_indices_order]
    fi = flat_indices_order[good]
    e_order = e_ord[good]

    n_assign = fi.numel()
    if n_assign > 0:
        dest_rows = torch.empty(n_assign, device=device, dtype=torch.long)
        t_src = fi // topk
        for e in range(local_num_experts):
            m = e_order == e
            cnt = int(m.sum().item())
            if cnt == 0:
                continue
            inds = torch.where(m)[0]
            se = int(starts[e].item())
            dest_rows[inds] = torch.arange(
                se, se + cnt, device=device, dtype=torch.long
            )

        inv_perm.view(-1)[fi] = dest_rows
        aq_out[dest_rows] = aq[t_src]
        aq_scale_out[dest_rows] = aq_scale[t_src]

    for e in range(local_num_experts):
        n = int(expert_num_tokens[e].item())
        if n <= 0:
            continue
        se = int(starts[e].item())
        expert_ids[se : se + n].fill_(int(e))

    return aq_out, aq_scale_out, expert_ids, inv_perm


def flydsl_moe_unpermute_and_reduce(
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Inverse of permute: weighted gather from grouped rows (PyTorch)."""
    device = a.device
    T, Kdim = output.shape
    topk = topk_ids.shape[1]
    mapped = _apply_expert_map_topk(topk_ids, expert_map)

    t_idx = torch.arange(T, device=device, dtype=torch.long).view(-1, 1).expand(-1, topk)
    valid = mapped >= 0
    flat_t = t_idx.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_src = inv_perm.reshape(-1).long()
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    flat_map = mapped.reshape(-1)

    v = flat_valid & (flat_map >= 0)
    if not v.any():
        output.zero_()
        return

    gathered = a.index_select(0, flat_src[v])
    w_exp = flat_w[v].unsqueeze(1)
    contrib = gathered.to(torch.float32) * w_exp
    dest_t = flat_t[v]
    out_fp32 = torch.zeros((T, Kdim), device=device, dtype=torch.float32)
    dest_exp = dest_t.unsqueeze(1).expand(-1, Kdim)
    out_fp32.scatter_add_(0, dest_exp, contrib)
    output.copy_(out_fp32.to(output.dtype))
