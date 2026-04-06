# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL grouped contiguous FP8 GEMM (Aiter), DeepGEMM-compatible I/O."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch

_FLYDSL_MK = 128
_flydsl_grouped_contiguous_impl: Callable[..., Any] | None | bool = False


def _lazy_init_flydsl() -> None:
    global _flydsl_grouped_contiguous_impl
    if _flydsl_grouped_contiguous_impl is not False:
        return
    try:
        mod = importlib.import_module("aiter.ops.flydsl.grouped_gemm_kernels")
        _flydsl_grouped_contiguous_impl = getattr(
            mod, "flydsl_grouped_gemm_contiguous", None
        )
    except Exception:
        _flydsl_grouped_contiguous_impl = None


def is_flydsl_grouped_gemm_available() -> bool:
    _lazy_init_flydsl()
    return _flydsl_grouped_contiguous_impl is not None


def flydsl_grouped_fp8_gemm_nt_contiguous(
    a_pair: tuple[torch.Tensor, torch.Tensor],
    b_pair: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    expert_ids: torch.Tensor,
    **kwargs: Any,
) -> None:
    """Match DeepGEMM ``m_grouped_fp8_gemm_nt_contiguous`` I/O; FlyDSL compute."""
    _lazy_init_flydsl()
    impl = _flydsl_grouped_contiguous_impl
    if impl is None:
        raise RuntimeError(
            "FlyDSL grouped GEMM is not available. Install aiter with "
            "``aiter.ops.flydsl.grouped_gemm_kernels.flydsl_grouped_gemm_contiguous``."
        )
    kwargs.pop("disable_ue8m0_cast", None)
    a_fp8, a_scale = a_pair
    b_fp8, b_scale = b_pair
    scale_a = a_scale.transpose(0, 1).contiguous()
    # FlyDSL / ROCm kernels expect dense row-major device tensors; avoid
    # non-contiguous views from workspace dtype reinterpretation.
    a_fp8 = a_fp8.contiguous()
    b_fp8 = b_fp8.contiguous()
    b_scale = b_scale.contiguous()
    expert_ids = expert_ids.contiguous()
    out = out.contiguous()
    result = impl(
        a_fp8,
        b_fp8,
        scale_a,
        b_scale,
        expert_ids,
        out=out,
        tile_m=_FLYDSL_MK,
        tile_n=_FLYDSL_MK,
        tile_k=_FLYDSL_MK,
        scale_block_k=_FLYDSL_MK,
        scale_block_n=_FLYDSL_MK,
        out_dtype="bf16",
    )
    if result is not None and result is not out:
        out.copy_(result)


def flydsl_contiguous_mk_alignment() -> list[int]:
    """Block alignment for FlyDSL grouped contiguous GEMM (matches DeepGEMM 128)."""
    return [_FLYDSL_MK, _FLYDSL_MK]
