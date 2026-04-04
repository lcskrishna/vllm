# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL grouped GEMM experts (Aiter) for contiguous MoE layout.

Same data flow as the standard FP8 contiguous MoE path (scatter -> grouped GEMM ->
act -> quant -> grouped GEMM -> gather), but grouped matmul uses
``flydsl_grouped_fp8_gemm_nt_contiguous`` (Aiter FlyDSL)

EP scatter/gather helpers live in ``deep_gemm_utils`` (Triton); the name is
historical and does not require the DeepGEMM library at runtime.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT
from vllm.utils.flydsl_grouped_gemm import (
    flydsl_contiguous_mk_alignment,
    flydsl_grouped_fp8_gemm_nt_contiguous,
    is_flydsl_grouped_gemm_available,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _valid_flydsl_grouped_gemm_shape(M: int, N: int, K: int) -> bool:
    """Same tiling constraints as contiguous grouped GEMM (128-wide)."""
    align = flydsl_contiguous_mk_alignment()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_flydsl_grouped_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    if not is_flydsl_grouped_gemm_available():
        logger.debug_once("FlyDSL grouped GEMM: aiter kernel not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()
    if not _valid_flydsl_grouped_gemm_shape(M, N, K):
        logger.debug_once(
            "FlyDSL grouped GEMM disabled due to unaligned problem size. "
            "M: %s, N: %s, K: %s.",
            M,
            N,
            K,
        )
        return False
    elif N <= 512:
        logger.debug_once(
            "FlyDSL grouped GEMM disabled for N <= 512. M: %s, N: %s, K: %s.",
            M,
            N,
            K,
        )
        return False

    if w1.dtype != current_platform.fp8_dtype() or w2.dtype != torch.float8_e4m3fn:
        logger.debug_once(
            "FlyDSL grouped GEMM disabled: invalid weight dtype(s). "
            "w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "FlyDSL grouped GEMM disabled: weights or activations not contiguous.",
        )
        return False

    return True


class FlydslGroupedExperts(mk.FusedMoEExpertsModular):
    """Contiguous-layout FP8 MoE; grouped GEMM via Aiter FlyDSL (not ``deep_gemm``)."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.block_shape == tuple(flydsl_contiguous_mk_alignment())
        assert quant_config.quant_dtype == current_platform.fp8_dtype()
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_flydsl_grouped_gemm_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.block_shape is not None
        block_m = self.block_shape[0]
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % block_m == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.block_shape is not None
        block_k = self.block_shape[1]
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        ## may not be needed.
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        if activation == MoEActivation.SILU:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
            )

        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, block_k, column_major_scales=True, out_q=output
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        # ``deep_gemm`` _lazy_init may never run on ROCm; scale-format oracle still
        # must be initialized for ``from_oracle()`` in _act_mul_quant.
        DeepGemmQuantScaleFMT.init_oracle_cache()

        a1q = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=flydsl_contiguous_mk_alignment()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=current_platform.fp8_dtype()), (M_sum, K)
        )
        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )
        assert a1q.size(0) == M_sum

        mm1_out = _resize_cache(workspace2, (M_sum, N))
        flydsl_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids
        )

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        quant_out = _resize_cache(
            workspace13.view(dtype=current_platform.fp8_dtype()), (M_sum, activation_out_dim)
        )
        a2q, a2q_scale = self._act_mul_quant(
            input=mm1_out.view(-1, N), output=quant_out, activation=activation
        )

        mm2_out = _resize_cache(workspace2, (M_sum, K))
        flydsl_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class TritonOrFlydslGroupedExperts(FallbackExperts):
    """FlyDSL grouped GEMM with Triton fallback for unsupported shapes."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(
            experts=FlydslGroupedExperts(moe_config, quant_config),
            fallback_experts=TritonExperts(moe_config, quant_config),
        )

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular],
        type[mk.FusedMoEExpertsModular],
    ]:
        return (FlydslGroupedExperts, TritonExperts)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if is_flydsl_grouped_gemm_available() and _valid_flydsl_grouped_gemm_shape(
            M, N, K
        ):
            return self.experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        return self.fallback_experts.workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEExpertsModular:
        if _valid_flydsl_grouped_gemm(hidden_states, w1, w2):
            return self.experts
        return self.fallback_experts

