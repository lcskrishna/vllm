# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL grouped GEMM experts (Aiter) for contiguous MoE layout.

Same data flow as the standard FP8 contiguous MoE path (scatter -> grouped GEMM ->
act -> quant -> grouped GEMM -> gather), but grouped matmul uses
``flydsl_grouped_fp8_gemm_nt_contiguous`` (Aiter FlyDSL)

Scatter/gather for this path uses ``flydsl_moe_scatter_gather`` (PyTorch) so we
do not depend on Triton ``ep_scatter`` / ``_fwd_kernel_ep_scatter_2`` (problematic
on some ROCm stacks). Grouped GEMM remains Aiter FlyDSL.
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
from vllm.model_executor.layers.fused_moe.flydsl_moe_scatter_gather import (
    compute_aligned_M,
    flydsl_moe_permute,
    flydsl_moe_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.flydsl_grouped_gemm import (
    flydsl_contiguous_mk_alignment,
    flydsl_grouped_fp8_gemm_nt_contiguous,
    is_flydsl_grouped_gemm_available,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

class FlydslGroupedExperts(mk.FusedMoEExpertsModular):
    """Contiguous-layout FP8 MoE; grouped GEMM via Aiter FlyDSL."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.block_shape == flydsl_contiguous_mk_alignment()
        assert quant_config.quant_dtype == current_platform.fp8_dtype()

    @staticmethod
    def is_supported_config(
        cls: type["mk.FusedMoEExperts"],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        # First, run the base checks (device support, activation, layout)
        supported, reason = super(FlydslGroupedExperts, cls).is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason

        # Second, enforce FlyDSL specific shape constraints
        align_m, align_k = flydsl_contiguous_mk_alignment()
        N = moe_config.intermediate_size_per_partition
        K = moe_config.hidden_dim

        if N % align_m != 0 or K % align_k != 0:
            return False, f"unaligned shapes (N={N}, K={K}, align={align_m})"
        if N <= 512:
            return False, f"N dimension too small (N={N} <= 512)"

        return True, None

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
        block_m = flydsl_contiguous_mk_alignment()[0]
        M_sum = compute_aligned_M(M, topk, local_num_experts, block_m, expert_tokens_meta)
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return ((M_sum, max(activation_out_dim, K)), (M_sum, max(N, K)), (M, K))

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_k = flydsl_contiguous_mk_alignment()[1]
        M_sum, N = input.size()
        if activation == MoEActivation.SILU:
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input, output=output, use_ue8m0=False,
            )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        act_out = torch.empty((M_sum, activation_out_dim), dtype=input.dtype, device=input.device)
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(act_out, block_k, column_major_scales=True, out_q=output)

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
        a1q = hidden_states
        local_num_experts, N, K = w1.size()

        M_sum = compute_aligned_M(
            M=topk_ids.size(0), num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts, alignment=flydsl_contiguous_mk_alignment()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(workspace13.view(dtype=current_platform.fp8_dtype()), (M_sum, K))
        a1q, a1q_scale, expert_ids, inv_perm = flydsl_moe_permute(
            aq=a1q, aq_scale=a1q_scale, topk_ids=topk_ids,
            local_num_experts=local_num_experts, expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta, aq_out=a1q_perm,
        )

        mm1_out = _resize_cache(workspace2, (M_sum, N))
        flydsl_grouped_fp8_gemm_nt_contiguous((a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids)

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        quant_out = _resize_cache(workspace13.view(dtype=current_platform.fp8_dtype()), (M_sum, activation_out_dim))
        a2q, a2q_scale = self._act_mul_quant(input=mm1_out.view(-1, N), output=quant_out, activation=activation)

        mm2_out = _resize_cache(workspace2, (M_sum, K))
        flydsl_grouped_fp8_gemm_nt_contiguous((a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids)

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        flydsl_moe_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
