# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-KV-cache-group handling in MoRIIOConnector (hybrid / HMA models).

Hybrid models (interleaved full attention + linear attention) allocate one block
list per KV cache group, so the connector has to carry a tuple of lists instead
of a single list. These tests pin the group-aware plumbing and the guard that
refuses layouts whose transfer path is not implemented yet.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

mori_available = importlib.util.find_spec("mori") is not None

if not (current_platform.is_rocm() and mori_available):
    pytest.skip(
        "MoRIIOs are only available on ROCm with mori package installed",
        allow_module_level=True,
    )

moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
moriio_connector = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector"
)
moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)

normalize_block_id_groups = moriio_common.normalize_block_id_groups
as_wire_block_ids = moriio_common.as_wire_block_ids
MoRIIOConnector = moriio_connector.MoRIIOConnector


def _attn_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )


def _mamba_spec(block_size: int = 16) -> MambaSpec:
    return MambaSpec(
        shapes=((4, 8), (4, 16)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        block_size=block_size,
        page_size_padded=None,
        mamba_type="mamba2",
        num_speculative_blocks=0,
    )


def _mamba_spec_with_mode(mode: str) -> MambaSpec:
    return MambaSpec(
        shapes=((4, 8), (4, 16)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        block_size=16,
        page_size_padded=None,
        mamba_type="mamba2",
        mamba_cache_mode=mode,
        num_speculative_blocks=0,
    )


def _kv_cache_config(*specs) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)
        ],
    )


# --------------------------------------------------------------------------
# block-id shape normalization
# --------------------------------------------------------------------------


def test_flat_block_ids_are_read_as_a_single_group():
    assert normalize_block_id_groups([1, 2, 3]) == ([1, 2, 3],)


def test_grouped_block_ids_are_preserved_per_group():
    assert normalize_block_id_groups(([1, 2], [7])) == ([1, 2], [7])


def test_empty_block_ids_yield_one_empty_group():
    assert normalize_block_id_groups(None) == ([],)
    assert normalize_block_id_groups([]) == ([],)


def test_single_group_stays_flat_on_the_wire():
    """A hybrid-aware prefill leg must stay readable by a pre-HMA peer."""
    assert as_wire_block_ids(([1, 2, 3],)) == [1, 2, 3]


def test_multi_group_is_sent_as_a_list_per_group():
    assert as_wire_block_ids(([1, 2], [5])) == [[1, 2], [5]]


def test_wire_format_round_trips():
    groups = ([1, 2], [5])
    assert normalize_block_id_groups(as_wire_block_ids(groups)) == groups


# --------------------------------------------------------------------------
# HMA declaration
# --------------------------------------------------------------------------


def test_connector_declares_hma_support():
    """Without this, vLLM unifies hybrid specs and fails to promote MambaSpec."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma

    assert supports_hma(MoRIIOConnector)
    assert hasattr(MoRIIOConnector, "request_finished_all_groups")


# --------------------------------------------------------------------------
# local/remote block alignment
# --------------------------------------------------------------------------


def _scheduler_stub(state_group_indices=frozenset()):
    return SimpleNamespace(
        _state_group_indices=state_group_indices,
        _align_local_to_remote_blocks=(
            moriio_connector.MoRIIOConnectorScheduler._align_local_to_remote_blocks
        ),
    )


def _align(local, remote, state_groups=frozenset()):
    stub = _scheduler_stub(state_groups)
    return stub._align_local_to_remote_blocks(stub, local, remote)


def test_equal_length_keeps_local_block_ids():
    assert _align(([1, 2],), ([7, 8],)) == ([1, 2],)


def test_prefix_cache_hit_takes_the_remote_tail():
    """Locally cached blocks mean only the peer's tail still has to be pulled."""
    assert _align(([1],), ([7, 8, 9],)) == ([9],)


def test_recurrent_state_group_is_never_trimmed():
    """One block is the whole state; a suffix of it would be meaningless."""
    aligned = _align(([1], [4]), ([7, 8, 9], [3]), state_groups=frozenset({1}))
    assert aligned == ([9], [3])


def test_group_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="group count mismatch"):
        _align(([1], [2]), ([1],))


def test_local_longer_than_remote_is_rejected():
    with pytest.raises(ValueError, match="longer than remote_block_ids"):
        _align(([1, 2, 3],), ([9],))


# --------------------------------------------------------------------------
# unsupported-layout guard
# --------------------------------------------------------------------------


def _validate(config):
    return moriio_connector._validate_supported_kv_cache_groups(
        config, moriio_layout.recurrent_state_group_indices(config)
    )


def test_single_attention_group_is_supported():
    _validate(_kv_cache_config(_attn_spec()))


def test_multiple_attention_groups_are_supported():
    """Each layer is addressed with its own group's block ids."""
    _validate(_kv_cache_config(_attn_spec(16), _attn_spec(32)))


def test_hybrid_attention_plus_recurrent_state_is_supported():
    _validate(_kv_cache_config(_attn_spec(), _mamba_spec()))


def test_multi_block_mamba_cache_mode_is_rejected():
    """With >1 live state block the post-prefill block is not identified."""
    config = _kv_cache_config(_attn_spec(), _mamba_spec())
    config.kv_cache_groups[1].kv_cache_spec = _mamba_spec_with_mode("align")
    with pytest.raises(NotImplementedError, match="mamba_cache_mode"):
        _validate(config)


# --------------------------------------------------------------------------
# recurrent-state page geometry
# --------------------------------------------------------------------------


def _page_view(num_blocks: int, page_size_bytes: int) -> torch.Tensor:
    """The layout the GPU model runner hands connectors for a mamba layer."""
    return torch.zeros(num_blocks * page_size_bytes, dtype=torch.int8).view(
        num_blocks, 1, 1, page_size_bytes
    )


def test_state_page_is_one_indivisible_transfer_per_block():
    spec = _mamba_spec()
    page = spec.page_size_bytes
    geometry = moriio_layout.get_layer_transfer_geometry(
        "mamba.0", _page_view(6, page), {"mamba.0": spec}
    )

    assert geometry.num_blocks == 6
    assert geometry.block_len == page
    # No K/V split and no per-token slots: the page moves as a unit.
    assert geometry.transfers_per_block == 1
    assert geometry.regions_per_block == 1
    assert geometry.split_kv_regions is False
    assert geometry.slot_size_bytes == page
    assert geometry.local_kv_stride is None
    assert geometry.remote_kv_stride is None


def test_state_page_block_stride_walks_whole_pages():
    spec = _mamba_spec()
    page = spec.page_size_bytes
    geometry = moriio_layout.get_layer_transfer_geometry(
        "mamba.0", _page_view(4, page), {"mamba.0": spec}
    )
    assert geometry.block_stride == page


def test_state_transfer_offsets_move_exactly_the_requested_pages():
    spec = _mamba_spec()
    page = spec.page_size_bytes
    local, remote, sizes = moriio_layout.compute_block_transfer_offsets(
        "mamba.0",
        _page_view(8, page),
        {"mamba.0": spec},
        local_block_ids=[2],
        remote_block_ids=[5],
        remote_num_blocks=8,
    )
    assert (local, remote, sizes) == ([2 * page], [5 * page], [page])


def test_unexpected_state_cache_shape_is_rejected():
    spec = _mamba_spec()
    bad = torch.zeros(4, 2, 1, spec.page_size_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match="page view"):
        moriio_layout.get_layer_transfer_geometry("mamba.0", bad, {"mamba.0": spec})


def test_state_page_size_disagreeing_with_spec_is_rejected():
    spec = _mamba_spec()
    with pytest.raises(ValueError, match="page size mismatch"):
        moriio_layout.get_layer_transfer_geometry(
            "mamba.0", _page_view(4, spec.page_size_bytes + 8), {"mamba.0": spec}
        )


def test_state_registration_covers_every_page():
    spec = _mamba_spec()
    page = spec.page_size_bytes
    regions = moriio_layout.iter_layer_registration_regions(
        "mamba.0", _page_view(5, page), {"mamba.0": spec}
    )
    assert len(regions) == 1
    assert regions[0][1] == 5 * page


# --------------------------------------------------------------------------
# per-layer group selection in the transfer path
# --------------------------------------------------------------------------


def _worker_stub(layer_to_group_index, state_groups=frozenset(), world_size=8):
    return SimpleNamespace(
        layer_to_group_index=layer_to_group_index,
        _state_group_indices=state_groups,
        world_size=world_size,
        _layer_block_ids=(moriio_connector.MoRIIOConnectorWorker._layer_block_ids),
        _validate_hybrid_peer_tp=(
            moriio_connector.MoRIIOConnectorWorker._validate_hybrid_peer_tp
        ),
    )


def test_each_layer_reads_its_own_groups_blocks():
    stub = _worker_stub({"attn.0": 0, "kda.1": 1}, state_groups=frozenset({1}))
    local, remote = ([10, 11], [99]), ([20, 21], [77])

    assert stub._layer_block_ids(stub, "attn.0", local, remote) == ([10, 11], [20, 21])
    assert stub._layer_block_ids(stub, "kda.1", local, remote) == ([99], [77])


def test_layer_falls_back_to_primary_group_for_single_group_peer():
    """A homogeneous peer sends one group; hybrid-local layers must not crash."""
    stub = _worker_stub({"attn.0": 0, "kda.1": 1}, state_groups=frozenset({1}))
    assert stub._layer_block_ids(stub, "kda.1", ([1],), ([2],)) == ([], [])


def test_equal_peer_tp_is_accepted_for_hybrid():
    stub = _worker_stub({"kda.0": 0}, state_groups=frozenset({0}), world_size=8)
    stub._validate_hybrid_peer_tp(stub, 8)


def test_unknown_peer_tp_is_treated_as_homogeneous():
    stub = _worker_stub({"kda.0": 0}, state_groups=frozenset({0}), world_size=8)
    stub._validate_hybrid_peer_tp(stub, 0)
    stub._validate_hybrid_peer_tp(stub, None)


def test_mismatched_peer_tp_is_rejected_for_hybrid():
    """Whole-page state transfer assumes both sides shard the page the same."""
    stub = _worker_stub({"kda.0": 0}, state_groups=frozenset({0}), world_size=8)
    with pytest.raises(NotImplementedError, match="tensor-parallel"):
        stub._validate_hybrid_peer_tp(stub, 4)


def test_mismatched_peer_tp_is_fine_without_recurrent_state():
    stub = _worker_stub({"attn.0": 0}, state_groups=frozenset(), world_size=8)
    stub._validate_hybrid_peer_tp(stub, 4)


def test_group_classification_maps_layers_and_state_groups():
    config = _kv_cache_config(_attn_spec(), _mamba_spec())
    assert moriio_layout.build_layer_to_group_index(config) == {
        "layer.0": 0,
        "layer.1": 1,
    }
    assert moriio_layout.recurrent_state_group_indices(config) == frozenset({1})
