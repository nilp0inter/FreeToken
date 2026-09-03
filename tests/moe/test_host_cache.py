import threading

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.moe.host_cache import (
    HostExpertCache,
    NvmeExpertStore,
    parse_ram_cache_size,
    resolve_ram_cache_plan,
)


def _make_ftw(tmp_path, *, layered=False):
    path = tmp_path / "weights.ftw"
    writer = FTWWriter(str(path))
    gate_up = torch.arange(6 * 2 * 2, dtype=torch.float32).reshape(6, 2, 2)
    down = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2, 1) + 1000
    if layered:
        for layer_id in range(2):
            row = slice(layer_id * 3, (layer_id + 1) * 3)
            writer.add_tensor(f"gate_up#L{layer_id:05d}", gate_up[row], kind="experts_bank")
            writer.add_tensor(f"down#L{layer_id:05d}", down[row], kind="experts_bank")
    else:
        writer.add_tensor("gate_up", gate_up, kind="experts_bank")
        writer.add_tensor("down", down, kind="experts_bank")
    writer.finalize(
        {
            "quant_format": "bf16",
            "expert_bank_num_layers": 2,
        }
    )
    return path, gate_up, down

def _budget(store, capacity):
    return (
        store.layer_storage_bytes
        + (capacity + 1) * store.slot_storage_bytes
    )


def test_parse_and_resolve_ram_cache_size():
    assert parse_ram_cache_size("1.5G") == 1.5 * (1 << 30)
    assert parse_ram_cache_size("all") == "all"
    assert parse_ram_cache_size("auto") == "auto"
    with pytest.raises(ValueError):
        parse_ram_cache_size("0")

    plan = resolve_ram_cache_plan(
        "auto",
        tp_size=2,
        minimum_bytes=32,
        available_bytes=100,
        reserve_bytes=8,
    )
    assert plan.host_budget_bytes == 92
    assert plan.per_rank_budget_bytes == 46
    assert plan.reserve_bytes == 8


def test_nvme_store_reads_unaligned_expert_rows(tmp_path):
    path, gate_up, down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        handle = cache.get(1, 2)
        assert torch.equal(handle.tensors["gate_up"], gate_up[5])
        assert torch.equal(handle.tensors["down"], down[5])
        handle.release()
        status = cache.status()
        assert status["capacity"] == 1
        assert status["warm_experts"] == 1
        assert status["disk_reads"] == 1
    finally:
        cache.close()


def test_nvme_store_reads_per_layer_expert_rows(tmp_path):
    path, gate_up, down = _make_ftw(tmp_path, layered=True)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        handle = cache.get(1, 1)
        assert torch.equal(handle.tensors["gate_up"], gate_up[4])
        assert torch.equal(handle.tensors["down"], down[4])
        handle.release()
    finally:
        cache.close()

def test_host_cache_reads_complete_layer_into_fixed_staging(tmp_path):
    path, gate_up, down = _make_ftw(tmp_path, layered=True)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        staging = cache.load_layer(1)
        assert torch.equal(staging["gate_up"], gate_up[3:])
        assert torch.equal(staging["down"], down[3:])
        status = cache.status()
        assert status["disk_reads"] == 3
        assert status["disk_bytes"] == store.logical_bytes_per_expert * 3
    finally:
        cache.close()


def test_host_cache_frequency_admission_and_eviction(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        cache.get(0, 0).release()
        cache.get(0, 0).release()
        bypass = cache.get(0, 1)
        bypass.release()
        after_bypass = cache.status()
        assert after_bypass["bypasses"] == 1
        assert after_bypass["evictions"] == 0

        cache.record_accesses(0, [1, 1, 1])
        cache.get(0, 1).release()
        status = cache.status()
        assert status["evictions"] == 1
        assert status["warm_experts"] == 1
        assert status["disk_reads"] == 3
    finally:
        cache.close()

def test_gpu_hits_keep_their_ram_copy_recent(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 2), pin=False)
    try:
        cache.get(0, 0).release()
        cache.get(0, 1).release()
        cache.record_accesses(0, [0])
        cache.record_accesses(0, [2, 2])
        cache.get(0, 2, observe=False).release()
        hits = cache.status()["hits"]
        cache.get(0, 0, observe=False).release()
        assert cache.status()["hits"] == hits + 1
    finally:
        cache.close()

def test_in_flight_slot_is_not_evicted(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        handle = cache.get(0, 0)
        cache.mark_in_flight(handle)
        bypass = cache.get(0, 1)
        bypass.release()
        assert cache.status()["evictions"] == 0

        cache.release_in_flight(handle)
        cache.record_accesses(0, [1, 1])
        cache.get(0, 1, observe=False).release()
        assert cache.status()["evictions"] == 1
    finally:
        cache.close()



def test_host_cache_loads_decode_misses_concurrently(tmp_path, monkeypatch):
    path, gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 2), pin=False)
    original = store.read_expert
    barrier = threading.Barrier(2)

    def synchronized_read(layer_id, expert_id, destinations):
        barrier.wait(timeout=2)
        return original(layer_id, expert_id, destinations)

    monkeypatch.setattr(store, "read_expert", synchronized_read)
    try:
        handles, bypass_ids = cache.get_many(0, [0, 1])
        assert bypass_ids == []
        assert torch.equal(handles[0].tensors["gate_up"], gate_up[0])
        assert torch.equal(handles[1].tensors["gate_up"], gate_up[1])
        for handle in handles.values():
            cache.release_in_flight(handle)
        assert cache.status()["disk_reads"] == 2
    finally:
        cache.close()


def test_host_cache_resize_reallocates_fixed_capacity(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        cache.get(0, 0).release()
        assert cache.status()["warm_experts"] == 1
        target = _budget(store, 2)
        cache.resize(
            target,
            host_budget_bytes=target * 2,
            requested=target * 2,
        )
        status = cache.status()
        assert status["capacity"] == 2
        assert status["warm_experts"] == 0
        assert status["hits"] == 0
        assert status["requested_bytes"] == target * 2
        assert cache.validate_per_rank_budget(1 << 40) == 6
    finally:
        cache.close()

def test_bounded_offload_cache_eagerly_maps_gpu_slots(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, gate_up, down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 1), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        cache.materialize_layer(0)
        assert cache.num_indices.item() == 3
        assert cache.slot_for_id[0].tolist() == [0, 1, 2]
        assert torch.equal(cache.bank_caches["gate_up"][0], gate_up[0])
        assert torch.equal(cache.bank_caches["down"][2], down[2])
        cache.copy_missing()
        expert_ids = torch.tensor([2, 0], dtype=torch.int32)
        cache.ensure_experts(1, expert_ids)
        assert expert_ids.min().item() >= 0
        assert cache.slot_for_id[0].tolist() == [-1, -1, 2]
        for expert_id in (2, 0):
            slot = int(cache.slot_for_id[1, expert_id].item())
            assert torch.equal(cache.bank_caches["gate_up"][slot], gate_up[3 + expert_id])
            assert torch.equal(cache.bank_caches["down"][slot], down[3 + expert_id])
        cache.copy_missing()
    finally:
        host_cache.close()
