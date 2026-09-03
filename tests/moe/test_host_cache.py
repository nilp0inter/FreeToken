from collections import Counter

import threading

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.moe.host_cache import (
    CacheTelemetry,
    ExpertResidencyDirectory,
    ExpertIoCoordinator,
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

def _make_ftw_pages(tmp_path, *, devices=()):
    path = tmp_path / "weights-pages.ftw"
    writer = FTWWriter(str(path), expert_devices=devices)
    gate_up = torch.arange(6 * 2 * 2, dtype=torch.float32).reshape(6, 2, 2)
    down = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2, 1) + 1000
    for layer_id in range(2):
        rows = slice(layer_id * 3, (layer_id + 1) * 3)
        writer.add_expert_layer(
            layer_id,
            {"gate_up": gate_up[rows], "down": down[rows]},
        )
    index = writer.finalize(
        {
            "quant_format": "bf16",
            "expert_bank_num_layers": 2,
        }
    )
    return path, gate_up, down, index

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


def test_cost_aware_admission_prefers_expensive_reused_expert(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        cache.get(0, 0).release()
        cache._frequency[(0, 0)] = 4
        cache._miss_cost_ns[(0, 0)] = 100
        cache._frequency[(0, 1)] = 20
        cache._miss_cost_ns[(0, 1)] = 1
        cache.get(0, 1, observe=False).release()
        assert (0, 0) in cache._map

        cache._miss_cost_ns[(0, 1)] = 100
        cache.get(0, 1, observe=False).release()
        assert (0, 1) in cache._map
    finally:
        cache.close()


def test_dense_layer_scan_does_not_change_admission_state(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path, layered=True)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(
        store,
        _budget(store, 1) + store.layer_storage_bytes,
        pin=False,
    )
    try:
        cache.get(0, 0).release()
        cache.get(0, 0).release()
        before_map = dict(cache._map)
        before_frequency = dict(cache._frequency)
        cache.load_layer(1)
        assert cache._map == before_map
        assert cache._frequency == before_frequency
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
    host_cache = HostExpertCache(store, _budget(store, 3), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        cache.ensure_experts(
            0, torch.tensor([0, 1, 2], dtype=torch.int32), phase="prefill"
        )
        assert cache.num_indices.item() == 0
        assert cache.slot_for_id[0].tolist() == [0, 1, 2]
        assert torch.equal(cache.bank_caches["gate_up"][0], gate_up[0])
        assert torch.equal(cache.bank_caches["down"][2], down[2])
        assert host_cache.status()["gpu_pinned_experts"] == 3
        cache.copy_missing()
        expert_ids = torch.tensor([2, 0], dtype=torch.int32)
        cache.ensure_experts(1, expert_ids)
        assert expert_ids.min().item() >= 0
        assert cache.slot_for_id[0].tolist() == [0, -1, -1]
        for expert_id in (2, 0):
            slot = int(cache.slot_for_id[1, expert_id].item())
            assert torch.equal(cache.bank_caches["gate_up"][slot], gate_up[3 + expert_id])
            assert torch.equal(cache.bank_caches["down"][slot], down[3 + expert_id])
        residency = host_cache.residency.snapshot()
        assert all(record["ram_slot"] is not None for record in residency)
        assert all(
            record["ram_slot"] is not None
            for record in residency
            if record["gpu_slot"] is not None
        )
        slot_bytes = [
            {
                name: bank.tensor.clone()
                for name, bank in host_cache._slots[slot_id].banks.items()
            }
            for slot_id in range(host_cache.capacity)
        ]
        disk_reads = host_cache.status()["disk_reads"]
        cache.reset()
        status = host_cache.status()
        assert status["gpu_pinned_experts"] == 0
        assert status["warm_experts"] == 3
        assert status["disk_reads"] == disk_reads
        for slot_id, expected in enumerate(slot_bytes):
            for name, tensor in expected.items():
                assert torch.equal(
                    host_cache._slots[slot_id].banks[name].tensor, tensor
                )
        cache.copy_missing()
    finally:
        host_cache.close()


def test_cache_telemetry_attributes_tiers_phases_and_layers():
    telemetry = CacheTelemetry()
    telemetry.record("prefill", 1, "nvme", "misses", nbytes=64)
    telemetry.record("decode", 2, "ram", "hits", nbytes=32)
    telemetry.record("prefetch", 3, "vram", "admissions", nbytes=16)

    rows = {
        (row["phase"], row["layer"], row["tier"]): row
        for row in telemetry.snapshot()["rows"]
    }
    assert rows[("prefill", 1, "nvme")]["misses"] == 1
    assert rows[("prefill", 1, "nvme")]["bytes"] == 64
    assert rows[("decode", 2, "ram")]["hits"] == 1
    assert rows[("prefetch", 3, "vram")]["admissions"] == 1


def test_cache_telemetry_unions_overlapping_dependency_intervals():
    telemetry = CacheTelemetry()
    telemetry.record(
        "decode", 0, "nvme", "requests", started_ns=100, finished_ns=300
    )
    telemetry.record(
        "decode", 0, "vram", "requests", started_ns=200, finished_ns=400
    )

    status = telemetry.snapshot()
    assert status["critical_path_ns"] == 300
    assert status["wait_ns"] == {"nvme": 200, "vram": 200}


def test_cache_telemetry_trace_is_bounded_and_opt_in():
    telemetry = CacheTelemetry(trace_capacity=2)
    telemetry.record(
        "decode", 0, "ram", "hits", started_ns=0, finished_ns=1
    )
    assert telemetry.snapshot()["trace"] == []
    telemetry.set_tracing(True)
    for expert_id in range(3):
        telemetry.record(
            "decode",
            0,
            "ram",
            "hits",
            started_ns=expert_id,
            finished_ns=expert_id + 1,
            expert_id=expert_id,
        )
    assert [item["expert"] for item in telemetry.snapshot()["trace"]] == [1, 2]
    telemetry.set_tracing(False)
    assert telemetry.snapshot()["tracing"] is False


def test_residency_directory_rejects_stale_gpu_publication():
    directory = ExpertResidencyDirectory()
    directory.publish_ram((0, 1), 2, 3)
    directory.publish_ram((0, 1), 2, 4)

    with pytest.raises(RuntimeError, match="current host backing"):
        directory.publish_gpu((0, 1), 2, 3, 5)
    assert directory.snapshot()[0]["gpu_slot"] is None


def test_gpu_publication_failure_rolls_back_host_pin(tmp_path, monkeypatch):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        handle = cache.get(0, 0)
        cache.mark_in_flight(handle)

        def fail_publication(*_args, **_kwargs):
            raise RuntimeError("publication failed")

        monkeypatch.setattr(cache.residency, "publish_gpu", fail_publication)
        with pytest.raises(RuntimeError, match="publication failed"):
            cache.pin_for_gpu(handle, 0)
        assert cache._slots[handle.slot_id].pin_count == 0
        handle.release()
        cache.get(0, 1).release()
        assert cache.status()["warm_experts"] == 1
    finally:
        cache.close()


def test_failed_gpu_rebuild_preserves_residency(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 3), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        cache.ensure_experts(
            0, torch.tensor([0, 1, 2], dtype=torch.int32), phase="prefill"
        )
        before_map = cache.slot_for_id.clone()
        before_residency = host_cache.residency.snapshot()
        with pytest.raises(ValueError, match="cannot back"):
            cache.rebuild(4)
        assert torch.equal(cache.slot_for_id, before_map)
        assert host_cache.residency.snapshot() == before_residency
    finally:
        cache.reset()
        host_cache.close()


def test_bounded_dense_staging_preserves_persistent_residency(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, gate_up, down = _make_ftw(tmp_path, layered=True)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(
        store,
        _budget(store, 3) + store.layer_storage_bytes,
        pin=False,
    )
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        cache.ensure_experts(
            0, torch.tensor([0, 1, 2], dtype=torch.int32), phase="prefill"
        )
        cache._bounded_prefill_cost = {
            "sparse_ns": 0,
            "sparse_experts": 0,
            "dense_ns": 0,
            "dense_experts": 0,
        }
        before_map = cache.slot_for_id.clone()
        before_residency = host_cache.residency.snapshot()
        before_persistent = {
            name: tensor[:3].clone()
            for name, tensor in cache.bank_caches.items()
        }
        routed = torch.tensor([0, 1, 2], dtype=torch.int32)
        assert cache.bounded_prefill_mode(routed) == "sparse"
        cache._record_bounded_prefill_cost("sparse", 10_000, 3)
        assert cache.bounded_prefill_mode(routed) == "dense"

        views = cache.bounded_dense_layer(1, routed)
        assert torch.equal(cache.slot_for_id, before_map)
        assert host_cache.residency.snapshot() == before_residency
        for name, expected in before_persistent.items():
            assert torch.equal(cache.bank_caches[name][:3], expected)
        assert torch.equal(views[0], gate_up[3:])
        assert torch.equal(views[1], down[3:])
        assert host_cache._frequency[(1, 2)] == 1
    finally:
        cache.reset()
        host_cache.close()


def test_bounded_sparse_prefill_loads_only_unique_routes(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 3), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        routed = torch.tensor([2, 2, 0], dtype=torch.int32)
        cache.ensure_experts(1, routed, phase="prefill")
        assert host_cache.status()["disk_reads"] == 2
        assert cache._bounded_prefill_cost["sparse_experts"] == 2
        assert routed[0].item() == routed[1].item()
        assert routed.unique().numel() == 2
    finally:
        cache.reset()
        host_cache.close()


def test_sparse_bounded_prefill_matches_dense_and_runs_hits_first(
    tmp_path, monkeypatch
):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    path, _gate_up, _down = _make_ftw(tmp_path, layered=True)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(
        store,
        _budget(store, 3) + store.layer_storage_bytes,
        pin=False,
    )
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=3,
        top_k=2,
        hidden_size=2,
        intermediate_size=1,
    )
    layer.offload_cache = cache
    hidden = torch.tensor([[0.5, -0.25], [1.0, 0.75]])
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    routes = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    def reference_gemm(
        _cache,
        routed_hidden,
        routed_weights,
        routed_ids,
        *,
        views,
        **_kwargs,
    ):
        gate_up, down = views
        output = torch.zeros_like(routed_hidden)
        for token_id in range(routed_hidden.shape[0]):
            for route_id in range(routed_ids.shape[1]):
                expert_id = int(routed_ids[token_id, route_id])
                projected = gate_up[expert_id] @ routed_hidden[token_id]
                gate, up = projected.chunk(2)
                activated = torch.nn.functional.silu(gate) * up
                output[token_id] += (
                    routed_weights[token_id, route_id]
                    * (down[expert_id] @ activated)
                )
        return output

    monkeypatch.setattr(layer, "_expert_gemm", reference_gemm)
    try:
        dense_views = cache.bounded_dense_layer(0, routes)
        expected = layer._expert_gemm(
            cache,
            hidden,
            weights,
            routes,
            views=dense_views,
            n=3,
            alphas=cache.alphas_for_layer(0),
            is_prefill=True,
        )
        cache._bounded_prefill_cost = {
            "sparse_ns": 0,
            "sparse_experts": 0,
            "dense_ns": 0,
            "dense_experts": 0,
        }
        cache.ensure_experts(
            0, torch.tensor([0], dtype=torch.int32), phase="decode"
        )
        host_cache.set_tracing(True)
        actual = layer._prefill_routed(hidden, weights, routes.clone())
        assert torch.allclose(actual, expected)
        trace = host_cache.status()["metrics"]["trace"]
        assert trace[0]["tier"] == "vram"
        assert any(item["tier"] == "nvme" for item in trace[1:])
    finally:
        cache.reset()
        host_cache.close()


def test_io_coordinator_prioritizes_demand_and_honors_cancellation():
    coordinator = ExpertIoCoordinator(workers=1)
    started = threading.Event()
    release = threading.Event()
    order = []

    def blocker():
        started.set()
        release.wait(timeout=2)

    def record(name):
        order.append(name)
        return name

    try:
        running = coordinator.submit("demand", blocker)
        assert started.wait(timeout=2)
        canceled = coordinator.submit("prefetch", record, "canceled")
        queued_prefetch = coordinator.submit("prefetch", record, "prefetch")
        queued_demand = coordinator.submit("demand", record, "demand")
        assert canceled.cancel()
        release.set()
        running.result(timeout=2)
        assert queued_demand.result(timeout=2) == "demand"
        assert queued_prefetch.result(timeout=2) == "prefetch"
        assert order == ["demand", "prefetch"]
    finally:
        release.set()
        coordinator.shutdown()


def test_failed_parallel_read_does_not_publish_or_leak_slots(
    tmp_path, monkeypatch
):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    cache = HostExpertCache(store, _budget(store, 2), pin=False)
    original_read = store.read_expert

    def fail_one(layer_id, expert_id, destinations):
        if expert_id == 1:
            raise OSError("read canceled")
        return original_read(layer_id, expert_id, destinations)

    monkeypatch.setattr(store, "read_expert", fail_one)
    try:
        with pytest.raises(OSError, match="read canceled"):
            cache.get_many(0, [0, 1], force_admit=True)
        assert cache._map == {}
        assert len(cache._free_slots) == 2
        assert all(slot.state.value == "free" for slot in cache._slots)
    finally:
        cache.close()


def test_ftw_expert_pages_round_trip_and_legacy_equivalence(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks

    page_path, gate_up, down, index = _make_ftw_pages(
        tmp_path, devices=("nvme0", "nvme1")
    )
    metadata = index["expert_pages"]
    assert metadata["version"] == 1
    assert metadata["alignment"] == 4096
    assert len(metadata["pages"]) == 6
    assert [
        page["device"] for page in metadata["pages"]
    ] == ["nvme0", "nvme1", "nvme0", "nvme1", "nvme0", "nvme1"]
    for device in metadata["devices"]:
        offsets = [
            page["device_offset"]
            for page in metadata["pages"]
            if page["device"] == device
        ]
        assert offsets == sorted(offsets)

    store = NvmeExpertStore(str(page_path), num_layers=2, num_experts=3)
    read_calls = 0
    original_read_into = store.reader.read_into

    def counted_read_into(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original_read_into(*args, **kwargs)

    store.reader.read_into = counted_read_into
    assert store.capability == "expert_pages"
    cache = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        with cache.get(1, 2) as handle:
            assert torch.equal(handle.tensors["gate_up"], gate_up[5])
            assert torch.equal(handle.tensors["down"], down[5])
        read_calls = 0
        staging = cache.load_layer(0)
        assert read_calls == 1
        assert torch.equal(staging["gate_up"], gate_up[:3])
        assert torch.equal(staging["down"], down[:3])
    finally:
        cache.close()

    loaded = load_ftw_banks(
        str(page_path),
        num_layers=2,
        layer_residency=["pageable", "pageable"],
    )
    assert loaded is not None
    assert torch.equal(loaded.sources["gate_up"][1], gate_up[3:])
    assert torch.equal(loaded.sources["down"][0], down[:3])

    legacy_path, _legacy_gate, _legacy_down = _make_ftw(
        tmp_path, layered=True
    )
    legacy = NvmeExpertStore(
        str(legacy_path), num_layers=2, num_experts=3
    )
    assert legacy.capability == "bank_rows"
    legacy.close()


def test_ftw_page_conversion_uses_expert_sized_temporaries(tmp_path):
    _path, _gate_up, _down, index = _make_ftw_pages(tmp_path)
    page_bytes = [page["nbytes"] for page in index["expert_pages"]["pages"]]
    assert len(set(page_bytes)) == 1
    assert max(page_bytes) == (2 * 2 + 2 * 1) * 4
    assert all(
        entry["kind"] != "experts_bank"
        for entry in index["tensors"]
    )


def test_prefetch_is_useful_but_cannot_evict_protected_demand(tmp_path):
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    protected = HostExpertCache(store, _budget(store, 1), pin=False)
    try:
        with protected.get(0, 0):
            pass
        with protected.get(0, 0):
            pass
        assert next(iter(protected._protected)) == protected._map[(0, 0)]
        assert protected.submit_prefetch(0, [1]).result(timeout=2) == (0, 1)
        assert set(protected._map) == {(0, 0)}
    finally:
        protected.close()

    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    useful = HostExpertCache(store, _budget(store, 2), pin=False)
    try:
        assert useful.submit_prefetch(0, [1]).result(timeout=2) == (1, 0)
        with useful.get(0, 1):
            pass
        rows = useful.status()["metrics"]["rows"]
        demand = next(
            row
            for row in rows
            if row["phase"] == "decode" and row["tier"] == "ram"
        )
        assert demand["useful_prefetches"] == 1
    finally:
        useful.close()


def test_static_bounded_slots_map_dynamic_routes_without_reads(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 6), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        cache.enable_static_graphs()
        reads = host_cache.status()["disk_reads"]
        first = torch.tensor([[0, 2]], dtype=torch.int32)
        second = torch.tensor([[1, 0]], dtype=torch.int32)
        cache.ensure_experts(1, first)
        cache.ensure_experts(0, second)
        assert first.tolist() == [[
            cache._bounded_slot_for_id[1][0],
            cache._bounded_slot_for_id[1][2],
        ]]
        assert second.tolist() == [[
            cache._bounded_slot_for_id[0][1],
            cache._bounded_slot_for_id[0][0],
        ]]
        assert host_cache.status()["disk_reads"] == reads
        before = list(cache._bounded_id_of_slot)
        with pytest.raises(ValueError, match="graph-stable"):
            cache.rebuild(3)
        assert cache._bounded_id_of_slot == before
    finally:
        cache.reset()
        host_cache.close()


def test_controller_validation_and_arena_accounting_are_atomic(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 3), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    try:
        status = cache.ram_cache_status()
        arenas = status["arenas"]
        assert (
            arenas["persistent"]["bytes"]
            + arenas["transfer"]["bytes"]
            + arenas["dense_staging"]["bytes"]
            == arenas["total_bytes"]
            == status["allocated_bytes"]
        )
        previous = cache.ram_cache_status()["controller"]
        with pytest.raises(ValueError, match="unknown cache controller"):
            cache.configure_controller(limits={"invalid": 1})
        current = cache.ram_cache_status()["controller"]
        assert current["enabled"] == previous["enabled"]
        assert current["limits"] == previous["limits"]
        assert current["arenas"]["vram"]["addresses"] == previous["arenas"]["vram"]["addresses"]
        host_cache._disk_bytes = 100
        host_cache._disk_latency_ns = 10_000
        cache._bounded_prefill_cost = {
            "sparse_ns": 300,
            "sparse_experts": 1,
            "dense_ns": 100,
            "dense_experts": 1,
        }
        host_cache.telemetry.record(
            "decode",
            0,
            "vram",
            "requests",
            nbytes=100,
            started_ns=0,
            finished_ns=100,
        )
        assert cache.ram_cache_status()["controller"]["ownership_priority"] == [
            "ram_expert",
            "dense_staging",
            "transfer",
        ]
        policy = cache.ram_cache_status()["controller"]["prefill_policy"]
        assert policy["sparse_ns_per_expert"] == 300
        assert policy["dense_ns_per_expert"] == 100
        assert policy["measured_crossover_density"] == pytest.approx(1 / 3)
    finally:
        cache.reset()
        host_cache.close()


def test_token_microbatch_computes_while_next_prefetch_runs(
    tmp_path, monkeypatch
):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 3), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=3,
        top_k=1,
        hidden_size=2,
        intermediate_size=1,
    )
    layer.offload_cache = cache
    prefetch_started = threading.Event()
    release_prefetch = threading.Event()
    original_prefetch = host_cache.prefetch_many

    def blocked_prefetch(layer_id, expert_ids):
        prefetch_started.set()
        assert release_prefetch.wait(timeout=2)
        return original_prefetch(layer_id, expert_ids)

    gemm_calls = 0

    def observed_gemm(
        _cache,
        routed_hidden,
        _weights,
        _ids,
        **_kwargs,
    ):
        nonlocal gemm_calls
        gemm_calls += 1
        if gemm_calls == 1:
            assert prefetch_started.wait(timeout=2)
            assert not cache._host_prefetches[0].done()
            release_prefetch.set()
        return torch.zeros_like(routed_hidden)

    monkeypatch.setattr(host_cache, "prefetch_many", blocked_prefetch)
    monkeypatch.setattr(
        cache, "bounded_prefill_microbatch_tokens", lambda _tokens: 1
    )
    monkeypatch.setattr(layer, "_expert_gemm", observed_gemm)
    try:
        output = layer._prefill_routed(
            torch.ones((2, 2)),
            torch.ones((2, 1)),
            torch.tensor([[0], [1]], dtype=torch.int32),
        )
        assert output.shape == (2, 2)
        assert gemm_calls == 2
        assert prefetch_started.is_set()
    finally:
        release_prefetch.set()
        cache.reset()
        host_cache.close()


def test_prefix_and_transition_prefetches_are_compact_and_signature_gated(
    tmp_path, monkeypatch
):
    from freetoken.moe.offload_cache import OffloadMoeCache

    path, _gate_up, _down = _make_ftw(tmp_path)
    store = NvmeExpertStore(str(path), num_layers=2, num_experts=3)
    host_cache = HostExpertCache(store, _budget(store, 3), pin=False)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.set_bounded_host_cache(host_cache)
    calls = []

    def record(layer_id, expert_ids):
        calls.append((layer_id, tuple(expert_ids)))

    monkeypatch.setattr(cache, "schedule_host_prefetch", record)
    try:
        cache.prepare_prefix_experts(
            {0: (2, 0)},
            cache.execution_signature,
        )
        cache.prepare_prefix_experts({0: (1,)}, "stale-signature")
        assert calls == [(0, (2, 0))]

        cache._route_transitions[(0, 1)] = Counter({2: 3, 0: 1})
        cache._update_route_predictor(0, [1])
        assert calls[-1] == (1, (2,))
    finally:
        cache.reset()
        host_cache.close()
