"""Bounded host-RAM cache for experts stored in an FTW checkpoint.

The normal offload path keeps one host tensor for every expert.  Bounded mode
keeps only a fixed number of expert rows in pinned host slots and reads misses
from FTW on demand. A slot remains LOADING until its read completes, then its
mapping becomes visible.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import count
from queue import PriorityQueue
import gc
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

from freetoken.utils import init_logger

from .host_banks import HostBank

logger = init_logger(__name__)

_RAM_SIZE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(b|k|kb|kib|m|mb|mib|g|gb|gib|t|tb|tib)?\s*$",
    re.IGNORECASE,
)
_RAM_MULTIPLIERS = {
    "b": 1,
    "k": 1 << 10,
    "kb": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mb": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gb": 1 << 30,
    "gib": 1 << 30,
    "t": 1 << 40,
    "tb": 1 << 40,
    "tib": 1 << 40,
}
_DEFAULT_RAM_RESERVE_BYTES = 8 << 30
_MAX_FREQUENCY = (1 << 31) - 1


class SlotState(str, Enum):
    FREE = "free"
    LOADING = "loading"
    READY = "ready"
    IN_FLIGHT = "in_flight"

class CachePhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    PREFETCH = "prefetch"


class CacheTier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"
    CPU = "cpu"


_METRIC_FIELDS = (

    "requests",
    "hits",
    "misses",
    "bytes",
    "admissions",
    "evictions",
    "bypasses",
    "useful_prefetches",
    "failed_prefetches",
)
class ExpertIoCoordinator:
    """Persistent priority scheduler for demand, dense, and prefetch reads."""

    _PRIORITY = {"demand": 0, "dense": 1, "prefetch": 2}

    def __init__(self, workers: int = 8):
        self._queue: PriorityQueue = PriorityQueue()
        self._sequence = count()
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"ft-moe-io-{worker_id}",
                daemon=True,
            )
            for worker_id in range(max(1, int(workers)))
        ]
        self._closed = False
        for thread in self._threads:
            thread.start()

    def submit(self, kind: str, function, *args) -> Future:
        if kind not in self._PRIORITY:
            raise ValueError(f"unknown I/O priority {kind!r}")
        if self._closed:
            raise RuntimeError("expert I/O coordinator is closed")
        future = Future()
        self._queue.put(
            (
                self._PRIORITY[kind],
                next(self._sequence),
                future,
                function,
                args,
            )
        )
        return future

    def _worker(self) -> None:
        while True:
            _priority, _sequence, future, function, args = self._queue.get()
            try:
                if function is None:
                    return
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(function(*args))
                except BaseException as exc:
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    def shutdown(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        for _thread in self._threads:
            self._queue.put(
                (3, next(self._sequence), Future(), None, ())
            )
        if wait:
            for thread in self._threads:
                thread.join()



class CacheTelemetry:
    """Thread-safe bounded counters and dependency intervals."""

    def __init__(self, trace_capacity: int = 4096):
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, int, str], dict[str, int]] = {}
        self._intervals: deque[dict] = deque(maxlen=trace_capacity)
        self._trace: deque[dict] = deque(maxlen=trace_capacity)
        self._tracing = False

    def record(
        self,
        phase: str | CachePhase,
        layer_id: int,
        tier: str | CacheTier,
        event: str,
        *,
        nbytes: int = 0,
        started_ns: int | None = None,
        finished_ns: int | None = None,
        expert_id: int | None = None,
    ) -> None:
        phase_name = str(phase.value if isinstance(phase, CachePhase) else phase)
        tier_name = str(tier.value if isinstance(tier, CacheTier) else tier)
        if event not in _METRIC_FIELDS:
            raise ValueError(f"unknown cache metric {event!r}")
        with self._lock:
            key = (phase_name, int(layer_id), tier_name)
            counters = self._counters.setdefault(
                key, {name: 0 for name in _METRIC_FIELDS}
            )
            counters[event] += 1
            counters["bytes"] += int(nbytes)
            if started_ns is not None and finished_ns is not None:
                interval = {
                    "phase": phase_name,
                    "layer": int(layer_id),
                    "tier": tier_name,
                    "start_ns": int(started_ns),
                    "end_ns": int(finished_ns),
                }
                self._intervals.append(interval)
                if self._tracing:
                    self._trace.append(
                        {**interval, "event": event, "expert": expert_id}
                    )

    def set_tracing(self, enabled: bool) -> None:
        with self._lock:
            self._tracing = bool(enabled)
            if enabled:
                self._trace.clear()

    @staticmethod
    def _union_ns(intervals: list[tuple[int, int]]) -> int:
        if not intervals:
            return 0
        merged = 0
        start, end = sorted(intervals)[0]
        for next_start, next_end in sorted(intervals)[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                merged += end - start
                start, end = next_start, next_end
        return merged + end - start

    def snapshot(self) -> dict:
        with self._lock:
            rows = [
                {
                    "phase": phase,
                    "layer": layer,
                    "tier": tier,
                    **counters,
                }
                for (phase, layer, tier), counters in sorted(self._counters.items())
            ]
            intervals = list(self._intervals)
            trace = list(self._trace)
            tracing = self._tracing
        by_tier: dict[str, list[tuple[int, int]]] = {}
        all_intervals = []
        for item in intervals:
            interval = (item["start_ns"], item["end_ns"])
            all_intervals.append(interval)
            by_tier.setdefault(item["tier"], []).append(interval)
        wait_ns = {
            tier: self._union_ns(tier_intervals)
            for tier, tier_intervals in by_tier.items()
        }
        dominant = (
            max(wait_ns, key=lambda tier: wait_ns[tier])
            if wait_ns
            else "none"
        )
        return {
            "rows": rows,
            "critical_path_ns": self._union_ns(all_intervals),
            "wait_ns": wait_ns,
            "classification": f"{dominant}-bound" if dominant != "none" else "idle",
            "tracing": tracing,
            "trace": trace,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._intervals.clear()
            self._trace.clear()


@dataclass
class ExpertResidencyRecord:
    layer_id: int
    expert_id: int
    ram_slot: int | None = None
    ram_generation: int = 0
    gpu_slot: int | None = None
    gpu_generation: int = 0


class ExpertResidencyDirectory:
    """Single authority for published host and device expert mappings."""

    def __init__(self):
        self._lock = threading.RLock()
        self._records: dict[tuple[int, int], ExpertResidencyRecord] = {}
        self._gpu_keys: dict[int, tuple[int, int]] = {}

    def publish_ram(self, key: tuple[int, int], slot_id: int, generation: int) -> None:
        with self._lock:
            record = self._records.setdefault(key, ExpertResidencyRecord(*key))
            record.ram_slot = int(slot_id)
            record.ram_generation = int(generation)

    def remove_ram(self, key: tuple[int, int], generation: int) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None or record.ram_generation != generation:
                return
            if record.gpu_slot is not None:
                raise RuntimeError("cannot remove host backing for a GPU-resident expert")
            self._records.pop(key, None)

    def publish_gpu(
        self,
        key: tuple[int, int],
        ram_slot: int,
        ram_generation: int,
        gpu_slot: int,
    ) -> int:
        with self._lock:
            record = self._records.get(key)
            if (
                record is None
                or record.ram_slot != ram_slot
                or record.ram_generation != ram_generation
            ):
                raise RuntimeError("GPU publication requires current host backing")
            if record.gpu_slot is not None and record.gpu_slot != gpu_slot:
                self._gpu_keys.pop(record.gpu_slot, None)
            old_key = self._gpu_keys.get(gpu_slot)
            if old_key is not None and old_key != key:
                old = self._records.get(old_key)
                if old is not None:
                    old.gpu_slot = None
            record.gpu_generation += 1
            record.gpu_slot = int(gpu_slot)
            self._gpu_keys[int(gpu_slot)] = key
            return record.gpu_generation

    def evict_gpu(self, gpu_slot: int) -> tuple[int, int] | None:
        with self._lock:
            key = self._gpu_keys.pop(int(gpu_slot), None)
            if key is None:
                return None
            record = self._records.get(key)
            if record is not None:
                record.gpu_slot = None
            return key

    def clear_gpu(self) -> list[tuple[int, int]]:
        with self._lock:
            keys = list(self._gpu_keys.values())
            for key in keys:
                record = self._records.get(key)
                if record is not None:
                    record.gpu_slot = None
            self._gpu_keys.clear()
            return keys

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "layer": record.layer_id,
                    "expert": record.expert_id,
                    "ram_slot": record.ram_slot,
                    "ram_generation": record.ram_generation,
                    "gpu_slot": record.gpu_slot,
                    "gpu_generation": record.gpu_generation,
                }
                for record in self._records.values()
            ]


@dataclass(frozen=True)
class RamCachePlan:
    """Resolved host-wide and per-rank RAM budgets for bounded mode."""

    requested: int | str
    host_budget_bytes: int
    per_rank_budget_bytes: int
    reserve_bytes: int = 0


def parse_ram_cache_size(value: str | int | None) -> int | str | None:
    """Parse a RAM cache size into bytes, ``"auto"``, ``"all"``, or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("RAM cache size must be a byte count, 'auto', or 'all'")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("RAM cache size must be greater than zero")
        return value
    text = str(value).strip().lower()
    if text in ("auto", "all"):
        return text
    match = _RAM_SIZE_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"invalid RAM cache size {value!r}; use bytes, K/M/G/T, 'auto', or 'all'"
        )
    amount = float(match.group(1))
    multiplier = _RAM_MULTIPLIERS.get((match.group(2) or "b").lower(), 1)
    result = amount * multiplier
    if not math.isfinite(result) or result <= 0 or result != int(result):
        raise ValueError(f"invalid RAM cache size {value!r}")
    return int(result)


def is_bounded_ram_cache(value: str | int | None) -> bool:
    """Return whether a RAM setting selects bounded mode instead of full residency."""
    parsed = parse_ram_cache_size(value)
    return parsed not in (None, "all")


def _mem_available_bytes() -> int | None:
    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError):
        pass
    if available is None:
        return None

    # A container can expose the host's MemAvailable while enforcing a smaller
    # cgroup limit.  Cap the usable value by the remaining cgroup allowance.
    # cgroup v2 keeps the pair in the unified hierarchy; v1 hosts keep it under
    # the memory controller.  Inside a cgroup namespace either path is the
    # container's own cgroup root.
    limit = current = None
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as stream:
            raw_limit = stream.read().strip()
        if raw_limit != "max":
            limit = int(raw_limit)
            with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as stream:
                current = int(stream.read().strip())
    except (OSError, ValueError):
        limit = current = None
    if limit is None or current is None:
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", encoding="utf-8") as stream:
                limit = int(stream.read().strip())
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", encoding="utf-8") as stream:
                current = int(stream.read().strip())
        except (OSError, ValueError):
            limit = current = None
    if limit is not None and current is not None:
        available = min(available, max(0, limit - current))
    return available


def _ram_reserve_bytes() -> int:
    raw = os.environ.get("FREETOKEN_MOE_RAM_RESERVE_GB", "")
    if not raw:
        return _DEFAULT_RAM_RESERVE_BYTES
    try:
        reserve = float(raw) * (1 << 30)
    except ValueError as exc:
        raise ValueError("FREETOKEN_MOE_RAM_RESERVE_GB must be a non-negative number") from exc
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("FREETOKEN_MOE_RAM_RESERVE_GB must be non-negative")
    return int(reserve)


def resolve_ram_cache_plan(
    requested: str | int | None,
    *,
    tp_size: int,
    minimum_bytes: int,
    available_bytes: int | None = None,
    reserve_bytes: int | None = None,
) -> RamCachePlan:
    """Resolve a bounded host-wide request into a per-rank budget.

    ``minimum_bytes`` includes one persistent slot and one bypass workspace.
    ``tp_size`` divides the host-wide request because every local rank owns a
    separate process and cache.
    """
    parsed = parse_ram_cache_size(requested)
    if parsed in (None, "all"):
        raise ValueError("a bounded RAM cache size is required")
    if tp_size < 1:
        raise ValueError(f"tensor parallel size must be positive, got {tp_size}")

    if parsed == "auto":
        available = _mem_available_bytes() if available_bytes is None else available_bytes
        if available is None:
            raise ValueError("cannot resolve --moe-ram-cache-size auto: MemAvailable is unknown")
        reserve = _ram_reserve_bytes() if reserve_bytes is None else max(0, reserve_bytes)
        host_budget = max(0, available - reserve)
        reserve_used = reserve
    else:
        assert isinstance(parsed, int)
        host_budget = parsed
        reserve_used = 0

    per_rank = host_budget // tp_size
    if per_rank < minimum_bytes:
        raise ValueError(
            f"RAM cache budget {host_budget} B provides {per_rank} B per TP rank, "
            f"but at least {minimum_bytes} B per rank is required"
        )
    return RamCachePlan(parsed, host_budget, per_rank, reserve_used)


@dataclass(frozen=True)
class BankSpec:
    name: str
    row_shape: tuple[int, ...]
    dtype: torch.dtype
    row_bytes: int


class NvmeExpertStore:
    """Serve individual expert rows from an FTW checkpoint."""

    def __init__(self, path: str, *, num_layers: int, num_experts: int):
        from freetoken.checkpoint.ftw import FTWReader, _LAYER_ENTRY_RE
        from freetoken.moe.offload_cache import _BANK_SCHEMAS

        self.path = path
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.reader = FTWReader(path)
        self.quant_format = self.reader.meta("quant_format")
        if self.quant_format not in _BANK_SCHEMAS:
            self.close()
            raise ValueError(
                "bounded MoE RAM cache requires an FTW checkpoint with a supported "
                f"quant_format, got {self.quant_format!r}"
            )
        self.bank_schema = tuple(_BANK_SCHEMAS[self.quant_format])
        self.bank_specs: dict[str, BankSpec] = {}
        self._locations: dict[str, list[list[tuple[int, int]]]] = {}

        pages_meta = self.reader.meta("expert_pages")
        if pages_meta is not None:
            self.capability = "expert_pages"
            self._add_expert_pages(pages_meta)
        else:
            self.capability = "bank_rows"
            entries = self.reader.entries("experts_bank")
            if not entries:
                self.close()
                raise ValueError(
                    f"FTW checkpoint {path!r} has no expert-bank entries"
                )
            row_entries = [
                entry
                for entry in entries
                if entry["name"]
                not in ("gate_up_alpha", "down_alpha")
            ]
            by_layer: dict[str, dict[int, dict]] = {}
            flat: dict[str, dict] = {}
            for entry in row_entries:
                match = _LAYER_ENTRY_RE.match(entry["name"])
                if match is None:
                    flat[entry["name"]] = entry
                else:
                    by_layer.setdefault(match.group("base"), {})[
                        int(match.group("layer"))
                    ] = entry

            meta_layers = self.reader.meta("expert_bank_num_layers")
            if meta_layers is not None and int(meta_layers) != num_layers:
                self.close()
                raise ValueError(
                    "FTW expert-bank metadata has "
                    f"{meta_layers} layers, expected {num_layers}"
                )
            try:
                for name in self.bank_schema:
                    if name in flat and name in by_layer:
                        raise ValueError(
                            f"FTW bank {name!r} mixes flat and per-layer entries"
                        )
                    if name in flat:
                        self._add_flat_bank(name, flat[name])
                    elif name in by_layer:
                        self._add_layer_bank(name, by_layer[name])
                    else:
                        raise ValueError(
                            "FTW checkpoint is missing expert-bank entries "
                            f"for {name!r}"
                        )
                unknown = (set(flat) | set(by_layer)) - set(
                    self.bank_schema
                )
                if unknown:
                    raise ValueError(
                        "FTW checkpoint has unsupported expert banks: "
                        f"{sorted(unknown)}"
                    )
            except Exception:
                self.close()
                raise

        self.logical_bytes_per_expert = sum(spec.row_bytes for spec in self.bank_specs.values())
        self.slot_storage_bytes = sum(_align_up(spec.row_bytes) for spec in self.bank_specs.values())
        self.total_expert_bytes = self.num_layers * self.num_experts * self.logical_bytes_per_expert
        self.layer_storage_bytes = sum(
            _align_up(spec.row_bytes * self.num_experts)
            for spec in self.bank_specs.values()
        )

    def _add_expert_pages(self, metadata: dict) -> None:
        if int(metadata.get("version", 0)) != 1:
            raise ValueError("unsupported FTW expert-page metadata version")
        pages = metadata.get("pages", [])
        expected_count = self.num_layers * self.num_experts
        if len(pages) != expected_count:
            raise ValueError(
                f"FTW has {len(pages)} expert pages, expected {expected_count}"
            )
        self._locations = {
            name: [
                [(0, 0) for _ in range(self.num_experts)]
                for _ in range(self.num_layers)
            ]
            for name in self.bank_schema
        }
        self._page_locations: list[list[tuple[int, int]]] = [
            [(0, 0) for _ in range(self.num_experts)]
            for _ in range(self.num_layers)
        ]
        seen = set()
        for page in pages:
            layer_id = int(page["layer"])
            expert_id = int(page["expert"])
            key = (layer_id, expert_id)
            if not (
                0 <= layer_id < self.num_layers
                and 0 <= expert_id < self.num_experts
            ):
                raise ValueError(f"FTW expert page {key} is out of range")
            if key in seen:
                raise ValueError(f"duplicate FTW expert page {key}")
            seen.add(key)
            self._page_locations[layer_id][expert_id] = (
                int(page["global_off"]),
                int(page["nbytes"]),
            )
            bank_meta = {item["name"]: item for item in page["banks"]}
            if set(bank_meta) != set(self.bank_schema):
                raise ValueError(f"FTW expert page {key} has invalid banks")
            for name in self.bank_schema:
                item = bank_meta[name]
                dtype = _dtype_of(item["dtype"])
                shape = tuple(int(value) for value in item["shape"])
                nbytes = int(item["nbytes"])
                spec = BankSpec(name, shape, dtype, nbytes)
                previous = self.bank_specs.get(name)
                if previous is not None and previous != spec:
                    raise ValueError(
                        f"FTW expert-page bank {name!r} changes layout"
                    )
                self.bank_specs[name] = spec
                self._locations[name][layer_id][expert_id] = (
                    int(page["global_off"]) + int(item["offset"]),
                    nbytes,
                )
        if len(seen) != expected_count:
            raise ValueError("FTW expert-page index is incomplete")

    def _check_entry(self, name: str, entry: dict, expected_rows: int) -> BankSpec:
        shape = tuple(int(v) for v in entry["shape"])
        if not shape or shape[0] != expected_rows:
            raise ValueError(
                f"FTW bank {name!r} has shape {shape}, expected first dimension {expected_rows}"
            )
        dtype = _dtype_of(entry["dtype"])
        row_bytes = math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
        if row_bytes <= 0 or row_bytes * expected_rows != int(entry["nbytes"]):
            raise ValueError(f"FTW bank {name!r} has inconsistent shape and byte count")
        return BankSpec(name, shape[1:], dtype, row_bytes)

    def _add_flat_bank(self, name: str, entry: dict) -> None:
        spec = self._check_entry(name, entry, self.num_layers * self.num_experts)
        locations = []
        for layer_id in range(self.num_layers):
            layer = []
            for expert_id in range(self.num_experts):
                row = layer_id * self.num_experts + expert_id
                layer.append((int(entry["global_off"]) + row * spec.row_bytes, spec.row_bytes))
            locations.append(layer)
        self.bank_specs[name] = spec
        self._locations[name] = locations

    def _add_layer_bank(self, name: str, entries: dict[int, dict]) -> None:
        expected = list(range(self.num_layers))
        if sorted(entries) != expected:
            raise ValueError(
                f"FTW bank {name!r} has per-layer entries {sorted(entries)}, expected {expected}"
            )
        locations = []
        first_spec = None
        for layer_id in expected:
            entry = entries[layer_id]
            spec = self._check_entry(name, entry, self.num_experts)
            if first_spec is None:
                first_spec = spec
            elif spec != first_spec:
                raise ValueError(f"FTW bank {name!r} changes shape or dtype by layer")
            locations.append([
                (int(entry["global_off"]) + expert_id * spec.row_bytes, spec.row_bytes)
                for expert_id in range(self.num_experts)
            ])
        assert first_spec is not None
        self.bank_specs[name] = first_spec
        self._locations[name] = locations

    def alpha(self, name: str) -> torch.Tensor | None:
        entry = self.reader.tensors.get(name)
        if entry is None:
            return None
        return self.reader.read_tensor(entry)

    def read_expert(
        self,
        layer_id: int,
        expert_id: int,
        destinations: dict[str, HostBank],
    ) -> int:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f"expert_id {expert_id} out of range [0, {self.num_experts})"
            )
        if set(destinations) != set(self.bank_schema):
            raise ValueError(
                "expert destination banks do not match the FTW bank schema"
            )
        if self.capability == "expert_pages":
            requests = []
            for name in self.bank_schema:
                offset, nbytes = self._locations[name][layer_id][expert_id]
                requests.append(
                    (destinations[name].memoryview(), offset, nbytes)
                )
            return self.reader.read_ranges_into(requests)
        disk_bytes = 0
        for name in self.bank_schema:
            offset, nbytes = self._locations[name][layer_id][expert_id]
            self.reader.read_range_into(
                destinations[name].memoryview(), offset, nbytes
            )
            disk_bytes += nbytes
        return disk_bytes

    def read_layer(
        self,
        layer_id: int,
        destinations: dict[str, HostBank],
        *,
        workers: int = 8,
    ) -> int:
        """Read one complete expert layer with parallel, contiguous bank reads."""
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"layer_id {layer_id} out of range [0, {self.num_layers})")
        if set(destinations) != set(self.bank_schema):
            raise ValueError("layer destination banks do not match the FTW bank schema")
        disk_bytes = 0
        if self.capability == "expert_pages":
            requests = []
            for name in self.bank_schema:
                destination = destinations[name].memoryview()
                row_bytes = self.bank_specs[name].row_bytes
                for expert_id in range(self.num_experts):
                    offset, nbytes = self._locations[name][layer_id][expert_id]
                    requests.append(
                        (
                            destination[
                                expert_id * row_bytes :
                                (expert_id + 1) * row_bytes
                            ],
                            offset,
                            nbytes,
                        )
                    )
            return self.reader.read_ranges_into(
                requests, workers=workers
            )
        for name in self.bank_schema:
            offset, _ = self._locations[name][layer_id][0]
            nbytes = self.bank_specs[name].row_bytes * self.num_experts
            self.reader.read_range_into(
                destinations[name].memoryview(),
                offset,
                nbytes,
                workers=workers,
            )
            disk_bytes += nbytes
        return disk_bytes

    def close(self) -> None:
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()

@dataclass
class _Slot:
    banks: dict[str, HostBank]
    state: SlotState = SlotState.FREE
    key: tuple[int, int] | None = None
    generation: int = 0
    segment: str = "probation"
    last_used: int = 0
    pin_count: int = 0


class HostExpertHandle:
    """A stable view of one loaded expert until the caller releases it."""

    __slots__ = ("tensors", "slot_id", "generation", "_cache", "_bypass")

    def __init__(self, cache, tensors: dict[str, torch.Tensor], slot_id: int | None,
                 generation: int, bypass: bool):
        self.tensors = tensors
        self.slot_id = slot_id
        self.generation = generation
        self._cache = cache
        self._bypass = bypass

    def release(self) -> None:
        if self._bypass:
            self._cache._release_bypass(self.generation)

    def __enter__(self) -> "HostExpertHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _align_up(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) // alignment * alignment


def _dtype_of(value: str) -> torch.dtype:
    try:
        return getattr(torch, value.removeprefix("torch."))
    except AttributeError as exc:
        raise ValueError(f"unknown FTW dtype {value!r}") from exc


class HostExpertCache:
    """Fixed-slot expert cache with frequency admission and segmented LRU."""

    def __init__(
        self,
        store: NvmeExpertStore,
        per_rank_budget_bytes: int,
        *,
        requested: int | str = "explicit",
        host_budget_bytes: int | None = None,
        tp_size: int = 1,
        pin: bool | None = None,
    ):
        self.store = store
        self.bank_schema = store.bank_schema
        self.per_rank_budget_bytes = int(per_rank_budget_bytes)
        self.host_budget_bytes = (
            int(host_budget_bytes)
            if host_budget_bytes is not None
            else self.per_rank_budget_bytes * max(1, tp_size)
        )
        self.requested = requested
        self.tp_size = max(1, tp_size)
        self.slot_storage_bytes = store.slot_storage_bytes
        can_stage_layer = (
            self.per_rank_budget_bytes
            >= store.layer_storage_bytes + self.slot_storage_bytes * 2
        )
        self.layer_workspace_bytes = store.layer_storage_bytes if can_stage_layer else 0
        self.workspace_bytes = self.slot_storage_bytes + self.layer_workspace_bytes
        self.minimum_bytes = self.workspace_bytes + self.slot_storage_bytes
        self._pin = torch.cuda.is_available() if pin is None else bool(pin)
        self._backing = "cuda" if self._pin else "mmap"
        self._lock = threading.RLock()
        self.telemetry = CacheTelemetry()
        self.residency = ExpertResidencyDirectory()
        self._frequency: dict[tuple[int, int], int] = {}
        self._miss_cost_ns: dict[tuple[int, int], int] = {}
        self._map: dict[tuple[int, int], int] = {}
        self._prefetched: set[tuple[int, int]] = set()
        self._clock = 0
        self._free_slots: deque[int] = deque()
        self._probation: OrderedDict[int, None] = OrderedDict()
        self._protected: OrderedDict[int, None] = OrderedDict()
        self._protected_limit = 1
        self._bypass_busy = False
        self._bypass_generation = 0
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._evictions = 0
        self._disk_bytes = 0
        self._disk_reads = 0
        self._disk_latency_ns = 0
        self._slots: list[_Slot] = []
        self._bypass: _Slot | None = None
        capacity = self._capacity_for_budget(self.per_rank_budget_bytes)
        self._io_pool = ExpertIoCoordinator(workers=8)
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ft-expert-prefetch",
        )
        try:
            self._slots = [self._new_slot() for _ in range(capacity)]
            self._layer_staging = (
                {
                    name: HostBank(
                        (self.store.num_experts, *self.store.bank_specs[name].row_shape),
                        self.store.bank_specs[name].dtype,
                        backing=self._backing,
                    )
                    for name in self.bank_schema
                }
                if self.layer_workspace_bytes
                else {}
            )
            self._bypass = self._new_slot()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"cannot allocate {self.allocated_bytes_for_capacity(capacity)} B "
                f"for the bounded MoE RAM cache"
            ) from exc
        self._capacity_reset()

    @classmethod
    def from_request(
        cls,
        store: NvmeExpertStore,
        requested: str | int,
        *,
        tp_size: int,
        available_bytes: int | None = None,
        reserve_bytes: int | None = None,
        pin: bool | None = None,
    ) -> "HostExpertCache":
        minimum = store.slot_storage_bytes * 2
        plan = resolve_ram_cache_plan(
            requested,
            tp_size=tp_size,
            minimum_bytes=minimum,
            available_bytes=available_bytes,
            reserve_bytes=reserve_bytes,
        )
        return cls(
            store,
            plan.per_rank_budget_bytes,
            requested=plan.requested,
            host_budget_bytes=plan.host_budget_bytes,
            tp_size=tp_size,
            pin=pin,
        )

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def has_layer_staging(self) -> bool:
        return bool(self._layer_staging)

    @property
    def logical_bytes_per_expert(self) -> int:
        return self.store.logical_bytes_per_expert

    @property
    def total_expert_bytes(self) -> int:
        return self.store.total_expert_bytes

    def allocated_bytes_for_capacity(self, capacity: int) -> int:
        return self.workspace_bytes + capacity * self.slot_storage_bytes

    def _capacity_for_budget(self, budget: int) -> int:
        if budget < self.minimum_bytes:
            raise ValueError(
                f"per-rank RAM cache budget {budget} B is below the minimum "
                f"{self.minimum_bytes} B"
            )
        requested = (budget - self.workspace_bytes) // self.slot_storage_bytes
        # Never allocate more slots than the local FTW store can supply.
        return min(requested, self.store.num_layers * self.store.num_experts)

    def validate_per_rank_budget(self, budget: int) -> int:
        """Validate a per-rank target and return its slot capacity."""
        return self._capacity_for_budget(int(budget))

    def _new_slot(self) -> _Slot:
        banks = {
            name: HostBank(
                (1, *self.store.bank_specs[name].row_shape),
                self.store.bank_specs[name].dtype,
                backing=self._backing,
            )
            for name in self.bank_schema
        }
        return _Slot(banks)

    def _record_access(self, key: tuple[int, int]) -> None:
        self._frequency[key] = min(_MAX_FREQUENCY, self._frequency.get(key, 0) + 1)

    def _admission_value(self, key: tuple[int, int]) -> float:
        reuse = max(1, self._frequency.get(key, 0))
        fallback_cost = (
            self._disk_latency_ns // self._disk_reads
            if self._disk_reads
            else 1
        )
        miss_cost = self._miss_cost_ns.get(key, fallback_cost)
        return reuse * miss_cost / self.slot_storage_bytes

    def record_accesses(
        self,
        layer_id: int,
        expert_ids: Iterable[int],
        *,
        phase: str | CachePhase = CachePhase.DECODE,
    ) -> None:
        """Record routed accesses and keep RAM copies of GPU hits recent."""
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.store.num_layers:
            raise ValueError(f"layer_id {layer_id} out of range")
        with self._lock:
            for expert_id in expert_ids:
                if not 0 <= int(expert_id) < self.store.num_experts:
                    raise ValueError(f"expert_id {expert_id} out of range")
                key = (layer_id, int(expert_id))
                self._record_access(key)
                slot_id = self._map.get(key)
                if slot_id is not None and self._slots[slot_id].state is SlotState.READY:
                    self._touch(slot_id)
                self.telemetry.record(phase, layer_id, CacheTier.RAM, "requests")

    def _touch(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        self._clock += 1
        slot.last_used = self._clock
        if slot.pin_count or slot.state is SlotState.IN_FLIGHT:
            return
        if slot.segment == "protected":
            self._protected[slot_id] = None
            self._protected.move_to_end(slot_id)
            return
        self._probation.pop(slot_id, None)
        slot.segment = "protected"
        self._protected[slot_id] = None
        if len(self._protected) > self._protected_limit:
            demote_id, _ = self._protected.popitem(last=False)
            self._slots[demote_id].segment = "probation"
            self._probation[demote_id] = None

    def _victim(self) -> int | None:
        if self._free_slots:
            return self._free_slots[0]
        if self._probation:
            return next(iter(self._probation))
        if self._protected:
            return next(iter(self._protected))
        return None

    def _claim_victim(self, slot_id: int) -> None:
        if self._free_slots and self._free_slots[0] == slot_id:
            self._free_slots.popleft()
        else:
            self._probation.pop(slot_id, None)
            self._protected.pop(slot_id, None)

    def _load(
        self,
        slot: _Slot,
        layer_id: int,
        expert_id: int,
        phase: str | CachePhase,
    ) -> None:
        started = time.perf_counter_ns()
        slot.state = SlotState.LOADING
        try:
            kind = (
                "prefetch"
                if phase == CachePhase.PREFETCH or phase == "prefetch"
                else "demand"
            )
            io_pool = self._io_pool
            assert io_pool is not None
            disk_bytes = io_pool.submit(
                kind,
                self.store.read_expert,
                layer_id,
                expert_id,
                slot.banks,
            ).result()
        except Exception:
            slot.state = SlotState.FREE
            slot.key = None
            self.telemetry.record(phase, layer_id, CacheTier.NVME, "misses")
            raise
        finished = time.perf_counter_ns()
        self._disk_bytes += disk_bytes
        self._disk_reads += 1
        self._disk_latency_ns += finished - started
        key = (layer_id, expert_id)
        elapsed = finished - started
        previous = self._miss_cost_ns.get(key)
        self._miss_cost_ns[key] = (
            elapsed if previous is None else (previous * 7 + elapsed) // 8
        )
        self.telemetry.record(
            phase,
            layer_id,
            CacheTier.NVME,
            "requests",
            nbytes=disk_bytes,
            started_ns=started,
            finished_ns=finished,
            expert_id=expert_id,
        )

    def _handle_for_slot(self, slot_id: int) -> HostExpertHandle:
        slot = self._slots[slot_id]
        assert slot.state is SlotState.READY and slot.key is not None
        return HostExpertHandle(
            self,
            {name: slot.banks[name].tensor[0] for name in self.bank_schema},
            slot_id,
            slot.generation,
            False,
        )

    def _load_bypass(
        self,
        layer_id: int,
        expert_id: int,
        phase: str | CachePhase,
    ) -> HostExpertHandle:
        if self._bypass is None or self._bypass_busy:
            raise RuntimeError("bounded MoE RAM cache bypass workspace is already in use")
        self._bypass_busy = True
        self._bypass_generation += 1
        generation = self._bypass_generation
        try:
            self._load(self._bypass, layer_id, expert_id, phase)
            self._bypass.state = SlotState.IN_FLIGHT
            return HostExpertHandle(
                self,
                {name: self._bypass.banks[name].tensor[0] for name in self.bank_schema},
                None,
                generation,
                True,
            )
        except Exception:
            self._bypass_busy = False
            raise

    def _release_bypass(self, generation: int) -> None:
        with self._lock:
            if generation != self._bypass_generation:
                raise RuntimeError("stale bounded MoE bypass handle")
            if self._bypass is not None:
                self._bypass.state = SlotState.FREE
            self._bypass_busy = False

    def get(
        self,
        layer_id: int,
        expert_id: int,
        *,
        prefill: bool = False,
        observe: bool = True,
        force_admit: bool = False,
        phase: str | CachePhase = CachePhase.DECODE,
    ) -> HostExpertHandle:
        """Return an expert from RAM or load it from FTW.

        Prefill misses use the bypass workspace and never enter the persistent
        cache.  This prevents a one-time full-layer scan from evicting decode
        working-set experts.
        """
        layer_id = int(layer_id)
        expert_id = int(expert_id)
        key = (layer_id, expert_id)
        with self._lock:
            if not 0 <= layer_id < self.store.num_layers:
                raise ValueError(f"layer_id {layer_id} out of range")
            if not 0 <= expert_id < self.store.num_experts:
                raise ValueError(f"expert_id {expert_id} out of range")
            if observe:
                self._record_access(key)
            slot_id = self._map.get(key)
            if slot_id is not None and self._slots[slot_id].state is SlotState.READY:
                self._hits += 1
                self._touch(slot_id)
                self.telemetry.record(phase, layer_id, CacheTier.RAM, "hits")
                if (
                    phase != CachePhase.PREFETCH
                    and phase != "prefetch"
                    and key in self._prefetched
                ):
                    self._prefetched.remove(key)
                    self.telemetry.record(
                        phase,
                        layer_id,
                        CacheTier.RAM,
                        "useful_prefetches",
                    )
                return self._handle_for_slot(slot_id)

            self._misses += 1
            victim_id = None if prefill else self._victim()
            victim = self._slots[victim_id] if victim_id is not None else None
            if (
                not prefill
                and not force_admit
                and victim is not None
                and victim.key is not None
                and self._admission_value(key)
                < self._admission_value(victim.key)
            ):
                self._bypasses += 1
                self.telemetry.record(phase, layer_id, CacheTier.RAM, "bypasses")
                return self._load_bypass(layer_id, expert_id, phase)
            if prefill or victim_id is None:
                self._bypasses += 1
                self.telemetry.record(phase, layer_id, CacheTier.RAM, "bypasses")
                return self._load_bypass(layer_id, expert_id, phase)

            if victim is None:
                raise RuntimeError("bounded MoE RAM cache has no eviction victim")
            self._claim_victim(victim_id)
            if victim.key is not None:
                old_key = victim.key
                self._map.pop(old_key, None)
                self.residency.remove_ram(old_key, victim.generation)
                self._evictions += 1
                self.telemetry.record(phase, layer_id, CacheTier.RAM, "evictions")
                if old_key in self._prefetched:
                    self._prefetched.remove(old_key)
                    self.telemetry.record(
                        phase,
                        old_key[0],
                        CacheTier.RAM,
                        "failed_prefetches",
                    )
            victim.generation += 1
            victim.key = key
            victim.segment = "probation"
            try:
                self._load(victim, layer_id, expert_id, phase)
            except Exception:
                victim.key = None
                victim.state = SlotState.FREE
                self._free_slots.appendleft(victim_id)
                raise
            victim.state = SlotState.READY
            self._map[key] = victim_id
            self.residency.publish_ram(key, victim_id, victim.generation)
            self.telemetry.record(phase, layer_id, CacheTier.RAM, "admissions")
            self._touch(victim_id)
            return self._handle_for_slot(victim_id)

    def get_many(
        self,
        layer_id: int,
        expert_ids: Iterable[int],
        *,
        observe: bool = True,
        force_admit: bool = False,
        phase: str | CachePhase = CachePhase.DECODE,
    ) -> tuple[dict[int, HostExpertHandle], list[int]]:
        """Load persistent-cache candidates concurrently.

        Returned handles are already IN_FLIGHT. Expert IDs that admission sends
        through the single bypass workspace are returned separately.
        """
        layer_id = int(layer_id)
        ids = list(dict.fromkeys(int(expert_id) for expert_id in expert_ids))
        handles: dict[int, HostExpertHandle] = {}
        bypass_ids: list[int] = []
        reserved: list[tuple[int, int]] = []
        with self._lock:
            if not 0 <= layer_id < self.store.num_layers:
                raise ValueError(f"layer_id {layer_id} out of range")
            for expert_id in ids:
                if not 0 <= expert_id < self.store.num_experts:
                    raise ValueError(f"expert_id {expert_id} out of range")
                key = (layer_id, expert_id)
                if observe:
                    self._record_access(key)
                slot_id = self._map.get(key)
                if slot_id is not None and self._slots[slot_id].state is SlotState.READY:
                    self._hits += 1
                    handle = self._handle_for_slot(slot_id)
                    self._probation.pop(slot_id, None)
                    self._protected.pop(slot_id, None)
                    self._slots[slot_id].state = SlotState.IN_FLIGHT
                    handles[expert_id] = handle
                    self.telemetry.record(phase, layer_id, CacheTier.RAM, "hits")
                    if (
                        phase != CachePhase.PREFETCH
                        and phase != "prefetch"
                        and key in self._prefetched
                    ):
                        self._prefetched.remove(key)
                        self.telemetry.record(
                            phase,
                            layer_id,
                            CacheTier.RAM,
                            "useful_prefetches",
                        )
                    continue

                victim_id = self._victim()
                victim = self._slots[victim_id] if victim_id is not None else None
                if (
                    victim is None
                    or (
                        (phase == CachePhase.PREFETCH or phase == "prefetch")
                        and victim.segment == "protected"
                    )
                    or (
                        not force_admit
                        and victim.key is not None
                        and self._admission_value(key)
                        < self._admission_value(victim.key)
                    )
                ):
                    bypass_ids.append(expert_id)
                    self.telemetry.record(phase, layer_id, CacheTier.RAM, "bypasses")
                    continue
                assert victim_id is not None
                self._misses += 1

                self._claim_victim(victim_id)
                if victim.key is not None:
                    old_key = victim.key
                    self._map.pop(old_key, None)
                    self.residency.remove_ram(old_key, victim.generation)
                    self._evictions += 1
                    self.telemetry.record(phase, layer_id, CacheTier.RAM, "evictions")
                    if old_key in self._prefetched:
                        self._prefetched.remove(old_key)
                        self.telemetry.record(
                            phase,
                            old_key[0],
                            CacheTier.RAM,
                            "failed_prefetches",
                        )
                victim.key = key
                victim.segment = "probation"
                victim.state = SlotState.LOADING
                reserved.append((expert_id, victim_id))

        def load_one(item: tuple[int, int]) -> tuple[int, int, int, int]:
            expert_id, slot_id = item
            started = time.perf_counter_ns()
            disk_bytes = self.store.read_expert(
                layer_id,
                expert_id,
                self._slots[slot_id].banks,
            )
            return expert_id, disk_bytes, started, time.perf_counter_ns()

        io_pool = self._io_pool
        assert io_pool is not None
        kind = (
            "prefetch"
            if phase == CachePhase.PREFETCH or phase == "prefetch"
            else "demand"
        )
        futures = [
            io_pool.submit(kind, load_one, item) for item in reserved
        ]
        try:
            loaded = [future.result() for future in futures]
        except Exception:
            for future in futures:
                future.cancel()
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass
            with self._lock:
                for expert_id, slot_id in reserved:
                    slot = self._slots[slot_id]
                    slot.key = None
                    slot.state = SlotState.FREE
                    self._free_slots.append(slot_id)
                for handle in handles.values():
                    self.release_in_flight(handle)
            raise

        loaded_stats = {
            expert_id: (disk_bytes, started, finished)
            for expert_id, disk_bytes, started, finished in loaded
        }
        with self._lock:
            for expert_id, slot_id in reserved:
                slot = self._slots[slot_id]
                disk_bytes, started, finished = loaded_stats[expert_id]
                elapsed = finished - started
                self._disk_bytes += disk_bytes
                self._disk_reads += 1
                self._disk_latency_ns += elapsed
                slot.state = SlotState.IN_FLIGHT
                key = (layer_id, expert_id)
                previous = self._miss_cost_ns.get(key)
                self._miss_cost_ns[key] = (
                    elapsed
                    if previous is None
                    else (previous * 7 + elapsed) // 8
                )
                self._map[key] = slot_id
                self.residency.publish_ram(key, slot_id, slot.generation)
                self.telemetry.record(
                    phase,
                    layer_id,
                    CacheTier.NVME,
                    "requests",
                    nbytes=disk_bytes,
                    started_ns=started,
                    finished_ns=finished,
                    expert_id=expert_id,
                )
                self.telemetry.record(phase, layer_id, CacheTier.RAM, "admissions")
                handles[expert_id] = HostExpertHandle(
                    self,
                    {name: slot.banks[name].tensor[0] for name in self.bank_schema},
                    slot_id,
                    slot.generation,
                    False,
                )
        return handles, bypass_ids
    def prefetch_many(
        self, layer_id: int, expert_ids: Iterable[int]
    ) -> tuple[int, int]:
        ids = list(dict.fromkeys(int(expert) for expert in expert_ids))
        with self._lock:
            absent = {
                expert
                for expert in ids
                if (int(layer_id), expert) not in self._map
            }
        handles, bypass_ids = self.get_many(
            layer_id,
            ids,
            observe=False,
            force_admit=False,
            phase=CachePhase.PREFETCH,
        )
        try:
            with self._lock:
                for expert_id in absent:
                    if expert_id in handles:
                        self._prefetched.add((int(layer_id), expert_id))
        finally:
            for handle in handles.values():
                self.release_in_flight(handle)
        return len(handles), len(bypass_ids)
    def submit_prefetch(
        self, layer_id: int, expert_ids: Iterable[int]
    ) -> Future[tuple[int, int]]:
        pool = self._prefetch_pool
        if pool is None:
            raise RuntimeError("bounded MoE RAM cache is closed")
        ids = tuple(int(expert) for expert in expert_ids)
        return pool.submit(self.prefetch_many, int(layer_id), ids)




    def load_layer(
        self,
        layer_id: int,
        *,
        phase: str | CachePhase = CachePhase.PREFILL,
    ) -> dict[str, torch.Tensor]:
        """Load a full layer into the fixed pinned staging workspace."""
        if not self._layer_staging:
            raise RuntimeError("RAM cache budget has no complete-layer staging workspace")
        with self._lock:
            started = time.perf_counter_ns()
            io_pool = self._io_pool
            assert io_pool is not None
            disk_bytes = io_pool.submit(
                "dense",
                self.store.read_layer,
                layer_id,
                self._layer_staging,
            ).result()
            finished = time.perf_counter_ns()
            self._disk_bytes += disk_bytes
            self._disk_reads += self.store.num_experts
            self._disk_latency_ns += finished - started
            self.telemetry.record(
                phase,
                layer_id,
                CacheTier.NVME,
                "requests",
                nbytes=disk_bytes,
                started_ns=started,
                finished_ns=finished,
            )
            return {
                name: self._layer_staging[name].tensor
                for name in self.bank_schema
            }

    def mark_in_flight(self, handle: HostExpertHandle) -> None:
        """Protect a persistent slot during an asynchronous consumer operation."""
        if handle.slot_id is None:
            return
        with self._lock:
            slot = self._slots[handle.slot_id]
            if slot.generation != handle.generation or slot.state is not SlotState.READY:
                raise RuntimeError("stale bounded MoE RAM cache handle")
            self._probation.pop(handle.slot_id, None)
            self._protected.pop(handle.slot_id, None)
            slot.state = SlotState.IN_FLIGHT

    def release_in_flight(self, handle: HostExpertHandle) -> None:
        if handle.slot_id is None:
            return
        with self._lock:
            slot = self._slots[handle.slot_id]
            if slot.generation != handle.generation or slot.state is not SlotState.IN_FLIGHT:
                raise RuntimeError("stale bounded MoE RAM cache handle")
            slot.state = SlotState.READY
            if slot.pin_count:
                return
            if slot.segment == "protected":
                self._protected[handle.slot_id] = None
            else:
                self._probation[handle.slot_id] = None
    def pin_for_gpu(
        self, handle: HostExpertHandle, gpu_slot: int
    ) -> tuple[int, int]:
        if handle._bypass or handle.slot_id is None:
            raise RuntimeError("GPU residency requires persistent host backing")
        with self._lock:
            slot = self._slots[handle.slot_id]
            if (
                slot.key is None
                or slot.generation != handle.generation
                or slot.state is not SlotState.IN_FLIGHT
            ):
                raise RuntimeError("cannot pin a stale host expert handle")
            slot.pin_count += 1
            self._probation.pop(handle.slot_id, None)
            self._protected.pop(handle.slot_id, None)
            try:
                gpu_generation = self.residency.publish_gpu(
                    slot.key, handle.slot_id, handle.generation, gpu_slot
                )
            except Exception:
                slot.pin_count -= 1
                raise
            return handle.slot_id, gpu_generation

    def unpin_gpu(self, gpu_slot: int) -> tuple[int, int] | None:
        with self._lock:
            key = self.residency.evict_gpu(gpu_slot)
            if key is None:
                return None
            slot_id = self._map.get(key)
            if slot_id is None:
                raise RuntimeError("GPU mapping lost its host backing")
            slot = self._slots[slot_id]
            if slot.pin_count <= 0:
                raise RuntimeError("GPU mapping has no host pin")
            slot.pin_count -= 1
            if slot.pin_count == 0 and slot.state is SlotState.READY:
                self._touch(slot_id)
            return key

    def set_tracing(self, enabled: bool) -> None:
        self.telemetry.set_tracing(enabled)

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._bypasses = 0
            self._evictions = 0
            self._disk_bytes = 0
            self._disk_reads = 0
            self._disk_latency_ns = 0
            self.telemetry.reset()

    def resize(
        self,
        per_rank_budget_bytes: int,
        *,
        host_budget_bytes: int | None = None,
        requested: int | str | None = None,
    ) -> None:
        """Resize the fixed arena while idle, resetting cached entries and counters."""
        with self._lock:
            capacity = self._capacity_for_budget(int(per_rank_budget_bytes))
            if self._bypass_busy or any(
                slot.state is SlotState.IN_FLIGHT or slot.pin_count
                for slot in self._slots
            ):
                raise RuntimeError(
                    "cannot resize the bounded MoE RAM cache while a slot is active"
                )
            current = len(self._slots)
            extras: list[_Slot] = []
            if capacity > current:
                try:
                    extras = [self._new_slot() for _ in range(capacity - current)]
                except Exception as exc:
                    del extras
                    gc.collect()
                    raise RuntimeError(
                        f"cannot grow the bounded MoE RAM cache to {capacity} slots"
                    ) from exc
                self._slots.extend(extras)
            elif capacity < current:
                tail = self._slots[capacity:]
                for slot in tail:
                    if slot.key is not None:
                        self._map.pop(slot.key, None)
                    self._release_slot_memory(slot)
                self._slots = self._slots[:capacity]
                del tail
                gc.collect()
            self._capacity_reset()
            self.residency = ExpertResidencyDirectory()
            self.reset_stats()
            self.per_rank_budget_bytes = int(per_rank_budget_bytes)
            if host_budget_bytes is not None:
                self.host_budget_bytes = int(host_budget_bytes)
            if requested is not None:
                self.requested = requested
            self.tp_size = max(1, self.tp_size)

    @staticmethod
    def _release_slot_memory(slot: _Slot) -> None:
        for bank in slot.banks.values():
            bank.release()

    def _capacity_reset(self) -> None:
        self._map.clear()
        self._prefetched.clear()
        self._frequency.clear()
        self._free_slots = deque(range(len(self._slots)))
        self._probation.clear()
        self._protected.clear()
        for slot in self._slots:
            slot.state = SlotState.FREE
            slot.key = None
            slot.segment = "probation"
            slot.last_used = 0
            slot.pin_count = 0
        self._clock = 0
        self._protected_limit = max(1, int(len(self._slots) * 0.8))

    def status(self) -> dict:
        with self._lock:
            warm = sum(
                slot.state in (SlotState.READY, SlotState.IN_FLIGHT)
                for slot in self._slots
            )
            loading = sum(slot.state is SlotState.LOADING for slot in self._slots)
            pinned_experts = sum(slot.pin_count > 0 for slot in self._slots)
            pinned = (
                self.allocated_bytes_for_capacity(len(self._slots))
                if self._pin
                else 0
            )
            bypass = self._bypass
            assert bypass is not None
            status = {
                "mode": "bounded",
                "requested": self.requested,
                "requested_bytes": self.host_budget_bytes,
                "allocated_bytes": self.allocated_bytes_for_capacity(len(self._slots)),
                "per_rank_bytes": self.per_rank_budget_bytes,
                "capacity": len(self._slots),
                "total_expert_bytes": self.total_expert_bytes,
                "resident_bytes": warm * self.logical_bytes_per_expert,
                "warm_experts": warm,
                "loading_experts": loading,
                "gpu_pinned_experts": pinned_experts,
                "pinned_bytes": pinned,
                "workspace_bytes": self.workspace_bytes,
                "arenas": {
                    "persistent": {
                        "bytes": len(self._slots) * self.slot_storage_bytes,
                        "addresses": [
                            bank.tensor.data_ptr()
                            for slot in self._slots
                            for bank in slot.banks.values()
                        ],
                    },
                    "transfer": {
                        "bytes": self.slot_storage_bytes,
                        "addresses": [
                            bank.tensor.data_ptr()
                            for bank in bypass.banks.values()
                        ],
                    },
                    "dense_staging": {
                        "bytes": self.layer_workspace_bytes,
                        "addresses": [
                            bank.tensor.data_ptr()
                            for bank in self._layer_staging.values()
                        ],
                    },
                    "total_bytes": self.allocated_bytes_for_capacity(
                        len(self._slots)
                    ),
                },
                "hits": self._hits,
                "misses": self._misses,
                "bypasses": self._bypasses,
                "evictions": self._evictions,
                "disk_reads": self._disk_reads,
                "disk_bytes": self._disk_bytes,
                "disk_latency_us": self._disk_latency_ns / 1000.0,
                "metrics": self.telemetry.snapshot(),
                "residency_entries": len(self.residency.snapshot()),
            }
            return status

    def close(self) -> None:
        prefetch_pool = getattr(self, "_prefetch_pool", None)
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True, cancel_futures=True)
            self._prefetch_pool = None
        io_pool = getattr(self, "_io_pool", None)
        if io_pool is not None:
            io_pool.shutdown(wait=True)
            self._io_pool = None
        with getattr(self, "_lock", threading.RLock()):
            slots = getattr(self, "_slots", [])
            for slot in slots:
                self._release_slot_memory(slot)
            bypass = getattr(self, "_bypass", None)
            if bypass is not None:
                self._release_slot_memory(bypass)
            layer_staging = getattr(self, "_layer_staging", {})
            for bank in layer_staging.values():
                bank.release()
            self._slots = []
            self._bypass = None
            self._layer_staging = {}
            self._map = {}
            store = getattr(self, "store", None)
            if store is not None:
                store.close()
            self.residency = ExpertResidencyDirectory()
            del slots
        gc.collect()


__all__ = [
    "BankSpec",
    "CachePhase",
    "CacheTelemetry",
    "CacheTier",
    "ExpertIoCoordinator",
    "ExpertResidencyDirectory",
    "ExpertResidencyRecord",
    "HostExpertCache",
    "HostExpertHandle",
    "NvmeExpertStore",
    "RamCachePlan",
    "SlotState",
    "is_bounded_ram_cache",
    "parse_ram_cache_size",
    "resolve_ram_cache_plan",
]
