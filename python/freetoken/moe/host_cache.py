"""Bounded host-RAM cache for experts stored in an FTW checkpoint.

The normal offload path keeps one host tensor for every expert.  Bounded mode
keeps only a fixed number of expert rows in pinned host slots and reads misses
from FTW on demand.  The cache is deliberately synchronous at this boundary:
its caller runs in eager mode, so an expert is fully read before its GPU slot
mapping becomes visible.
"""

from __future__ import annotations

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
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as stream:
            raw_limit = stream.read().strip()
        if raw_limit != "max":
            limit = int(raw_limit)
            with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as stream:
                current = int(stream.read().strip())
            available = min(available, max(0, limit - current))
    except (OSError, ValueError):
        pass
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

        entries = self.reader.entries("experts_bank")
        if not entries:
            self.close()
            raise ValueError(f"FTW checkpoint {path!r} has no expert-bank entries")
        row_entries = [e for e in entries if e["name"] not in ("gate_up_alpha", "down_alpha")]
        by_layer: dict[str, dict[int, dict]] = {}
        flat: dict[str, dict] = {}
        for entry in row_entries:
            match = _LAYER_ENTRY_RE.match(entry["name"])
            if match is None:
                flat[entry["name"]] = entry
            else:
                by_layer.setdefault(match.group("base"), {})[int(match.group("layer"))] = entry

        meta_layers = self.reader.meta("expert_bank_num_layers")
        if meta_layers is not None and int(meta_layers) != num_layers:
            self.close()
            raise ValueError(
                f"FTW expert-bank metadata has {meta_layers} layers, expected {num_layers}"
            )

        try:
            for name in self.bank_schema:
                if name in flat and name in by_layer:
                    raise ValueError(f"FTW bank {name!r} mixes flat and per-layer entries")
                if name in flat:
                    self._add_flat_bank(name, flat[name])
                elif name in by_layer:
                    self._add_layer_bank(name, by_layer[name])
                else:
                    raise ValueError(
                        f"FTW checkpoint is missing expert-bank entries for {name!r}"
                    )
            unknown = (set(flat) | set(by_layer)) - set(self.bank_schema)
            if unknown:
                raise ValueError(f"FTW checkpoint has unsupported expert banks: {sorted(unknown)}")
        except Exception:
            self.close()
            raise

        self.logical_bytes_per_expert = sum(spec.row_bytes for spec in self.bank_specs.values())
        self.slot_storage_bytes = sum(_align_up(spec.row_bytes) for spec in self.bank_specs.values())
        self.total_expert_bytes = self.num_layers * self.num_experts * self.logical_bytes_per_expert

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

    def read_expert(self, layer_id: int, expert_id: int, destinations: dict[str, HostBank]) -> int:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"layer_id {layer_id} out of range [0, {self.num_layers})")
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f"expert_id {expert_id} out of range [0, {self.num_experts})")
        if set(destinations) != set(self.bank_schema):
            raise ValueError("expert destination banks do not match the FTW bank schema")
        disk_bytes = 0
        for name in self.bank_schema:
            offset, nbytes = self._locations[name][layer_id][expert_id]
            self.reader.read_range_into(destinations[name].memoryview(), offset, nbytes)
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
        self.workspace_bytes = store.slot_storage_bytes
        self.minimum_bytes = self.slot_storage_bytes + self.workspace_bytes
        self._pin = torch.cuda.is_available() if pin is None else bool(pin)
        self._backing = "cuda" if self._pin else "mmap"
        self._lock = threading.RLock()
        self._frequency: dict[tuple[int, int], int] = {}
        self._map: dict[tuple[int, int], int] = {}
        self._clock = 0
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
        try:
            self._slots = [self._new_slot() for _ in range(capacity)]
            self._bypass = self._new_slot()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"cannot allocate {self.allocated_bytes_for_capacity(capacity)} B "
                f"for the bounded MoE RAM cache"
            ) from exc
        self._protected_limit = max(1, int(capacity * 0.8))

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

    def record_accesses(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        """Record routed accesses without loading or admitting experts."""
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.store.num_layers:
            raise ValueError(f"layer_id {layer_id} out of range")
        with self._lock:
            for expert_id in expert_ids:
                if not 0 <= int(expert_id) < self.store.num_experts:
                    raise ValueError(f"expert_id {expert_id} out of range")
                self._record_access((layer_id, int(expert_id)))

    def _touch(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        self._clock += 1
        slot.last_used = self._clock
        if slot.segment == "probation":
            slot.segment = "protected"
            protected = [
                s for s in self._slots
                if s.state is SlotState.READY and s.segment == "protected"
            ]
            if len(protected) > self._protected_limit:
                demote = min(protected, key=lambda s: s.last_used)
                demote.segment = "probation"

    def _victim(self) -> int | None:
        free = [i for i, slot in enumerate(self._slots) if slot.state is SlotState.FREE]
        if free:
            return free[0]
        probation = [
            (i, slot) for i, slot in enumerate(self._slots)
            if slot.state is SlotState.READY and slot.segment == "probation"
        ]
        if not probation:
            protected = [
                slot for slot in self._slots
                if slot.state is SlotState.READY and slot.segment == "protected"
            ]
            if not protected:
                return None
            demote = min(protected, key=lambda s: s.last_used)
            demote.segment = "probation"
            probation = [
                (i, slot) for i, slot in enumerate(self._slots)
                if slot.state is SlotState.READY and slot.segment == "probation"
            ]
        return min(probation, key=lambda item: item[1].last_used)[0]

    def _load(self, slot: _Slot, layer_id: int, expert_id: int) -> None:
        started = time.perf_counter_ns()
        slot.state = SlotState.LOADING
        try:
            disk_bytes = self.store.read_expert(layer_id, expert_id, slot.banks)
        except Exception:
            slot.state = SlotState.FREE
            slot.key = None
            raise
        self._disk_bytes += disk_bytes
        self._disk_reads += 1
        self._disk_latency_ns += time.perf_counter_ns() - started

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

    def _load_bypass(self, layer_id: int, expert_id: int) -> HostExpertHandle:
        if self._bypass is None or self._bypass_busy:
            raise RuntimeError("bounded MoE RAM cache bypass workspace is already in use")
        self._bypass_busy = True
        self._bypass_generation += 1
        generation = self._bypass_generation
        try:
            self._load(self._bypass, layer_id, expert_id)
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
                return self._handle_for_slot(slot_id)

            self._misses += 1
            victim_id = None if prefill else self._victim()
            victim = self._slots[victim_id] if victim_id is not None else None
            candidate_frequency = self._frequency.get(key, 0)
            if (
                not prefill
                and victim is not None
                and victim.key is not None
                and candidate_frequency < self._frequency.get(victim.key, 0)
            ):
                self._bypasses += 1
                return self._load_bypass(layer_id, expert_id)
            if prefill or victim_id is None:
                self._bypasses += 1
                return self._load_bypass(layer_id, expert_id)

            if victim is None:
                raise RuntimeError("bounded MoE RAM cache has no eviction victim")
            if victim.key is not None:
                self._map.pop(victim.key, None)
                self._evictions += 1
            victim.generation += 1
            victim.key = key
            victim.segment = "probation"
            try:
                self._load(victim, layer_id, expert_id)
            except Exception:
                victim.key = None
                victim.state = SlotState.FREE
                raise
            victim.state = SlotState.READY
            self._map[key] = victim_id
            self._touch(victim_id)
            return self._handle_for_slot(victim_id)

    def mark_in_flight(self, handle: HostExpertHandle) -> None:
        """Protect a persistent slot during an asynchronous consumer operation."""
        if handle.slot_id is None:
            return
        with self._lock:
            slot = self._slots[handle.slot_id]
            if slot.generation != handle.generation or slot.state is not SlotState.READY:
                raise RuntimeError("stale bounded MoE RAM cache handle")
            slot.state = SlotState.IN_FLIGHT

    def release_in_flight(self, handle: HostExpertHandle) -> None:
        if handle.slot_id is None:
            return
        with self._lock:
            slot = self._slots[handle.slot_id]
            if slot.generation != handle.generation or slot.state is not SlotState.IN_FLIGHT:
                raise RuntimeError("stale bounded MoE RAM cache handle")
            slot.state = SlotState.READY

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._bypasses = 0
            self._evictions = 0
            self._disk_bytes = 0
            self._disk_reads = 0
            self._disk_latency_ns = 0

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
            if self._bypass_busy or any(slot.state is SlotState.IN_FLIGHT for slot in self._slots):
                raise RuntimeError("cannot resize the bounded MoE RAM cache while a slot is in flight")
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
        self._frequency.clear()
        for slot in self._slots:
            slot.state = SlotState.FREE
            slot.key = None
            slot.segment = "probation"
            slot.last_used = 0
        self._clock = 0
        self._protected_limit = max(1, int(len(self._slots) * 0.8))

    def status(self) -> dict:
        with self._lock:
            warm = sum(slot.state in (SlotState.READY, SlotState.IN_FLIGHT) for slot in self._slots)
            loading = sum(slot.state is SlotState.LOADING for slot in self._slots)
            pinned = self.allocated_bytes_for_capacity(len(self._slots)) if self._pin else 0
            return {
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
                "pinned_bytes": pinned,
                "workspace_bytes": self.workspace_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "bypasses": self._bypasses,
                "evictions": self._evictions,
                "disk_reads": self._disk_reads,
                "disk_bytes": self._disk_bytes,
                "disk_latency_us": self._disk_latency_ns / 1000.0,
            }

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            slots = getattr(self, "_slots", [])
            for slot in slots:
                self._release_slot_memory(slot)
            bypass = getattr(self, "_bypass", None)
            if bypass is not None:
                self._release_slot_memory(bypass)
            self._slots = []
            self._bypass = None
            self._map = {}
            store = getattr(self, "store", None)
            if store is not None:
                store.close()
            del slots
        gc.collect()


__all__ = [
    "BankSpec",
    "HostExpertCache",
    "HostExpertHandle",
    "NvmeExpertStore",
    "RamCachePlan",
    "SlotState",
    "is_bounded_ram_cache",
    "parse_ram_cache_size",
    "resolve_ram_cache_plan",
]
