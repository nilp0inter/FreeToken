from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models import nvfp4_banks


_KEY_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_SPEC = nvfp4_banks.Nvfp4ExpertSourceSpec(
    key_pattern=_KEY_PATTERN,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, _config: layer,
    desc="synthetic NVFP4 experts",
)


@pytest.fixture
def nvfp4_checkpoint(tmp_path):
    E, H, I, L = 2, 16, 16, 2
    tensors = {}
    expected = {}
    for layer in range(L):
        for expert in range(E):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = 10 * (layer + 1) + expert + {
                    "gate_proj": 1,
                    "up_proj": 2,
                    "down_proj": 3,
                }[proj]
                prefix = (
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                )
                if proj == "down_proj":
                    weight_shape = (H, I // 2)
                    scale_shape = (H, I // 16)
                else:
                    weight_shape = (I, H // 2)
                    scale_shape = (I, H // 16)
                tensors[f"{prefix}.weight"] = torch.full(
                    weight_shape, base, dtype=torch.uint8
                )
                tensors[f"{prefix}.weight_scale"] = torch.ones(
                    scale_shape, dtype=torch.float8_e4m3fn
                )
                tensors[f"{prefix}.weight_scale_2"] = torch.tensor(
                    [base / 10], dtype=torch.float16
                )
                expected[layer, expert, proj] = base

    shard = tmp_path / "model.safetensors"
    save_file(tensors, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return tmp_path, SimpleNamespace(
        num_experts=E,
        hidden_size=H,
        moe_intermediate_size=I,
        num_moe_layers=L,
    ), expected


@pytest.mark.parametrize(
    "loader_name",
    ["load_nvfp4_expert_source_banks", "load_nvfp4_expert_source_banks_parallel"],
)
def test_nvfp4_streaming_loads_one_layer_at_a_time(
    nvfp4_checkpoint, monkeypatch, loader_name
):
    folder, config, expected = nvfp4_checkpoint
    monkeypatch.setattr(nvfp4_banks, "download_hf_weight", lambda _path: str(folder))

    allocations = []
    allocate = nvfp4_banks._alloc_nvfp4_host_banks

    def recording_allocate(*args, **kwargs):
        allocations.append((args[0], kwargs.get("backing")))
        return allocate(*args, **kwargs)

    monkeypatch.setattr(nvfp4_banks, "_alloc_nvfp4_host_banks", recording_allocate)

    delivered = []

    def sink(layer_id, banks):
        snapshots = {name: bank.tensor.clone() for name, bank in banks.items()}
        delivered.append((layer_id, snapshots))
        for bank in banks.values():
            bank.release()

    loader = getattr(nvfp4_banks, loader_name)
    sources = loader(
        str(folder),
        config,
        _SPEC,
        drop_page_cache=lambda _path: None,
        primary=False,
        layer_sink=sink,
    )

    assert allocations == [(1, "mmap"), (1, "mmap")]
    assert [layer_id for layer_id, _snapshots in delivered] == [0, 1]
    assert all(len(per_layer) == 2 for per_layer in sources.values())

    for layer_id, snapshots in delivered:
        packed = snapshots["gate_up_packed"]
        down = snapshots["down_packed"]
        for expert in range(config.num_experts):
            assert packed[expert, 0, 0].item() == expected[layer_id, expert, "gate_proj"]
            assert packed[expert, config.moe_intermediate_size, 0].item() == expected[
                layer_id, expert, "up_proj"
            ]
            assert down[expert, 0, 0].item() == expected[layer_id, expert, "down_proj"]

        gate_global = snapshots["gate_up_global"]
        down_global = snapshots["down_global"]
        for expert in range(config.num_experts):
            assert gate_global[expert, 0].item() == pytest.approx(
                torch.tensor(
                    expected[layer_id, expert, "gate_proj"] / 10, dtype=torch.float16
                ).item()
            )
            assert down_global[expert, 0].item() == pytest.approx(
                torch.tensor(
                    expected[layer_id, expert, "down_proj"] / 10, dtype=torch.float16
                ).item()
            )
