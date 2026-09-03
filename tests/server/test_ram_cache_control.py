from freetoken.cache_report import format_cache_status
from freetoken.control_cli import _controller_target, _rebuild_body, parse_bytes


def test_control_cli_parses_ram_target_and_builds_wire_body():
    assert parse_bytes("1.5G") == 1.5 * (1 << 30)
    geometry = {"ram_cache": {"mode": "bounded"}}
    assert _rebuild_body({"ram": 64 << 30}, geometry) == {
        "ram_bytes": 64 << 30
    }


def test_cache_status_reports_host_ram_separately_from_vram():
    doc = {
        "state": "serving",
        "geometry": {
            "ram_cache": {
                "requested_bytes": 64 << 30,
                "allocated_bytes": 32 << 30,
                "warm_experts": 3,
                "hits": 7,
                "misses": 4,
                "evictions": 1,
                "disk_bytes": 128 << 20,
                "controller": {
                    "enabled": True,
                    "limits": {
                        "prefetch_experts": 8,
                        "microbatch_tokens": 128,
                    },
                    "static_graph_slots": False,
                },
            },
        },
    }
    text = format_cache_status(doc, prefix="")
    assert "ram 64.0 GiB requested, 32.0 GiB allocated" in text
    assert "hits=7" in text
    assert "warm=3" in text
    assert "disk=128.0 MiB" in text
    assert "controller on, prefetch=8 experts, microbatch=128 tokens" in text


def test_server_parser_accepts_ram_cache_size():
    from freetoken.server.args import parse_args

    args, run_shell = parse_args(
        [
            "--model-path", "dummy", "--dtype", "bfloat16",
            "--moe-backend", "offload",
            "--moe-ram-cache-size", "64G",
            "--no-moe-cache-controller",
            "--moe-prefetch-experts", "8",
            "--moe-prefill-microbatch-tokens", "128",
        ]
    )
    assert not run_shell
    assert args.moe_ram_cache_size == 64 << 30
    assert args.moe_cache_auto
    assert not args.moe_cache_controller
    assert args.moe_prefetch_experts == 8
    assert args.moe_prefill_microbatch_tokens == 128


def test_control_cli_rejects_ram_target_without_bounded_status():
    from freetoken.control_cli import ControlCliError

    try:
        _rebuild_body({"ram": 64 << 30}, {})
    except ControlCliError as exc:
        assert "no ram pool" in str(exc)
    else:
        raise AssertionError("expected a missing RAM pool error")

def test_shell_cache_parser_accepts_binary_ram_suffixes():
    from freetoken.cache_report import CachePools
    from freetoken.shell.tui import _parse_cache_command

    command = _parse_cache_command(["ram", "64G"], CachePools(kv=False, ram=True))
    assert command.action == "rebuild"
    assert command.ram_bytes == 64 << 30


def test_shell_and_control_cli_parse_controller_targets():
    from argparse import Namespace
    from freetoken.cache_report import CachePools
    from freetoken.shell.tui import _parse_cache_command

    command = _parse_cache_command(
        ["controller", "off", "prefetch", "8", "microbatch", "128"],
        CachePools(kv=False, ram=True),
    )
    assert command.controller_enabled is False
    assert command.prefetch_experts == 8
    assert command.microbatch_tokens == 128

    target = _controller_target(
        Namespace(
            controller="off",
            prefetch_experts=8,
            microbatch_tokens=128,
        )
    )
    assert target == {
        "controller_enabled": False,
        "controller_limits": {
            "prefetch_experts": 8,
            "microbatch_tokens": 128,
        },
    }


def test_cache_geometry_prefers_live_ram_metrics_over_rebuild_snapshot():
    from types import SimpleNamespace
    from freetoken.server.api_server import cache_geometry

    live = {"requested_bytes": 64, "hits": 9}
    stale = {"requested_bytes": 64, "hits": 0}
    state = SimpleNamespace(
        stats=SimpleNamespace(
            kv_total_pages=0,
            mamba_total_slots=0,
        ),
        config=SimpleNamespace(page_size=1, model_config=None),
        cache_pools={"moe_cache_size": 1},
        last_rebuild={"ram_cache": stale},
        ram_cache=live,
    )

    assert cache_geometry(state)["ram_cache"]["hits"] == 9
