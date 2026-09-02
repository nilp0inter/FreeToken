from freetoken.cache_report import format_cache_status
from freetoken.control_cli import _rebuild_body, parse_bytes


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
            }
        },
    }
    text = format_cache_status(doc, prefix="")
    assert "ram 64.0 GiB requested, 32.0 GiB allocated" in text
    assert "hits=7" in text
    assert "warm=3" in text
    assert "disk=128.0 MiB" in text

def test_server_parser_accepts_ram_cache_size():
    from freetoken.server.args import parse_args

    args, run_shell = parse_args(
        ["--model-path", "dummy", "--dtype", "bfloat16", "--moe-backend", "offload",
         "--moe-ram-cache-size", "64G"]
    )
    assert not run_shell
    assert args.moe_ram_cache_size == 64 << 30
    assert args.moe_cache_auto

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
