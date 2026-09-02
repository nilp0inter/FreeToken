# Repository Guidelines

## Project Overview

FreeToken is a Python inference engine for local serving of large open-weight language models. It supports dense and Mixture-of-Experts models across GPU, CPU, host memory, and disk-backed weight tiers.

The runtime exposes OpenAI-compatible and Anthropic-compatible HTTP APIs. The `ft` CLI provides serving, control, checkpoint conversion, benchmarks, a terminal shell, and agent launch commands.

The main package is under `python/freetoken`. Setuptools builds the Python package and native C++ extensions. CUDA and Triton code provide accelerated attention, sampling, expert execution, and cache operations.

## Architecture & Data Flow

1. `ft` dispatches through `freetoken.cli:main`.
2. `ft serve` starts the FastAPI and Uvicorn server through `server/launch.py` and `server/api_server.py`.
3. API workers send requests to tokenizer and backend workers through multiprocessing queues and typed IPC messages.
4. Tokenizer workers perform chat templating, tokenization, and detokenization outside the API event loop.
5. `server/supervisor.py` starts backend workers for tensor-parallel ranks.
6. `scheduler/scheduler.py` admits requests, runs chunked prefill and decode batches, allocates KV pages, and manages radix or hybrid caches.
7. `engine/engine.py` loads weights, selects attention and MoE backends, manages memory budgets, and runs model execution.
8. `models/loader.py` and `models/register.py` resolve model families. Layers call attention, KV-cache, and MoE components.
9. Sampling returns token IDs to the tokenizer and API stream. The daemon provides a separate process supervisor for managed server instances.

Important boundaries:

- The API process must not block on tokenization or backend work.
- CUDA stream and event synchronization coordinates host metadata preparation with engine execution.
- MoE expert banks use GPU caches, pinned host memory, CPU execution, or asynchronous host-to-device transfers.
- CUDA graph capture requires stable execution shapes and correct cache rebuild synchronization.
- The daemon import path is torch-free. Keep `freetoken.daemon` independent from heavy model and accelerator imports.

## Key Directories

| Path | Purpose |
| --- | --- |
| `python/freetoken/core.py` | Request, batch, context, and sampling state. |
| `python/freetoken/server/` | HTTP APIs, server launch, supervisor, control, statistics, and request handling. |
| `python/freetoken/scheduler/` | Prefill, decode, request lifecycle, page tables, cache management, and scheduler IPC. |
| `python/freetoken/engine/` | Model execution, CUDA graphs, sampling, backend selection, and cache budgets. |
| `python/freetoken/models/` | Model registry, loaders, sharded weights, and model-family implementations. |
| `python/freetoken/layers/` | Reusable linear, normalization, rotary, MoE, MHC, and GGUF layers. |
| `python/freetoken/attention/` | FlashAttention, FlashInfer, Triton, TRT-LLM, sparse, and linear-attention backends. |
| `python/freetoken/kvcache/` | Radix caches, paged pools, sliding-window pools, and recurrent-state pools. |
| `python/freetoken/moe/` | Expert banks, offload caches, quantized execution, and CPU or GPU MoE paths. |
| `python/freetoken/kernel/` | Python wrappers, Triton kernels, native extensions, and CUDA build or JIT helpers. |
| `python/freetoken/distributed/` | Tensor parallel state, PyNCCL wrappers, and process communication. |
| `python/freetoken/tokenizer/` | Tokenizer and detokenizer worker processes. |
| `python/freetoken/daemon/` | Managed serving processes, proxying, metrics, and lifecycle state. |
| `python/freetoken/checkpoint/` | Checkpoint conversion and FTW weight-format support. |
| `freetoken-kernel-cache/` | Independent package for prebuilt TVM FFI kernel-cache wheels. |
| `tests/` | Tests grouped by the source subsystem they exercise. |

## Development Commands

Enter the configured Nix environment before running project commands:

```bash
direnv allow
direnv exec . ft --version
```

On a fresh checkout, create the project environment and install runtime, accelerator, and development dependencies:

```bash
direnv exec . sh -c 'uv venv --python "$(command -v python)"'
direnv exec . uv pip install --python .venv/bin/python -e '.[dev,accel]'
direnv exec . uv pip check --python .venv/bin/python
```

Build the package and native extensions through the setuptools backend:

```bash
direnv exec . uv pip install --python .venv/bin/python -e .
```

The build uses Ninja, C++17, and CUDA. `python/freetoken/kernel/_toolchain.py` rejects an `nvcc` version that does not match PyTorch CUDA.

Run the service and common control commands:

```bash
ft serve --model <path-or-hf-id>
ft ctl health
ft ctl stats
ft ctl generate "Prompt" --max-tokens 128
ft shell
ft launch claude --dry-run
```

Convert a checkpoint or run the bandwidth benchmark:

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> --dtype bfloat16
ft bench bw --dtype nvfp4,bf16 --gpu 1
```

Run the Nix and Python smoke checks:

```bash
nix flake check --no-build
direnv exec . python -m compileall -q python tests
```

Build release wheels only when the release workflow is the target:

```bash
scripts/build-release-wheels.sh
scripts/publish-wheels.sh
```

`build-release-wheels.sh` requires a clean tree unless `FREETOKEN_BUILD_NO_STAMP=1`. It stamps `python/freetoken/version.py` and removes generated build directories. `publish-wheels.sh` prunes release assets before upload. Do not run either script for normal source changes.

No repository lint command or lint configuration is defined. Do not invent a lint workflow in a focused change.

## Code Conventions & Common Patterns

- Use Python type annotations and `from __future__ import annotations`.
- Use `snake_case` for Python modules, functions, and variables. Use `PascalCase` for classes.
- Keep request and runtime state in the existing dataclasses: `Req`, `Batch`, `Context`, and `SamplingParams` in `core.py`.
- Use the typed message classes under `python/freetoken/message/` for process and backend IPC. Do not replace message schemas with untyped dictionaries.
- Preserve the `set_global_ctx` and `get_global_ctx` lifecycle. Tests clear the global context between cases.
- Register model families through `models/register.py`. Keep model-specific loading and execution in the model package.
- Probe optional accelerator packages in `kernel/backend.py`. Preserve Triton or PyTorch fallbacks when an optional package is absent.
- Use rank-aware logging from `utils/logger.py` so tensor-parallel workers do not duplicate messages.
- Keep blocking work out of async API handlers. Use the existing worker, queue, process, and CUDA-stream boundaries.
- Preserve explicit resource cleanup in cache rebuild and failure paths. GPU tensors, pinned buffers, process groups, and temporary files need deterministic cleanup.
- Keep C++ extension source lists and include paths aligned between `setup.py` and `.omp/lsp.json`.
- Add tests beside the subsystem they protect. Prefer an independent PyTorch reference, CPU mirror, round trip, dequantizer oracle, or live registry over assertions that repeat production branches.

Contribution rules that affect coding agents:

- Make one focused change per pull request and link the issue.
- Add a regression test for a bug fix when the code permits it.
- Report the exact hardware, model, command, and test result.
- Performance changes need A/B tokens-per-second results and TTFT when prefill changes.
- Use Conventional Commits, for example `fix(server): stop API server when backend dies`.
- A human must understand and run AI-assisted changes on real hardware before submitting a pull request.

## Important Files

- `pyproject.toml`: dependency ranges, optional extras, console script, package layout, and pytest configuration.
- `setup.py`: native C++ extension definitions and the CUDA toolchain check.
- `flake.nix` and `flake.lock`: reproducible x86_64-linux devshell with Python, CUDA, compilers, Ninja, LSP tools, and debugpy.
- `.envrc`: loads the Nix flake with `use flake`.
- `.omp/lsp.json`: project LSP configuration. It generates the ignored `.clangd` file with Python, PyTorch, TVM FFI, NCCL, CUDA, and project include paths.
- `.omp/dap.json`: debugpy launch defaults using `.venv/bin/python`.
- `python/freetoken/cli.py`: `ft` command router.
- `python/freetoken/server/launch.py`: API and backend process launch.
- `python/freetoken/server/api_server.py`: FastAPI application startup.
- `python/freetoken/daemon/__main__.py`: daemon debug and service entrypoint.
- `tests/README.md`: test layout, markers, environment variables, and focused commands.
- `docs/cli.md`: CLI command reference.
- `CONTRIBUTING.md`: issue, pull request, AI, and commit requirements.

## Runtime/Tooling Preferences

- Target runtime: Linux x86_64 with an NVIDIA GPU and an NVIDIA driver that supports the CUDA 13 stack. The documented minimum driver is r580+.
- Use the Nix flake and direnv environment. Use `uv` for Python environments and dependencies. Do not install project dependencies into the system Python.
- The devshell defaults to Python 3.13. The package supports Python 3.10 and newer.
- The active CUDA toolkit is `cudaPackages_13_0`. `CUDA_HOME`, `CUDA_PATH`, and `CUDACXX` are exported by the devshell.
- The active native tools are GCC, Ninja, LLVM clangd 22.1.8, Pyright, and debugpy.
- The devshell exports `PYTHONPATH` with the repository root and `python`. This permits the `pytest` console command to import the local `tests` package.
- `clangd` handles C, C++, headers, `.cu`, and `.cuh` files. Edit `.omp/lsp.json` when include paths or server behavior changes. Do not hand-edit the generated `.clangd` file.
- Use the configured LSP integration for definitions, references, diagnostics, and symbol-aware renames. Use the configured debugpy integration for breakpoints and runtime state.
- The debugger starts `python/freetoken/daemon/__main__.py` with `.venv/bin/python`. Keep the daemon launch path torch-free until the daemon starts a serving process.
- No MCP server is configured for this project.
- The main environment excludes the Marlin `vllm` path because its dependency constraints conflict with the main `transformers` range. Use a separate environment for `vllm>=0.14,<0.15`.
- `uv.lock` and `.envrc` are ignored. Dependency ranges in `pyproject.toml` are the repository dependency contract.

## Testing & QA

Pytest discovers `tests/` and files named `test_*.py`. Run a focused test for the subsystem that changed:

```bash
uv run pytest tests/server/test_openai_api.py
uv run pytest tests/scheduler/test_hybrid_cache_manager.py
uv run pytest tests/kvcache/
uv run pytest tests/daemon/
```

Use the full suite only when the change needs it:

```bash
uv run pytest tests/
uv run pytest tests/ -m "not slow"
```

The `slow` marker covers large kernel sweeps and real-checkpoint reads. The `needs_weights` marker requires a local checkpoint. GPU tests skip when CUDA is unavailable. Marlin NVFP4 tests require `vllm` in a separate compatible environment.

The test directories mirror the protected source subsystems. The CLI surface and `install.sh` are thin, so run those commands directly instead of adding tests that repeat their current dispatch logic.

End-to-end tests need explicit local assets and hardware. Common gates include `FREETOKEN_TEST_MODEL`, `FREETOKEN_REBUILD_TEST_MODEL`, `FREETOKEN_AIME_SERIES`, `FREETOKEN_AIME{24,25,26}_JSONL`, `FREETOKEN_AIME_REQ`, `FREETOKEN_AIME_MAX_TOKENS`, `FREETOKEN_AIME_MIN_FREE_GIB`, and `FREETOKEN_GEMMA4_GGUF_GLOB`.

Before a rebuild E2E test, provide a small local model:

```bash
FREETOKEN_REBUILD_TEST_MODEL=<small model dir> uv run pytest tests/e2e/test_cache_rebuild.py
```

No coverage threshold is defined in the repository. Do not claim full-suite, benchmark, or real-hardware results unless the exact command and environment were run.
