# DGX Spark Skill

Claude Code skill for the **NVIDIA DGX Spark** (GB10 Grace-Blackwell Superchip).

The first SKILL.md targeting this hardware. Eliminates the guesswork that causes broken builds, crashed training runs, and bricked machines on the GB10's unique architecture.

## Install

```bash
npx skills add leap21ai/djx-spark-skill
```

## What It Covers

| Topic | What You Get |
|-------|-------------|
| **CUDA 13.0 Compatibility** | Fix `libcudart.so.12` errors, cu130 wheel install, cross-platform pyproject.toml, full package compatibility matrix |
| **Unified Memory (128 GB)** | Zero-copy architecture rules, safetensors double-load bug, buffer cache competition, cudaMemGetInfo gotchas |
| **OOM Prevention** | 5-layer defense against the swap death spiral (the #1 reported issue) — disable swap, cgroups, SSH protection, earlyoom, memory watchdog |
| **llama.cpp Builds** | GPU build (sm_121 targeting), CPU build (ARMv9-A SVE2/BF16/I8MM), NVFP4 build (`121f`), optimal runtime flags, `-fit off` for MoE, `--no-mmap` critical perf note |
| **Multi-Node Clustering** | ConnectX-7 RoCE v2 fabric, llama.cpp RPC backend, NCCL over RoCE, RDMA verification, netplan config |
| **vLLM Setup** | 4 install paths — NGC container, cu130 wheels, one-command installer, from-source. MOE kernel fixes |
| **SGLang** | Official spark container, EAGLE3 speculative decoding (2x throughput) |
| **Training** | SFT/DPO/GRPO recipes with cgroup memory jails, batch size memory budget tables, estimation formulas |
| **Benchmarks** | Official llama.cpp data for 6 models, context degradation curves, SGLang numbers, dual-node results |
| **Systemd Service** | Production service template, WatchdogSec incompatibility warning, CPUAffinity guidance, load time expectations by model/ctx size |
| **Why NOT Ollama** | Overhead analysis — Go runtime, no UMA flags, no RPC support, no direct GPU control |
| **Advanced** | PyTorch from source, Triton from source, compiler flags, `nvidia-smi` per-process GPU query, env var reference, kernel 6.17 upgrade |

## Why This Exists

Without this skill, Claude gives **generic NVIDIA advice** that causes real problems on DGX Spark:

| Question | Without Skill | With Skill |
|----------|--------------|------------|
| llama.cpp build flags | `CMAKE_CUDA_ARCHITECTURES=native` (wrong) | `121` + `GGML_CUDA_F16=ON` + `--no-mmap` |
| OOM freezes | Generic Linux sysctl tuning | 5-layer defense: swapoff, cgroups, SSH protection, earlyoom, watchdog |
| `pip install vllm` fails | "Fix LD_LIBRARY_PATH" (wrong — .so.12 doesn't exist) | CUDA 13 ABI break, cu130 wheels, NGC container |
| Attention implementation | "Use flash_attention_2" (crashes on sm_121) | "Never flash_attention_2 — use SDPA" |
| nvidia-smi shows "Not Supported" | Suggests tegrastats (wrong tool) | Explains UMA, `free -h`, drop caches, `--mlock` |
| systemd WatchdogSec | "Add watchdog for health check" (causes kill loop) | "Never WatchdogSec — llama-server has no sd_notify" |
| CPUAffinity | Pin to subset of cores | "Use all 20 cores — don't restrict to efficiency cores" |
| Multi-node inference | "Use NCCL with mpirun" (wrong for llama.cpp) | llama.cpp RPC backend with `-DGGML_RPC=ON`, ConnectX-7 RoCE v2 |
| Ollama for inference | "Install Ollama for easy setup" | "Never Ollama — Go overhead, no UMA flags, no RPC, use llama.cpp directly" |

## Hardware Quick Reference

| Spec | Value |
|------|-------|
| Superchip | NVIDIA GB10 Grace-Blackwell |
| CPU | 20 ARM cores (10x Cortex-X925 + 10x Cortex-A725), aarch64 ARMv9.2-A |
| GPU | Blackwell, 48 SMs, sm_121 (compute capability 12.1) |
| Memory | 128 GB unified LPDDR5x (CPU + GPU shared) |
| CUDA | 13.0 |
| Key limitation | No FlashAttention, no WGMMA, no tcgen05 (uses Ampere-era mma.sync) |

## Sources

Built from official and community sources:

- [Arm Learning Path: llama.cpp on GB10](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/)
- [llama.cpp DGX Spark Benchmarks](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md)
- [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup)
- [GuigsEvt/dgx_spark_config](https://github.com/GuigsEvt/dgx_spark_config)
- [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup)
- [LMSYS In-Depth Review](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [NVIDIA DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/)
- [backend.ai: Is DGX Spark Actually Blackwell?](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)

## License

MIT
