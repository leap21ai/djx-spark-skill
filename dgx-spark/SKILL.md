---
name: dgx-spark
description: >
  Procedural knowledge for the NVIDIA DGX Spark (GB10 Grace-Blackwell Superchip).
  Use when building, deploying, training, or running inference on DGX Spark hardware —
  covers CUDA 13.0 compatibility, unified memory architecture, llama.cpp builds,
  vLLM/SGLang setup, OOM prevention, and performance optimization.
version: 1.1.0
author: Trey Anderson
tags: [nvidia, dgx-spark, gb10, blackwell, cuda-13, aarch64, llama-cpp, vllm, inference, training]
---

# NVIDIA DGX Spark (GB10) — Complete Procedural Skill

## When to Use This Skill

- Building or compiling **any** software targeting DGX Spark (GB10, sm_121, CUDA 13.0)
- Running LLM inference on DGX Spark (llama.cpp, vLLM, SGLang, Ollama)
- Training or fine-tuning models on DGX Spark (SFT, DPO, GRPO, LoRA)
- Diagnosing OOM freezes, swap death spirals, or unresponsive SSH on DGX Spark
- Setting up Docker containers for GPU workloads on DGX Spark
- Installing Python ML packages that fail with `libcudart.so.12` errors
- Optimizing memory usage on the 128 GB unified memory architecture
- Building PyTorch, Triton, or vLLM from source for sm_121

**Use the generic `llama-cpp` skill instead when**: targeting Apple Silicon (Metal), AMD ROCm, or standard NVIDIA datacenter GPUs (A100/H100/B200). This skill is specifically for the GB10's unique architecture.

---

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **Superchip** | NVIDIA GB10 Grace-Blackwell |
| **CPU** | 20 ARM cores: 10x Cortex-X925 (perf) + 10x Cortex-A725 (efficiency) |
| **CPU Architecture** | aarch64, ARMv9.2-A, SVE2, BF16, INT8 matrix multiply (I8MM) |
| **GPU** | NVIDIA Blackwell, 48 SMs, 6,144 CUDA cores, 192 5th-gen Tensor Cores |
| **Compute Capability** | **sm_121** (12.1) — NOT sm_120 (consumer) or sm_100 (datacenter) |
| **Memory** | 128 GB unified LPDDR5x, ~273 GB/s bandwidth |
| **CUDA-visible memory** | 119.68 GB (reported by runtime) |
| **Memory model** | CPU + GPU share same physical DRAM via NVLink-C2C (zero-copy) |
| **CUDA Version** | 13.0 (Toolkit 13.0.2) |
| **Driver** | 580.95.05+ (580.126.09 as of Jan 2026) |
| **OS** | Ubuntu 24.04 LTS, kernel 6.11 (6.17 recommended) |
| **Networking** | Dual QSFP (ConnectX-7), 200 Gb/s aggregate |
| **Peak Performance** | 1 PFLOP sparse FP4 tensor |

### sm_121 vs Other Blackwell Variants

| Aspect | SM100 (B200/GB200 Datacenter) | SM12x (GB10 DGX Spark) |
|--------|-------------------------------|------------------------|
| Shared Memory/SM | 228 KB | 128 KB |
| TMEM (Tensor Memory) | 256 KB per SM | **None** |
| Tensor Instructions | `tcgen05` (new) | Extended `mma.sync` (Ampere-era) |
| Warpgroup MMA | WGMMA supported | **Not supported** |
| FlashAttention | Supported | **Not supported** (use SDPA) |
| FlashAttention 4 | Supported | **Not supported** |
| FlashMLA | Supported | **Not supported** |

**Critical implication**: SM12x reverts to Ampere-era `mma.sync` with registers, only adding new numeric formats (FP4/FP6). Kernels requiring Hopper WGMMA or datacenter Blackwell `tcgen05` instructions will **fail silently or crash** on GB10.

---

## System Verification

Run these commands to confirm a healthy DGX Spark before any build or deployment.

```bash
# CPU — expect 20 cores, aarch64, Armv9-A
lscpu

# OS — expect Ubuntu 24.04 LTS
lsb_release -a

# GPU + driver — expect GB10, driver 580.x, CUDA 13.0
nvidia-smi

# CUDA toolkit — expect nvcc 13.0
nvcc --version

# Kernel version — 6.17+ recommended for fast model loads
uname -r
```

**Expected `nvidia-smi` output** (note: Memory-Usage shows "Not Supported" — this is normal for unified memory):

```
GPU: NVIDIA GB10 (On, Persistence-M)
Driver Version: 580.95.05      CUDA Version: 13.0
Memory-Usage: Not Supported
```

---

## CUDA 13.0 Compatibility — The #1 Pain Point

### The Problem

Almost every Python ML package on PyPI ships CUDA 12.x wheels linked to `libcudart.so.12`. DGX Spark only has `libcudart.so.13`. Importing any such package fails:

```
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

### The Fix: Always Use cu130 Wheels

```bash
# Create venv with uv (preferred)
uv venv .venv --python 3.12
source .venv/bin/activate

# PyTorch — ALWAYS use cu130 index
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# vLLM — ALWAYS use cu130 nightly wheels
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130

# NEVER install flash-attn (causes libcudart.so.12 errors)
# Use SDPA instead: --attn_implementation sdpa
```

### PyTorch sm_121 Warning — Safe to Ignore

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

sm_120 and sm_121 are **binary compatible**. Confirmed by PyTorch maintainers. This warning is harmless.

### Required Environment Variables

```bash
export TORCH_CUDA_ARCH_LIST="12.1a"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
```

### pyproject.toml for Cross-Platform Projects

```toml
[project]
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "vllm>=0.12.0; platform_machine != 'aarch64'",
    "vllm>=0.13.0; platform_machine == 'aarch64'",
    "flash-attn; platform_machine != 'aarch64'",  # NO flash-attn on DGX Spark
]

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

### Package Compatibility Matrix

| Package | Status | Notes |
|---------|--------|-------|
| PyTorch 2.9.0+cu130 | Working | sm_121 warning is safe to ignore |
| Transformers | Working | Standard HF, no special config |
| DeepSpeed | Working | Builds via `uv sync` |
| Accelerate | Working | `uv run python -m accelerate.commands.launch` |
| vLLM 0.13.0+cu130 | Working | Use nightly cu130 wheels or NGC container |
| FlashInfer 0.5.3+ | Working | Earlier versions fail |
| Triton 3.5.0+ | Working | Requires `TRITON_PTXAS_PATH`; 3.6.0 fixes SM12x fully |
| flash-attn | **Broken** | `libcudart.so.12` errors — skip entirely |
| FlashAttention 4 | **Not supported** | SM100-only |
| FlashMLA | **Not supported** | Requires tcgen05/WGMMA |
| CUTLASS FP8 kernels | **Broken** | Fail to dispatch on SM121 |

---

## Unified Memory Architecture

### How It Works

CPU and GPU share the **same 128 GB physical DRAM** pool connected via NVLink-C2C. There is no PCIe bus, no dedicated VRAM, no explicit `cudaMemcpy` required for most CPU-GPU data transfers. This is what allows 70B+ models to fit on a single node.

### Critical Rules

1. **`cudaMalloc` memory is NOT coherently CPU-accessible.** Pinned device memory cannot be read by the CPU complex or PCIe peripherals. For RDMA applications, use `cudaHostAlloc` + `ib_reg_mr`.

2. **`cudaMemGetInfo` underreports.** It doesn't account for reclaimable OS page cache. Always drop caches before memory-intensive operations:
   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   ```

3. **safetensors double-load bug.** The mmap loader first loads to "RAM" then copies to "VRAM" — on UMA this duplicates the model in memory. Workaround: `--load-format fastsafetensors` in vLLM (but don't use if model exceeds ~85% of available RAM).

4. **Linux buffer cache competes with GPU.** The kernel caches file I/O in the shared pool. Mount HuggingFace cache across containers to prevent redundant downloads. Drop caches between runs.

5. **`nvidia-smi` shows "Memory-Usage: Not Supported"** for total pool. Per-process GPU memory is listed. This is expected — integrated GPUs have no dedicated framebuffer.

### Technologies NOT Available on GB10

- GPUDirect RDMA
- nvidia-peermem (DOCA-Host)
- dma-buf
- GDRCopy

Detect at runtime: `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED`.

---

## Memory Safety — Preventing the Swap Death Spiral

### The Problem

This is the **#1 reported critical issue** on NVIDIA forums. Because GPU and CPU share 128 GB, exceeding available memory does NOT produce a clean `RuntimeError: CUDA out of memory`. Instead:

1. Training/inference exceeds available memory
2. OS starts swapping aggressively to NVMe
3. Swap thrashing spirals — SSH hangs, GUI freezes
4. Machine becomes a "zombie" requiring **hard reboot** (power cycle)

### Defense-in-Depth: 5 Layers

#### Layer 1: Disable Swap (MOST IMPORTANT)

```bash
sudo swapoff -a
swapon --show  # should print nothing

# Make permanent — comment out swap entries
sudoedit /etc/fstab
```

Converts "brick the box" into "job dies, OS lives."

#### Layer 2: Memory Jails via cgroups

```bash
sudo systemd-run --scope \
    -p MemoryMax=100G \
    -p MemorySwapMax=0 \
    -p OOMScoreAdjust=500 \
    bash -lc '<your-training-command>'
```

- `MemoryMax=100G`: hard cap (leaves ~19 GB for OS + other services)
- `MemorySwapMax=0`: no swap for this job
- `OOMScoreAdjust=500`: kernel prefers killing this over sshd

#### Layer 3: Protect SSH

```bash
sudo mkdir -p /etc/systemd/system/ssh.service.d
sudo tee /etc/systemd/system/ssh.service.d/oom.conf >/dev/null <<'EOF'
[Service]
OOMScoreAdjust=-1000
MemoryMin=512M
EOF
sudo systemctl daemon-reload && sudo systemctl restart ssh.service
```

Optional: install **Dropbear** on port 2222 as a backup SSH daemon (~500 KB, survives extreme memory pressure).

#### Layer 4: earlyoom Watchdog

**earlyoom is NOT preinstalled on DGX Spark** — you must install and enable it:

```bash
sudo apt install -y earlyoom
sudo systemctl enable --now earlyoom
# Configure: 3% RAM / 10% swap thresholds
# Protect: systemd, SSH, journald
# Kill targets: vllm, python, triton
```

Or a manual memory watchdog script:

```bash
#!/bin/bash
WATCH_PID=$1
THRESHOLD_KB=$((16 * 1024 * 1024))  # 16 GB
while true; do
    avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt "$THRESHOLD_KB" ]; then
        echo "MemAvailable low (${avail_kb}kB). Killing $WATCH_PID"
        kill -TERM "$WATCH_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$WATCH_PID" 2>/dev/null || true
        break
    fi
    sleep 1
done
```

#### Layer 5: Drop Caches Between Runs

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Pre-flight Checklist (Run Before Every Job)

```bash
free -h                                          # Verify >80 GB free
pkill -9 -f "ray::" 2>/dev/null || true          # Kill stale Ray processes
pkill -9 -f "VLLM" 2>/dev/null || true           # Kill stale vLLM
sleep 10
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
swapon --show                                    # Should print nothing
```

---

## llama.cpp — GPU Build (Recommended)

### Install Dependencies

```bash
sudo apt update
sudo apt install -y git cmake build-essential nvtop htop
```

### Clone and Build

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

mkdir -p build-gpu && cd build-gpu
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=nvcc

make -j"$(nproc)"
```

Build completes in 2-4 minutes.

### Verify Build

```bash
# Confirm CUDA linkage
ldd bin/llama-cli | grep cuda
# Expect: libcudart.so.13, libcublas.so.13

# Confirm compute capability
./bin/llama-server --version
# Expect: "compute capability 12.1"
```

### Optimal Runtime Flags

```bash
./bin/llama-server \
    -m ~/models/your-model.gguf \
    -ngl 99 \              # offload ALL layers to GPU
    --flash-attn \          # enable flash attention (llama.cpp's internal impl)
    --no-mmap \             # CRITICAL — significantly improves perf on UMA
    --mlock \               # lock model in memory, prevent page-outs
    --cont-batching \       # enable continuous batching for server mode
    -ub 2048 \              # micro-batch size (matches bench settings)
    -t 20 \                 # 20 threads (one per core)
    --host 0.0.0.0 \
    --port 8080
```

**`--no-mmap` is critical on DGX Spark** — mmap causes the kernel to manage model pages through the buffer cache, competing with GPU memory. Direct I/O (`--no-mmap`) bypasses this entirely.

### Key Build Flags Reference

| Flag | Purpose |
|------|---------|
| `-DGGML_CUDA=ON` | Enable CUDA backend |
| `-DGGML_CUDA_F16=ON` | FP16 kernels — reduces memory, improves perf |
| `-DCMAKE_CUDA_ARCHITECTURES=121` | Target GB10 Blackwell (sm_121) |

---

## llama.cpp — CPU Build (For CPU-Only Workloads)

Useful for embedding models, small classifiers, or when GPU is occupied.

```bash
cd ~/llama.cpp
mkdir -p build-cpu && cd build-cpu
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DLLAMA_ACCELERATE=ON \
    -DLLAMA_BLAS=OFF \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS="-O3 -march=armv9-a+sve2+bf16+i8mm -mtune=native -fopenmp" \
    -DCMAKE_CXX_FLAGS="-O3 -march=armv9-a+sve2+bf16+i8mm -mtune=native -fopenmp"

make -j"$(nproc)"
```

Build completes in ~20 seconds.

### CPU Flag Reference

| Flag | Purpose |
|------|---------|
| `-march=armv9-a` | Target ARMv9-A base architecture |
| `+sve2` | Scalable Vector Extension 2 (variable-length SIMD) |
| `+bf16` | BFloat16 arithmetic |
| `+i8mm` | INT8 matrix multiply (quantized inference) |
| `-mtune=native` | Optimize for local Grace CPU microarchitecture |
| `-fopenmp` | Multi-threaded execution across all 20 cores |
| `-DLLAMA_ACCELERATE=ON` | Enable llama.cpp's internal Arm Neon/SVE optimized kernels |

### CPU Inference

```bash
./bin/llama-cli \
    -m ~/models/model.gguf \
    -ngl 0 \     # disable GPU offloading
    -t 20 \      # use all 20 cores
    -p "Your prompt here"
```

Monitor with `htop` — expect 75-85% utilization across all 20 cores.

If build fails after changing flags:
```bash
cmake --fresh .
make -j"$(nproc)"
```

---

## vLLM Setup

### Option 1: NGC Container (Fastest)

```bash
docker pull nvcr.io/nvidia/vllm:25.11-py3

docker run --gpus all --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/vllm:25.11-py3 \
    vllm serve "Qwen/Qwen2.5-7B-Instruct" \
        --gpu-memory-utilization 0.7 \
        --load-format fastsafetensors
```

**Always set `--gpu-memory-utilization 0.7`** (not 0.9) — leave 30% headroom to prevent zombie OOM.

### Option 2: cu130 Wheels (Native Install)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

# PyTorch cu130
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# vLLM cu130 nightly
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130

# DO NOT install flash-attn
```

### Option 3: One-Command Setup (eelbaz)

```bash
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
```

Installs PyTorch 2.9.0+cu130, Triton 3.5.0+ from main, vLLM with Blackwell patches. ~20-30 minutes.

### Option 4: Build from Source

```bash
# Activate venv with PyTorch cu130 already installed
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_USE_FLASHINFER_MXFP4_MOE=1

git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .  # ~30 min on 20 cores
```

### vLLM Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.7,  # ALWAYS 0.7 on DGX Spark
)

outputs = llm.generate(
    ["Explain unified memory architecture"],
    SamplingParams(temperature=0.7, max_tokens=512),
)
print(outputs[0].outputs[0].text)
```

---

## SGLang Setup

### Official Spark Container

```bash
docker run --gpus all --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    lmsysorg/sglang:spark \
    python3 -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 --port 30000
```

### With Speculative Decoding (EAGLE3) — Up to 2x Throughput

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --mem-fraction 0.6 \
    --cuda-graph-max-bs 2 \
    --dtype float16 \
    --speculative-algorithm EAGLE3 \
    --host 0.0.0.0 --port 30000
```

**Note**: SGLang SM121a support is still on a development branch. The `spark` container image works but is not on stable main. GPT-OSS models may trigger PTXAS compilation errors related to shared memory.

---

## Docker / Container Setup

NVIDIA Container Toolkit is **preinstalled** on DGX Spark.

```bash
# Verify
nvidia-ctk --version
cat /etc/docker/daemon.json

# Add user to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER && newgrp docker
```

### Standard GPU Container

```bash
docker run -it --gpus=all \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    --ulimit memlock=-1 \
    --shm-size=32g \
    nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 bash
```

The `-v /usr/local/cuda:/usr/local/cuda:ro` bind-mount exposes the host CUDA 13 toolkit inside the container.

### Memory Flags for Inference Containers

```bash
--gpus all
--ulimit memlock=-1         # unlimited memory locking
--shm-size=32g              # SGLang needs 32g; 1g for lighter workloads
```

### Snapshot a Working Container

```bash
docker commit --pause=false <container_id> my-dgx-spark-env:snapshot
```

### Ollama (Simplest Path)

```bash
# Works out of the box on DGX Spark
ollama run qwen2.5:7b

# Recommended env vars
OLLAMA_MAX_LOADED_MODELS=1      # prevent memory contention on UMA
OLLAMA_FLASH_ATTENTION=1        # reduce memory footprint
```

---

## Training on DGX Spark

### Attention Implementation

**Always use SDPA**, never flash_attention_2:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",  # NOT "flash_attention_2"
)
```

CLI flag: `--attn_implementation sdpa`

### SFT Training

```bash
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

sudo systemd-run --scope \
    -p MemoryMax=100G \
    -p MemorySwapMax=0 \
    -p OOMScoreAdjust=500 \
    bash -lc 'source .venv/bin/activate && python scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name_or_path OpenAssistant/oasst2 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --output_dir ./sft_output'
```

### GRPO Training (with vLLM generation)

```bash
sudo systemd-run --scope \
    -p MemoryMax=100G \
    -p MemorySwapMax=0 \
    -p OOMScoreAdjust=500 \
    bash -lc 'source .venv/bin/activate && python scripts/grpo.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --num_generation_batches 4 \
    --vllm_gpu_memory_utilization 0.5 \
    --output_dir ./grpo_output'
```

### Memory Budget Reference (Qwen3-0.6B, seq_len=1024, grad_checkpointing=true)

| Task | Batch | Peak Memory | Headroom | Verdict |
|------|-------|-------------|----------|---------|
| SFT | 2 | 21 GB | 98 GB | Safe |
| SFT | 4 | 29 GB | 90 GB | Safe |
| SFT | 8 | 47 GB | 72 GB | **Recommended** |
| SFT | 16 | 81 GB | 38 GB | Limit |
| DPO | 2 | 24 GB | 95 GB | Safe |
| DPO | 4 | 27 GB | 92 GB | Safe |
| DPO | 8 | 62 GB | 57 GB | **Recommended** |

Memory scales **super-linearly** with batch size. Doubling batch 8->16 adds 34 GB, not 24 GB.

### Memory Estimation Formulas

```
SFT:  ~6 * model_params_B + activation_memory
DPO:  ~1.3 * SFT_memory
GRPO: SFT_memory + vllm_gpu_memory_utilization * 119 GB
```

---

## Performance Benchmarks (llama.cpp, Single GB10)

All benchmarks: `ngl=-1, flash_attn=1, n_ubatch=2048, mmap=0, dio=1, threads=20`.

### Prefill (pp2048) and Generation (tg32)

| Model | Format | Size | pp2048 t/s | tg32 t/s |
|-------|--------|------|-----------|---------|
| gemma-3-4B | Q4_0 | 2.35 GiB | **5,949** | **81** |
| gpt-oss-20B | MXFP4 | 11.27 GiB | **4,506** | **83** |
| Qwen3-Coder-30B-A3B | Q8_0 | 30.25 GiB | **2,987** | **61** |
| gpt-oss-120B | MXFP4 | 59.02 GiB | **2,444** | **59** |
| GLM-4.7-Flash | Q8_0 | 29.65 GiB | **2,364** | **49** |
| Qwen2.5-Coder-7B | Q8_0 | 7.54 GiB | **2,250** | **29** |

### Context Length Degradation (gpt-oss-120B MXFP4)

| Context Depth | pp2048 t/s | tg32 t/s |
|---------------|-----------|---------|
| d=0 (fresh) | 2,444 | 59 |
| d=4096 | 2,310 | 56 |
| d=8192 | 2,217 | 53 |
| d=16384 | 1,956 | 49 |
| d=32768 | 1,567 | 43 |

### SGLang Benchmarks (from LMSYS review)

| Model | Format | Batch | Prefill t/s | Decode t/s |
|-------|--------|-------|------------|-----------|
| Llama 3.1 8B | FP8 | 1 | 7,991 | 20.5 |
| Llama 3.1 8B | FP8 | 32 | 7,949 | 368 |
| Llama 3.1 70B | FP8 | 1 | 803 | 2.7 |
| GPT-OSS 20B | — | 1 | 2,053 | 49.7 |

### Comparative Reference

- **Mac M4 Max 64GB**: ~100-200 t/s prefill on similar models — GB10 is **5-10x faster on prefill**
- **Kernel upgrade impact**: model load time drops from ~68s (kernel 6.11) to ~22s (kernel 6.17)
- **NVFP4 vs AWQ**: ~20% faster across workloads; ~40% less memory
- **Speculative decoding (EAGLE3)**: up to **2x end-to-end throughput**

---

## Building PyTorch from Source (Advanced)

Only needed if cu130 wheels don't exist for your use case. Prefer `pip install --index-url https://download.pytorch.org/whl/cu130` first.

### System Dependencies

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build git curl wget pkg-config \
    python3 python3-dev python3-pip python3-setuptools python3-wheel python3-venv \
    libopenblas-dev libcublas-dev-13-0 libomp-dev \
    libopenmpi-dev mpi-default-bin libuv1-dev libssl-dev zlib1g \
    cudnn9-cuda-13-0
```

### Build Environment

```bash
export USE_CUDA=1
export USE_CUDNN=1
export USE_CUBLAS=1
export USE_CUSPARSELT=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1
export USE_FLASH_ATTENTION=0       # OFF on GB10 — use SDPA
export USE_MEM_EFF_ATTENTION=1

export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export CUDAARCHS="121"
export CMAKE_CUDA_ARCHITECTURES="120;121"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
```

### Clone and Build

```bash
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch && git checkout v2.9.1
git submodule update --init --recursive

pip install -r requirements.txt -r requirements-build.txt
pip wheel . -w dist --no-deps --verbose
pip install dist/torch-*.whl
```

### Verify

```python
import torch
print(torch.__version__)       # 2.9.1
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_properties(0))
# Expect: CUDA capability 12.1, GB10
```

---

## Building Triton from Source (Advanced)

Official Triton 3.5.0 has bugs with sm_121a. Build from main:

```bash
sudo apt install -y llvm-20-dev python3.12-dev

git clone https://github.com/triton-lang/triton.git
cd triton

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
pip install --no-build-isolation .
```

If building with custom LLVM:

```bash
cd ~/llvm-project
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX"
ninja

export LLVM_BUILD_DIR=$(pwd)
# Then build Triton wheel
```

---

## Compiler Flags Reference (CPU-side)

### Optimal for Grace CPU (Cortex-X925 / A725)

```bash
# Best — requires LLVM 21+ or GCC 15+
-mcpu=gb10

# Current recommended (GCC 12-14)
-march=armv9-a+sve2+bf16+i8mm -mtune=native

# OpenMP for multi-core
-fopenmp

# Full optimization
-O3 -march=armv9-a+sve2+bf16+i8mm -mtune=native -fopenmp
```

### Notes

- Cache line size: 64 bytes
- 2 CPU clusters (10 cores each) — minimize cross-cluster cache line conflicts
- Use `guided` or `dynamic` OpenMP scheduling for heterogeneous core workloads
- Profiling: Nsight Systems for GPU timeline, `perf` for ARM PMU counters

---

## Dual-Node Configuration (256 GB Combined)

Two DGX Sparks can interconnect via QSFP for 256 GB combined memory, handling models up to 405B in FP4.

### vLLM Dual-Node Benchmarks

| Model | Config | Throughput |
|-------|--------|-----------|
| gpt-oss-120B | MXFP4, single | 58.82 tok/s |
| gpt-oss-120B | MXFP4, dual | 75.96 tok/s |
| Qwen-235B | NVFP4, dual | 23,477 t/s prefill |

---

## Troubleshooting

### Machine becomes unresponsive / zombie
1. Swap was enabled → `sudo swapoff -a`
2. No memory cap → use `systemd-run --scope -p MemoryMax=100G`
3. Install earlyoom and Dropbear SSH
4. Reduce batch size or `--gpu-memory-utilization`

### `libcudart.so.12: cannot open shared object file`
- Using PyPI wheels compiled against CUDA 12 → switch to cu130 index
- `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`
- For stubborn packages: `sudo ln -sf /usr/local/cuda/lib64/libcudart.so.13 /usr/local/cuda/lib64/libcudart.so.12` (use with caution)

### `FATAL: kernel fmha_cutlassF_f16_aligned_* is for sm80-sm100, but was built for sm121`
- Flash attention kernels don't support SM12x → use SDPA: `--attn_implementation sdpa`
- Set `USE_FLASH_ATTENTION=0` when building PyTorch from source

### Training is very slow
1. Verify `--attn_implementation sdpa` (not flash_attention_2)
2. Check `watch -n 1 nvidia-smi` for GPU utilization
3. Ensure `--gradient_checkpointing` is enabled for large models

### llama.cpp slow model loading
- Upgrade kernel to 6.17+ (load time: 68s → 22s)
- Use `--no-mmap` flag

### CMake stale cache after changing flags
```bash
cmake --fresh .
make -j"$(nproc)"
```

### Triton build errors
- Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
- Use Triton main branch, not stable 3.5.0 release
- Non-editable build: `pip install --no-build-isolation .` (not `pip install -e`)

### vLLM MOE kernel errors (`undefined symbol: _Z20cutlass_moe_mm_sm100`)
- vLLM CMakeLists.txt missing SM12x in MOE kernel arch list
- Apply the eelbaz fix or use NGC container

---

## Systemd Service Template (llama.cpp Inference)

### Critical: No WatchdogSec

**NEVER use `WatchdogSec` with llama-server.** llama-server does not send `sd_notify` watchdog pings, so systemd will kill the process after the timeout expires — even if it's healthy. On large models (77GB+) with big context windows (131K+), the initial load can take 2-5 minutes, causing a **restart loop** (observed: 110 restarts in production before diagnosis). Use `Restart=on-failure` without any watchdog.

### CPUAffinity: Use All 20 Cores

Do **not** pin llama-server to a subset of cores (e.g., `CPUAffinity=10-19`). The GB10 has 10 performance cores (Cortex-X925, cores 0-9) and 10 efficiency cores (Cortex-A725, cores 10-19). Pinning to efficiency cores only halves throughput. Either omit `CPUAffinity` entirely (recommended) or set `CPUAffinity=0-19`.

### Load Time Expectations

Model load time depends on more than just the GGUF file size. The q8_0 KV cache for large context windows adds significant allocation time:

| Config | Approximate Load Time (kernel 6.17) |
|--------|--------------------------------------|
| 7B model, 32K ctx | ~5s |
| 30B model, 43K ctx | ~15s |
| 77GB model, 32K ctx | ~22s |
| 77GB model, 131K ctx, q8_0 KV | **2-3 minutes** |

### nvidia-smi on UMA

`nvidia-smi --query-gpu=memory.used,memory.total` returns `[N/A]` on DGX Spark (no dedicated VRAM). To check GPU memory per process, use:

```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
```

### Template

```ini
[Unit]
Description=llama.cpp server (%i) on DGX Spark
After=network.target

[Service]
Type=simple
User=nvidia
# Do NOT set WatchdogSec — llama-server has no sd_notify support
# Do NOT set CPUAffinity to a core subset — use all 20 cores
ExecStart=/usr/local/bin/llama-server \
    -m /models/%i.gguf \
    -ngl 99 \
    --flash-attn \
    --no-mmap \
    --mlock \
    --cont-batching \
    --parallel 2 \
    --ctx-size 32768 \
    -ub 2048 \
    -t 20 \
    --host 0.0.0.0 \
    --port 8080
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity
OOMScoreAdjust=500

[Install]
WantedBy=multi-user.target
```

Usage: `sudo systemctl enable --now llama-server@my-model`

---

## Environment Variables — Complete Reference

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# PyTorch
export TORCH_CUDA_ARCH_LIST="12.1a"

# Triton
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# vLLM
export VLLM_USE_FLASHINFER_MXFP4_MOE=1

# Ollama
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
```

---

## References

- [Arm Learning Path: llama.cpp on GB10](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/) — official 3-part build guide
- [llama.cpp DGX Spark Benchmarks](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md) — official bench data
- [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) — best ML training setup guide
- [GuigsEvt/dgx_spark_config](https://github.com/GuigsEvt/dgx_spark_config) — PyTorch/Triton from-source build
- [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup) — one-command vLLM installer
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — community vLLM Docker with model recipes
- [NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — official NVIDIA playbooks
- [LMSYS In-Depth Review](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) — comprehensive benchmarks
- [NVIDIA DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/) — official CUDA porting docs
- [NVIDIA Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) — official bug tracker
- [backend.ai: Is DGX Spark Actually Blackwell?](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell) — SM12x vs SM100 deep dive
