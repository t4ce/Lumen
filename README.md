# Lumen

> A compact Rust deep learning core with dynamic autograd, flexible dtype control, safetensors loading, quantization-aware inference, and a CPU-oriented Llama runtime.

[中文说明](./README_zh-CN.md) · [English README](./README_EN.md)

---

## What this project is

Lumen is a **small but complete Rust ML stack** that connects several layers of the system in one repository:

- a tensor core with dynamic autograd;
- reusable layers, modules, losses, and optimizers;
- a Llama-style decoder implementation;
- safetensors loading with optional streaming;
- runtime dtype control for parameters, activations, and KV cache;
- optional on-load or offline `i8` quantization;
- CPU inference kernels and benchmark tools for kernel work.

This repository is best understood as:

- a **learning-oriented deep learning core** written in Rust, and
- a **CPU LLM inference playground** centered on a compact Llama runtime.

It is **not** trying to be a full training framework, a production serving stack, or a universal launcher for arbitrary checkpoints.

---

## Current focus

The project currently leans hardest into **CPU inference**.

Notable parts of the current codebase:

- dynamic autograd and general tensor ops;
- a Llama-family decoder with RMSNorm, RoPE, GQA, SwiGLU-style MLP, and KV-cache decoding;
- support for `f32`, `f16`, `bf16`, and `i8` in storage / loading / runtime configuration;
- optional parameter dtype copies for faster mixed-precision execution;
- optional streamed weight loading for lower peak memory usage;
- a development-only kernel benchmark and an end-to-end prefill/decode benchmark.

---

## Highlights

- **Pure Rust** implementation
- **Dynamic autograd** built around tensor graph construction
- **Module-style abstraction** for model components
- **Separated layers / ops / models** for easier experimentation
- **Flexible precision system**
  - parameter dtype
  - runtime dtype
  - activation dtype
  - KV-cache dtype
- **Quantization-aware loading**
  - load float weights normally
  - quantize on load to `i8`
  - generate offline quantized safetensors
- **CPU-oriented inference path** with explicit kernel/backend work
- **Hugging Face `tokenizers`** integration
- **Safetensors** support with memory-mapped and streamed loading modes
- Release profile tuned with `lto`, `panic = "abort"`, and `strip`

---

## Repository layout

```text
src/
├─ autograd.rs                  # Tensor + dynamic autograd core
├─ module.rs                    # Module trait / macros
├─ loader.rs                    # Safetensors loading and streamed loading
├─ tokenizer.rs                 # Tokenizer wrapper
├─ kv_cache.rs                  # KV cache implementation
├─ precision.rs                 # DType / runtime precision configuration
├─ ops/                         # Tensor ops and CPU kernels
├─ layers/                      # Neural-network layers and attention building blocks
├─ models/llama.rs              # Llama model implementation
├─ main.rs                      # Minimal local inference CLI
└─ bin/
   ├─ quantize_safetensors.rs   # Offline quantization utility
   ├─ kernel_bench.rs           # Dev-only kernel benchmark
   └─ prefill_decode_bench.rs   # Dev-only end-to-end benchmark
```

---

## Build

Release build:

```bash
cargo build --release
```

For better local CPU codegen:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell:

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

Default release builds produce:

- `lumen`
- `quantize_safetensors`

Development benchmarks are intentionally gated behind `dev-tools`:

```bash
cargo build --release --features dev-tools --bin kernel_bench
cargo build --release --features dev-tools --bin prefill_decode_bench
```

---

## Running the minimal inference CLI

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Useful flags:

- `--system TEXT`
- `--temperature FLOAT`
- `--top-p FLOAT`
- `--repetition-penalty FLOAT`
- `--recent-window N`
- `--max-gen N`
- `--parameter-dtype f32|f16|bf16|i8`
- `--runtime-dtype f32|f16|bf16`
- `--activation-dtype f32|f16|bf16|i8`
- `--kv-cache-dtype f32|f16|bf16`
- `--quantize off|i8`
- `--quant-scale FLOAT`
- `--allow-parameter-copies`
- `--stream-weights`
- `--max-seq-len N`
- `--load-only`

Example: BF16 runtime:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --parameter-dtype bf16 \
  --runtime-dtype bf16 \
  --activation-dtype bf16 \
  --kv-cache-dtype bf16 \
  --allow-parameter-copies
```

Example: `i8` weights with BF16 runtime:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --parameter-dtype i8 \
  --runtime-dtype bf16 \
  --activation-dtype i8 \
  --kv-cache-dtype bf16 \
  --quantize i8 \
  --allow-parameter-copies
```

You can print backend diagnostics during startup with:

```bash
LUMEN_SHOW_BACKENDS=1 cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Interactive commands:

- `/reset` — clear chat history and KV cache
- `/exit` — quit

---

## Offline quantization

Generate an `i8` safetensors checkpoint ahead of time:

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

Optional manual scale:

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8 \
  --scale 0.02
```

---

## Benchmark tools

### Kernel benchmark

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench -- \
  --iters 400 --samples 7 --hidden 2048 --inter 5632 --vocab 32000
```

### End-to-end prefill/decode benchmark

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin prefill_decode_bench -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --prompt "Explain Transformer KV cache." \
  --runs 5 --warmup 1 --max-gen 128 --mode sample \
  --parameter-dtype bf16 \
  --runtime-dtype bf16 \
  --activation-dtype bf16 \
  --kv-cache-dtype bf16 \
  --allow-parameter-copies
```

---

## Representative performance on the current baseline

The following numbers come from the current **AVX-512 baseline** that successfully enables BF16 kernels on the author's machine.

### Kernel-level snapshot

Representative results observed during tuning:

- `backend: float=x86-avx512 int8=x86-avx2`
- `avx512_bf16_available=true`
- `matvec_bf16io ≈ 104 us`
- `fused_qkv ≈ 90 us`

These are not universal claims for every CPU. They are a snapshot of one working baseline on one machine.

### End-to-end snapshot

For a run with `prompt_tokens=60`, `max_gen=128`, `runs=5`, `warmup=1`:

| Configuration | Prefill forward | Decode forward | End-to-end decode |
|---|---:|---:|---:|
| BF16 | 140.70 tok/s | 19.09 tok/s | 17.64 tok/s |
| F16 | 131.89 tok/s | 14.99 tok/s | 14.04 tok/s |
| F32 | 44.56 tok/s | 11.18 tok/s | 9.86 tok/s |
| I8 weights + BF16 runtime | **203.66 tok/s** | **25.13 tok/s** | **23.16 tok/s** |

Practical takeaway on that machine:

- **BF16** is the recommended floating-point path.
- **I8 weights + BF16 runtime** is the fastest tested configuration so far.
- **F16 is currently not the main optimization target**, since it underperforms BF16 in this implementation.

---

## Design notes and limitations

`src/main.rs` intentionally uses a **hard-coded `model_config()`** and a lightweight CLI. That keeps the example easy to inspect, but it also means:

- the architecture must match the loaded checkpoint;
- adapting to a different model may require editing dimensions, layer counts, KV-head layout, or prompt formatting;
- this is a compact local runner, not a universal inference frontend.

Similarly, the benchmark tools are intended for **development and kernel tuning**, not polished public benchmarking infrastructure.

---

## Who this project is for

Lumen is a good fit if you want to:

- learn how a Rust tensor/autograd core can be structured;
- study a small Llama runtime without a huge framework wrapped around it;
- experiment with dtype management, quantization, and CPU inference kernels;
- benchmark and tune a compact Rust inference stack on your own machine.

It is probably **not** the right fit if you need:

- large-scale training features;
- a mature serving system;
- GPU-first deployment tooling;
- plug-and-play support for arbitrary model families.

---

## License

This repository is released under the license included in [`LICENSE`](./LICENSE).
