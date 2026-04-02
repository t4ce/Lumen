# Lumen

> A lightweight Rust deep learning core with dynamic autograd, flexible dtype management, quantization-aware weight loading, and a CPU-oriented Llama inference path.

[中文 README](./README_zh-CN.md) · [English README](./README_EN.md)

---

## Overview

Lumen is a compact deep learning project written in Rust. The current release keeps:

- a reusable **core library** for tensors, autograd, layers, modules, losses, and optimizers;
- a **minimal runnable Llama inference CLI** in `src/main.rs`;
- an **offline safetensors quantization utility** for generating quantized checkpoints;
- an optional **development-only benchmark tool** kept behind a Cargo feature instead of being shipped as a default release binary.

The repository is useful in two roles:

- as a **learning-oriented DL core** for understanding how tensors, autograd, modules, and kernels fit together in Rust;
- as a **small LLM inference skeleton** centered on a Llama-style decoder, safetensors loading, tokenization, KV cache decoding, and low-memory loading options.

`src/main.rs` is intentionally a **small integration example**, not a full production model runner or serving stack.

---

## Technical Highlights

- Dynamic autograd engine with tensor-based graph construction
- `Module`-style neural network abstraction
- Layer / op separation for cleaner modeling and kernel evolution
- Flexible precision system with:
  - global defaults plus local overrides
  - distinct parameter dtype and runtime dtype
  - optional cached parameter dtype copies
- Storage and loading support for `f32`, `f16`, `bf16`, and `i8`
- Quantization-aware parameter handling:
  - direct float-to-`i8` quantized loading
  - post-load parameter quantization
  - automatic or manual quantization scale
- Llama-family decoder implementation with RMSNorm, RoPE, causal self-attention, GQA, SwiGLU-style MLP, and incremental decoding via KV cache
- CPU-oriented inference hot paths with parallel decode kernels and fused infer-time paths
- Safetensors loading through either:
  - memory-mapped loading for normal use
  - streamed loading for tighter peak-memory control
- Hugging Face `tokenizers` integration
- Release profile tuned with `lto`, `panic = "abort"`, and `strip`

---

## Repository Layout

```text
src/
├─ autograd.rs                  # Tensor + dynamic autograd core
├─ module.rs                    # Module trait/macros
├─ ops/                         # Tensor ops and kernels
├─ layers/                      # NN layers and attention building blocks
├─ models/llama.rs              # Llama model implementation
├─ loader.rs                    # Safetensors loading and streamed loading
├─ tokenizer.rs                 # Hugging Face tokenizer wrapper
├─ kv_cache.rs                  # Legacy/simple KV cache implementation
├─ optim.rs                     # Optimizers
├─ loss.rs                      # Loss functions
├─ init.rs                      # Parameter initialization helpers
├─ main.rs                      # Minimal runnable local inference CLI
└─ bin/
   ├─ quantize_safetensors.rs   # Offline quantization utility
   └─ kernel_bench.rs           # Dev-only benchmark tool
```

---

## Build

Use release mode for practical performance:

```bash
cargo build --release
```

For better CPU utilization on your local machine:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell:

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

Default release builds ship the user-facing binaries:

- `lumen`
- `quantize_safetensors`

The benchmark binary is intentionally excluded from default builds. To build it:

```bash
cargo build --release --features dev-tools --bin kernel_bench
```

---

## Example Usage

Run the example CLI with explicit weight and tokenizer paths:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Useful optional arguments:

- `--parameter-dtype f32|f16|bf16|i8`
- `--runtime-dtype f32|f16|bf16`
- `--quantize off|i8`
- `--quant-scale FLOAT`
- `--stream-weights`
- `--allow-parameter-copies`
- `--max-seq-len N`
- `--system`
- `--temperature`
- `--top-p`
- `--repetition-penalty`
- `--recent-window`
- `--max-gen`
- `--load-only`

Example: low-memory float checkpoint loading with on-load `i8` quantization:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --runtime-dtype bf16 \
  --quantize i8 \
  --stream-weights
```

Interactive commands in the example chat loop:

- `/reset` — clear conversation state and KV cache
- `/exit` — quit the program

---

## Offline Quantization Utility

To generate a quantized safetensors checkpoint ahead of time:

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

## Benchmark Tool

Kernel benchmarks are still kept in the repository, but they are now isolated as a development tool rather than a default release artifact:

```bash
cargo run --release --features dev-tools --bin kernel_bench -- --iters 200 --samples 5
```

This keeps the release build smaller while preserving the benchmark workflow for kernel work.

---

## Important Note About `main.rs`

`src/main.rs` uses a **hard-coded `model_config()`** and a lightweight CLI flow. This is deliberate: it keeps the example easy to read.

But it also means:

- the model architecture in `model_config()` must match the loaded weights;
- tokenizer vocabulary and special tokens must be compatible with the configured `vocab_size` and prompt format;
- adapting to other checkpoints may require updating dimensions, layer counts, attention layout, special tokens, or prompt-template logic.

In other words, **`main.rs` is a simple integration example, not a universal launcher**.

---

## Why This Project Stands Out

Unlike many toy Rust ML repos that stop at tensors or MLP demos, Lumen connects several layers of the stack together in one codebase:

- tensor + autograd fundamentals
- reusable NN modules
- dtype-aware parameter storage
- quantization-aware checkpoint loading
- Llama decoder architecture
- tokenizer bridging
- stateful autoregressive decoding

That makes it a good foundation for studying how a small Rust-native DL / LLM runtime can be assembled end to end.

---

## License

GPL v3.0
