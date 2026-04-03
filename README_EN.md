# Lumen

> A lightweight deep learning core written in Rust, featuring dynamic autograd, flexible dtype management, quantization-aware loading, and a CPU-oriented Llama inference path.

[中文 README](./README_zh-CN.md) · [Repository README](./README.md)

---

## Overview

Lumen is a compact Rust deep learning project. The current release keeps:

- a reusable **core library** for tensors, autograd, layers, modules, losses, and optimizers;
- a **minimal runnable Llama inference CLI** in `src/main.rs`;
- an **offline safetensors quantization utility**;
- an optional **development-only benchmark tool** isolated behind a Cargo feature instead of shipping as a default release binary.

The repository is useful in two ways:

- as a **learning-oriented DL core** for understanding tensors, autograd, modules, and kernels in Rust;
- as a **small LLM inference skeleton** showing a Llama-style decoder, safetensors loading, tokenizer integration, KV-cache decoding, and low-memory loading options.

`src/main.rs` is intentionally a **small integration example**, not a full production CLI, serving stack, or universal launcher.

---

## Technical Highlights

- Dynamic autograd engine built around tensor-based graph construction
- `Module`-style abstraction for trainable components
- Separated layer / op design for cleaner modeling and kernel evolution
- Flexible precision system with:
  - global defaults plus local overrides
  - separate parameter dtype and runtime dtype
  - optional cached parameter dtype copies
- Storage and loading support for `f32`, `f16`, `bf16`, and `i8`
- Quantization-aware parameter handling:
  - direct float-to-`i8` quantized loading
  - post-load parameter quantization
  - automatic or manual quantization scale
- Llama-family decoder implementation with RMSNorm, RoPE, causal self-attention, GQA, SwiGLU-style MLP, and KV-cache incremental decoding
- CPU-oriented inference hot-path optimization
- Safetensors loading through both memory-mapped and streamed paths
- Hugging Face `tokenizers` integration
- Release profile tuned with `lto`, `panic = "abort"`, and `strip`

---

## Repository Layout

```text
src/
├─ autograd.rs                  # Tensor + dynamic autograd core
├─ module.rs                    # Module trait/macros
├─ ops/                         # Tensor ops and kernels
├─ layers/                      # NN layers and attention components
├─ models/llama.rs              # Llama model implementation
├─ loader.rs                    # Safetensors loading and streamed loading
├─ tokenizer.rs                 # Hugging Face tokenizer wrapper
├─ kv_cache.rs                  # Legacy/simple KV cache implementation
├─ optim.rs                     # Optimizers
├─ loss.rs                      # Loss functions
├─ init.rs                      # Parameter initialization helpers
├─ main.rs                      # Minimal local inference CLI
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

To better utilize your local CPU:

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

## Running the Example

Run the example CLI with explicit weight and tokenizer paths:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Useful optional arguments:

- `--parameter-dtype f32|f16|bf16|i8`
- `--runtime-dtype f32|f16|bf16`
- `--activation-dtype f32|f16|bf16`
- `--kv-cache-dtype f32|f16|bf16`
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

Example: explicitly split parameter / activation / KV-cache dtypes:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --parameter-dtype i8 \
  --activation-dtype bf16 \
  --kv-cache-dtype bf16
```

By default, the release CLI keeps startup logs intentionally concise:

- a short model-loading summary
- a checkpoint loading summary
- the ready prompt

If you need backend diagnostics while tuning kernels, you can opt in:

```bash
LUMEN_SHOW_BACKENDS=1 cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
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

The quantization tool keeps tensor-data decoding safe and alignment-independent, so it does not rely on unchecked byte reinterpretation.

---

## Benchmark Tool

Kernel benchmarks are still kept in the repository, but they are now isolated as a development tool rather than a default release artifact:

```bash
cargo run --release --features dev-tools --bin kernel_bench -- --iters 200 --samples 5
```

This keeps the release build smaller while preserving the benchmark workflow for kernel work.

---

## Important Note on `main.rs`

`src/main.rs` uses a **hard-coded `model_config()`** and a lightweight CLI flow. This is intentional: it keeps the example easy to inspect and modify.

But it also means:

- the architecture in `model_config()` must match the loaded checkpoint;
- the tokenizer vocabulary and special tokens must be compatible with `vocab_size` and prompt formatting;
- adapting to other checkpoints may require changing dimensions, layer counts, attention layout, special tokens, and prompt-template logic.

So, **`main.rs` should be understood as a simple integration example, not a universal launcher**.

---

## Why This Project Is Interesting

Many Rust ML repositories stop at tensors or MLP demos. Lumen goes further by connecting multiple layers of the stack in one codebase:

- tensor + autograd fundamentals
- reusable NN modules
- dtype-aware parameter storage
- quantization-aware checkpoint loading
- a Llama decoder implementation
- tokenizer bridging
- stateful autoregressive decoding

That makes it a solid starting point for studying how a small Rust-native DL / LLM runtime can be assembled end to end.

---

## License

GPL v3.0
