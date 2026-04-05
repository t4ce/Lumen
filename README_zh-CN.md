# Lumen

> 一个紧凑的 Rust 深度学习核心，包含动态自动微分、灵活的 dtype 控制、safetensors 加载、量化感知推理，以及面向 CPU 的 Llama 运行时。

[English README](./README_EN.md) · [仓库首页 README](./README.md)

---

## 这是什么项目

Lumen 是一个“小而完整”的 Rust ML 工程，试图在一个仓库里把几层东西连起来：

- 张量核心与动态 autograd；
- 层、模块、损失函数与优化器；
- 一个 Llama 风格的解码器；
- safetensors 权重加载与流式加载；
- 参数、激活、KV cache 的运行时 dtype 控制；
- 加载时量化与离线 `i8` 量化；
- 面向 CPU 的推理 kernel 与 benchmark 工具。

它更适合被理解为：

- 一个**学习型 Rust 深度学习核心**；
- 一个围绕紧凑 Llama runtime 搭起来的**CPU LLM 推理实验场**。

它**不是**完整训练框架，不是成熟服务栈，也不是适配任意 checkpoint 的通用启动器。

---

## 当前重点

这个项目当前最有特色的方向是 **CPU 推理**。

目前仓库里比较值得关注的部分有：

- 动态自动微分与通用张量算子；
- 包含 RMSNorm、RoPE、GQA、SwiGLU 风格 MLP、KV-cache decode 的 Llama 解码器；
- `f32` / `f16` / `bf16` / `i8` 的存储、加载与运行时配置；
- 可选的参数多 dtype 缓存副本；
- 可选的流式权重加载；
- 面向 kernel 调优的微基准和端到端 benchmark。

---

## 特点

- **纯 Rust** 实现
- 基于张量图构建的**动态自动微分**
- 便于组织模型组件的 **Module 风格抽象**
- 清晰拆分的 **layers / ops / models** 结构
- 灵活的精度系统，可分别控制：
  - 参数 dtype
  - runtime dtype
  - activation dtype
  - KV-cache dtype
- 支持 `f32`、`f16`、`bf16`、`i8`
- 支持量化感知加载与运行时配置
- 包含 RMSNorm、RoPE、GQA、SwiGLU 风格 MLP 的 Llama-family decoder
- safetensors 支持 `memmap` 与流式加载
- 集成 Hugging Face `tokenizers`
- 提供仅开发期使用的 kernel / end-to-end benchmark 工具

---

## 仓库结构

```text
src/
├─ autograd.rs                  # Tensor + 动态自动微分核心
├─ module.rs                    # Module trait / 宏
├─ loader.rs                    # Safetensors 加载与流式加载
├─ tokenizer.rs                 # Tokenizer 封装
├─ kv_cache.rs                  # KV cache 实现
├─ precision.rs                 # DType / 运行时精度配置
├─ ops/                         # 张量算子与 CPU kernel
├─ layers/                      # 神经网络层与注意力组件
├─ models/llama.rs              # Llama 模型实现
├─ main.rs                      # 本地推理最小 CLI
└─ bin/
   ├─ quantize_safetensors.rs   # 离线量化工具
   ├─ kernel_bench.rs           # 仅开发期 kernel benchmark
   └─ prefill_decode_bench.rs   # 仅开发期端到端 benchmark
```

---

## 构建

```bash
cargo build --release
```

为了更好利用本机 CPU 指令集，可以这样构建：

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell：

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

默认 release 构建会生成：

- `lumen`
- `quantize_safetensors`

benchmark 二进制通过 `dev-tools` 隔离：

```bash
cargo build --release --features dev-tools --bin kernel_bench
cargo build --release --features dev-tools --bin prefill_decode_bench
```

---

## 运行最小推理 CLI

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

常用参数：

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

示例：使用 BF16 运行：

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

示例：`i8` 权重 + BF16 runtime：

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

如果你在调 kernel，想看后端诊断信息：

```bash
LUMEN_SHOW_BACKENDS=1 cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

交互命令：

- `/reset` —— 清空对话状态与 KV cache
- `/exit` —— 退出

---

## 离线量化

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

可选的手动 scale：

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8 \
  --scale 0.02
```

---

## Benchmark 工具

### Kernel benchmark

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench -- \
  --iters 400 --samples 7 --hidden 2048 --inter 5632 --vocab 32000
```

### 端到端 prefill/decode benchmark

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

## 当前基线性能

下面这组数据来自当前 **AVX-512 基线**，而且已经在作者机器上确认可以启用 BF16 kernel。

### Kernel 级别快照

调优过程中出现过的代表性结果：

- `backend: float=x86-avx512 int8=x86-avx2`
- `avx512_bf16_available=true`
- `matvec_bf16io ≈ 104 us`
- `fused_qkv ≈ 90 us`

这不是对所有 CPU 的普适承诺，只是一个已经跑通的本机基线快照。

### 端到端快照

在 `prompt_tokens=60`、`max_gen=128`、`runs=5`、`warmup=1` 下：

| 配置 | Prefill forward | Decode forward | End-to-end decode |
|---|---:|---:|---:|
| BF16 | 140.70 tok/s | 19.09 tok/s | 17.64 tok/s |
| F16 | 131.89 tok/s | 14.99 tok/s | 14.04 tok/s |
| F32 | 44.56 tok/s | 11.18 tok/s | 9.86 tok/s |
| I8 权重 + BF16 runtime | **203.66 tok/s** | **25.13 tok/s** | **23.16 tok/s** |

这组数据在当前实现上的实际结论是：

- **BF16** 是推荐的浮点主路径；
- **I8 权重 + BF16 runtime** 是目前测到的最快配置；
- **F16 不是当前这套实现的主优化目标**，因为它目前打不过 BF16。

---

## 设计说明与限制

`src/main.rs` 故意使用**硬编码 `model_config()`** 和轻量 CLI，这让它更容易读，也更容易改，但也意味着：

- 加载的 checkpoint 必须和硬编码结构匹配；
- 如果换别的模型，可能需要改 hidden size、层数、head 布局、prompt 格式等；
- 这更像一个本地实验 runner，不是通用推理前端。

同样，benchmark 工具的定位也是**开发与 kernel 调优**，不是成熟的公开 benchmark 基础设施。

---

## 适合谁

如果你想做下面这些事，Lumen 会比较合适：

- 学习 Rust 里的 tensor/autograd core 该怎么组织；
- 看一个没有被巨型框架包起来的小型 Llama runtime；
- 试验 dtype 管理、量化和 CPU 推理 kernel；
- 在自己的机器上 benchmark 和调优一个紧凑的 Rust 推理栈。

---

## License

本仓库遵循 [`LICENSE`](./LICENSE) 中给出的许可协议。
