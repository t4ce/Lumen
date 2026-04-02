# Lumen

> 一个使用 Rust 编写的轻量级深度学习核心，包含动态自动微分、灵活的 dtype 管理、量化感知加载，以及面向 CPU 的 Llama 推理路径。

[English README](./README_EN.md) · [仓库首页 README](./README.md)

---

## 项目简介

Lumen 是一个用 Rust 编写的紧凑型深度学习项目。当前这个发布版保留了：

- 一个可复用的**核心库**，包含张量、autograd、层、模块、损失函数与优化器；
- 一个最小可运行的 **Llama 推理 CLI**，位于 `src/main.rs`；
- 一个用于生成量化 checkpoint 的 **离线 safetensors 量化工具**；
- 一个通过 Cargo feature 隔离的**开发期 benchmark 工具**，而不是默认发布二进制的一部分。

这个仓库适合两类用途：

- 作为一个**学习型 DL Core**，用于理解 Rust 中的张量、自动微分、模块与 kernel；
- 作为一个**小型 LLM 推理骨架**，展示 Llama 风格解码器、safetensors 权重加载、tokenizer 接入、KV cache 解码，以及低峰值内存加载方式。

`src/main.rs` 被有意保持为一个**小型集成示例**，而不是完整的生产级 CLI、服务框架或通用启动器。

---

## 技术特点

- 动态自动微分引擎，基于张量构建计算图
- `Module` 风格抽象，便于组织可训练组件
- 层 / 算子分层设计，便于建模和 kernel 演进
- 灵活的精度系统，支持：
  - 全局默认值与局部覆盖
  - 参数 dtype 与运行时 dtype 分离
  - 可选的参数多 dtype 缓存副本
- 支持 `f32`、`f16`、`bf16`、`i8` 的存储与加载
- 量化感知的参数处理，支持：
  - 浮点权重直接加载为 `i8`
  - 加载后再量化
  - 自动或手动量化 scale
- Llama 系列解码器实现，包括 RMSNorm、RoPE、因果自注意力、GQA、SwiGLU 风格 MLP，以及基于 KV cache 的增量解码
- 面向 CPU 的推理热路径优化
- 同时支持基于 `memmap` 的 safetensors 加载和流式加载
- 通过 Hugging Face `tokenizers` 接入 tokenizer
- `release` 配置启用 `lto`、`panic = "abort"`、`strip`

---

## 仓库结构

```text
src/
├─ autograd.rs                  # Tensor 与动态自动微分核心
├─ module.rs                    # Module trait / 宏
├─ ops/                         # 张量算子与 kernel
├─ layers/                      # 神经网络层与注意力组件
├─ models/llama.rs              # Llama 模型实现
├─ loader.rs                    # Safetensors 加载与流式加载
├─ tokenizer.rs                 # Hugging Face tokenizer 封装
├─ kv_cache.rs                  # 旧版 / 简化 KV cache 实现
├─ optim.rs                     # 优化器
├─ loss.rs                      # 损失函数
├─ init.rs                      # 参数初始化辅助
├─ main.rs                      # 本地推理最小 CLI
└─ bin/
   ├─ quantize_safetensors.rs   # 离线量化工具
   └─ kernel_bench.rs           # 仅开发期使用的 benchmark 工具
```

---

## 构建

建议使用 release 模式构建：

```bash
cargo build --release
```

为了更好利用本机 CPU 指令集，可以额外启用：

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell：

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

默认 release 构建会生成面向用户的二进制：

- `lumen`
- `quantize_safetensors`

benchmark 二进制不会默认编进发布构建。若需要它，请显式开启：

```bash
cargo build --release --features dev-tools --bin kernel_bench
```

---

## 最小示例运行方式

用显式的权重和 tokenizer 路径运行示例 CLI：

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

常用可选参数：

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

示例：在低内存模式下，将浮点 checkpoint 在加载时直接量化为 `i8`：

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --runtime-dtype bf16 \
  --quantize i8 \
  --stream-weights
```

示例聊天循环支持的命令：

- `/reset` —— 清空对话状态与 KV cache
- `/exit` —— 退出程序

---

## 离线量化工具

如果你希望提前生成量化后的 safetensors：

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

也可以手动指定量化 scale：

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8 \
  --scale 0.02
```

---

## Benchmark 工具

Kernel benchmark 仍然保留在仓库中，但现在被隔离成开发工具，不属于默认发布产物：

```bash
cargo run --release --features dev-tools --bin kernel_bench -- --iters 200 --samples 5
```

这样既能保留 kernel 调优工作流，又不会让正式发布构建混入开发工具。

---

## 关于 `main.rs` 的重要说明

`src/main.rs` 使用了**硬编码的 `model_config()`** 和轻量 CLI 流程。这是有意为之：它更容易阅读和修改。

但这也意味着：

- `model_config()` 中的模型结构必须与你加载的权重匹配；
- tokenizer 的词表和特殊 token 必须与 `vocab_size` 以及 prompt 格式兼容；
- 若要适配其他 checkpoint，通常需要调整维度、层数、注意力布局、特殊 token，甚至 prompt template。

也就是说，**`main.rs` 是一个简单集成示例，不是通用启动器。**

---

## 这个项目的价值

很多 Rust 机器学习项目只做到张量层或 MLP 示例；而 Lumen 把下面这些部分串在了一起：

- 张量与 autograd 基础能力
- 可复用神经网络模块
- dtype 感知的参数存储
- 量化感知的 checkpoint 加载
- Llama 解码器结构
- tokenizer 桥接
- 有状态自回归解码

因此，它不仅适合阅读，也适合作为一个小型 Rust 原生 DL / LLM runtime 的起点。

---

## License

GPL v3.0
