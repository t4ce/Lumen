# Lumen

> 一个使用 Rust 编写的轻量级深度学习核心，包含动态自动微分、模块化神经网络组件，以及面向 CPU 的 Llama 推理路径。

[English README](./README_EN.md) · [仓库首页 README](./README.md)

---

## 项目简介

Lumen 是一个用 Rust 编写的紧凑型深度学习项目。当前这个发布版保留了**核心库代码**与**一个最小可运行的 Llama 推理示例**，并移除了开发阶段使用的测试、基准、计时和扫参代码。

这个仓库适合两类用途：

- 作为一个 **学习型 DL Core**，用于理解 Rust 中的张量、自动微分、层、模块和优化器；
- 作为一个 **小型 LLM 推理骨架**，展示 Llama 风格模型、safetensors 权重加载、tokenizer 接入，以及基于 KV cache 的增量解码。

> `src/main.rs` **只是一个简单示例程序**。它的作用是演示如何把库串起来跑通本地推理，而**不是**完整的生产级 CLI、服务框架，或通用模型启动器。

---

## 技术特点

- **动态自动微分引擎**：基于张量构建计算图并执行反向传播
- **`Module` 风格抽象**：方便组织可训练模块与网络结构
- **层 / 算子分层设计**：上层建模和底层计算实现解耦，便于演进
- **Llama 系列解码器实现**，包括：
  - RMSNorm
  - RoPE
  - 因果自注意力
  - GQA（`num_key_value_heads`）
  - SwiGLU 风格 MLP
  - 基于 **KV Cache** 的增量解码
- **面向 CPU 的推理热路径优化**，包括：
  - 基于 Rayon 的并行计算
  - 面向 decode 场景的 row-major 并行 matvec 路径
  - MLP 中推理态 fused gate/up/SiLU 路径
  - `release` 配置启用 `lto`、`panic = "abort"`、`strip`
- 通过 **`safetensors` + `memmap2`** 进行高效权重加载
- 通过 Hugging Face **`tokenizers`** 接入 `tokenizer.json`

---

## 仓库结构

```text
src/
├─ autograd.rs          # Tensor 与动态自动微分核心
├─ module.rs            # Module trait / 宏
├─ ops/                 # 张量算子与底层 kernel
├─ layers/              # 神经网络层与注意力组件
├─ models/llama.rs      # Llama 模型实现
├─ loader.rs            # Safetensors 权重加载
├─ tokenizer.rs         # Hugging Face tokenizer 封装
├─ kv_cache.rs          # 旧版 / 简化 KV cache 实现
├─ optim.rs             # 优化器
├─ loss.rs              # 损失函数
├─ init.rs              # 参数初始化
└─ main.rs              # 本地推理最小示例入口
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

---

## 最小示例运行方式

示例程序需要显式提供权重文件和 tokenizer：

```bash
cargo run --release -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

可选参数：

- `--system`
- `--temperature`
- `--top-p`
- `--repetition-penalty`
- `--recent-window`
- `--max-gen`

示例聊天循环支持的命令：

- `/reset` —— 清空对话状态与 KV cache
- `/exit` —— 退出程序

---

## 关于 `main.rs` 的重要说明

`src/main.rs` 使用了**硬编码的 `model_config()`** 和一个非常轻量的 CLI 流程。这是有意为之：它更容易读，也更适合作为示例。

但这也意味着：

- `model_config()` 中的模型结构必须与你加载的权重匹配；
- tokenizer 的词表与特殊 token 需要和 `vocab_size`、提示词模板兼容；
- 若要适配其他 checkpoint，通常需要调整维度、层数、注意力头配置、特殊 token，甚至 prompt template。

也就是说，**`main.rs` 是一个简单集成示例，不是通用启动器。**

---

## 这个项目的价值

很多 Rust 机器学习项目只做到张量层、MLP 层，或者停留在玩具示例；而 Lumen 把下面这些部分串在了一起：

- 张量与 autograd 基础能力
- 可复用神经网络模块
- Llama 解码器结构
- checkpoint 权重加载
- tokenizer 桥接
- 有状态自回归解码

因此，它不仅适合阅读，也适合作为一个小型 Rust 原生 DL / LLM runtime 的起点。

---

## License

GPL v3.0
