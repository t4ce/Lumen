use mimalloc::MiMalloc;

use lumen::autograd::{Tensor, no_grad};
use lumen::init::{ParameterInitMode, with_parameter_init_mode};
use lumen::loader::{ModelLoader, WeightLoadOptions};
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::ops::fp_kernels::active_float_backend_name;
use lumen::ops::int8_kernels::active_int8_backend_name;
use lumen::precision::{
    DType, ParameterQuantization, PrecisionConfig, with_parameter_quantization,
    with_precision_config, with_runtime_component_dtypes,
};
use lumen::tokenizer::LlamaTokenizer;

use ndarray::{Array, Array1, Ix3, s};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeMode {
    Greedy,
    Sample,
}

#[derive(Debug, Clone)]
struct Args {
    weights: String,
    tokenizer: String,
    prompt: String,
    system: String,
    runs: usize,
    warmup: usize,
    max_gen: usize,
    mode: DecodeMode,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    recent_window: usize,
    parameter_dtype: DType,
    runtime_dtype: DType,
    activation_dtype: DType,
    kv_cache_dtype: DType,
    allow_parameter_copies: bool,
    parameter_quantization: ParameterQuantization,
    stream_weights: bool,
    max_seq_len: usize,
    stop_on_eos: bool,
    decode_text_each_step: bool,
}

#[derive(Debug, Clone, Default)]
struct RunStats {
    prompt_tokens: usize,
    generated_tokens: usize,
    total: Duration,
    prefill_input_build: Duration,
    prefill_forward: Duration,
    prefill_logits_extract: Duration,
    decode_input_build: Duration,
    decode_forward: Duration,
    decode_logits_extract: Duration,
    sampling: Duration,
    tokenizer_decode: Duration,
}

impl RunStats {
    fn measured_total(&self) -> Duration {
        self.prefill_input_build
            + self.prefill_forward
            + self.prefill_logits_extract
            + self.decode_input_build
            + self.decode_forward
            + self.decode_logits_extract
            + self.sampling
            + self.tokenizer_decode
    }

    fn overhead(&self) -> Duration {
        self.total.saturating_sub(self.measured_total())
    }
}

fn usage(program: &str) {
    eprintln!(
        "Usage:\n  {program} --weights PATH --tokenizer PATH [options]\n\nOptions:\n  --prompt TEXT               User prompt text\n  --prompt-file PATH          Read user prompt text from file\n  --system TEXT               System prompt\n  --runs N                    Timed runs (default: 5)\n  --warmup N                  Warmup runs (default: 1)\n  --max-gen N                 Decode tokens per run (default: 128)\n  --mode MODE                 greedy/sample (default: sample)\n  --temperature FLOAT         Sampling temperature (default: 0.8)\n  --top-p FLOAT               Top-p nucleus sampling (default: 0.9)\n  --repetition-penalty FLOAT  Repetition penalty (default: 1.05)\n  --recent-window N           Recent token window (default: 96)\n  --stop-on-eos               Stop decode when EOS/stop token appears\n  --decode-text-each-step     Include tokenizer.decode(gen_tokens) timing each step\n  --parameter-dtype DTYPE     f32/f16/bf16/i8 (default: f32)\n  --runtime-dtype DTYPE       f32/f16/bf16 (default: f32)\n  --activation-dtype DTYPE    f32/f16/bf16/i8\n  --kv-cache-dtype DTYPE      f32/f16/bf16\n  --quantize DTYPE            off/i8 (default: off)\n  --quant-scale FLOAT         Manual quant scale override\n  --allow-parameter-copies    Allow cached parameter dtype copies\n  --stream-weights            Stream weights from disk\n  --max-seq-len N             Override KV cache max seq len (default: 2048)\n\nExamples:\n  cargo run --release --features \"dev-tools x86-fp-kernels x86-int8-kernels\" --bin prefill_decode_bench -- \\\n    --weights model.safetensors --tokenizer tokenizer.json --prompt \"你好，请解释 Transformer 的 KV cache\" \\\n    --parameter-dtype f16 --activation-dtype f16 --kv-cache-dtype f16 --allow-parameter-copies\n"
    );
}

fn model_config(max_seq_len: usize) -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        max_seq_len,
        rope_theta: 10000.0,
    }
}

fn parse_dtype_flag(flag: &str, value: &str, allow_integer: bool) -> Result<DType, String> {
    let dtype = match value.to_ascii_lowercase().as_str() {
        "f32" | "float32" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "i8" | "int8" => DType::I8,
        other => {
            return Err(format!(
                "{flag} 不支持的 dtype: {other}，可选值为 f32/f16/bf16{}",
                if allow_integer { "/i8" } else { "" }
            ));
        }
    };
    if !allow_integer && dtype.is_integer() {
        return Err(format!("{flag} 当前只支持浮点 dtype，不能使用 {:?}", dtype));
    }
    Ok(dtype)
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = env::args().collect();
    let program = argv.first().cloned().unwrap_or_else(|| "prefill_decode_bench".to_string());
    if argv.len() == 1 {
        usage(&program);
        return Err("缺少参数".to_string());
    }

    let mut weights: Option<String> = None;
    let mut tokenizer: Option<String> = None;
    let mut prompt = "请用简洁但准确的方式解释一下 Transformer 里的 KV cache，以及它为什么能加速 decode。".to_string();
    let mut prompt_file: Option<String> = None;
    let mut system = "You are a helpful AI assistant.".to_string();
    let mut runs = 5usize;
    let mut warmup = 1usize;
    let mut max_gen = 128usize;
    let mut mode = DecodeMode::Sample;
    let mut temperature = 0.8f32;
    let mut top_p = 0.9f32;
    let mut repetition_penalty = 1.05f32;
    let mut recent_window = 96usize;
    let mut parameter_dtype = DType::F32;
    let mut runtime_dtype = DType::F32;
    let mut activation_dtype: Option<DType> = None;
    let mut kv_cache_dtype: Option<DType> = None;
    let mut allow_parameter_copies = false;
    let mut quantize_dtype: Option<DType> = None;
    let mut quant_scale: Option<f32> = None;
    let mut stream_weights = false;
    let mut max_seq_len = 2048usize;
    let mut stop_on_eos = false;
    let mut decode_text_each_step = false;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                std::process::exit(0);
            }
            "--weights" => {
                i += 1;
                weights = Some(argv.get(i).ok_or("--weights 缺少路径")?.clone());
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = Some(argv.get(i).ok_or("--tokenizer 缺少路径")?.clone());
            }
            "--prompt" => {
                i += 1;
                prompt = argv.get(i).ok_or("--prompt 缺少文本")?.clone();
            }
            "--prompt-file" => {
                i += 1;
                prompt_file = Some(argv.get(i).ok_or("--prompt-file 缺少路径")?.clone());
            }
            "--system" => {
                i += 1;
                system = argv.get(i).ok_or("--system 缺少文本")?.clone();
            }
            "--runs" => {
                i += 1;
                runs = argv.get(i).ok_or("--runs 缺少数字")?.parse::<usize>().map_err(|_| "--runs 需要 usize")?;
                if runs == 0 { return Err("--runs 必须 >= 1".to_string()); }
            }
            "--warmup" => {
                i += 1;
                warmup = argv.get(i).ok_or("--warmup 缺少数字")?.parse::<usize>().map_err(|_| "--warmup 需要 usize")?;
            }
            "--max-gen" => {
                i += 1;
                max_gen = argv.get(i).ok_or("--max-gen 缺少数字")?.parse::<usize>().map_err(|_| "--max-gen 需要 usize")?;
            }
            "--mode" => {
                i += 1;
                mode = match argv.get(i).ok_or("--mode 缺少模式")?.to_ascii_lowercase().as_str() {
                    "greedy" => DecodeMode::Greedy,
                    "sample" | "sampling" => DecodeMode::Sample,
                    other => return Err(format!("--mode 不支持 {other}，可选 greedy/sample")),
                };
            }
            "--temperature" => {
                i += 1;
                temperature = argv.get(i).ok_or("--temperature 缺少数字")?.parse::<f32>().map_err(|_| "--temperature 需要 f32")?;
            }
            "--top-p" => {
                i += 1;
                top_p = argv.get(i).ok_or("--top-p 缺少数字")?.parse::<f32>().map_err(|_| "--top-p 需要 f32")?;
            }
            "--repetition-penalty" => {
                i += 1;
                repetition_penalty = argv.get(i).ok_or("--repetition-penalty 缺少数字")?.parse::<f32>().map_err(|_| "--repetition-penalty 需要 f32")?;
            }
            "--recent-window" => {
                i += 1;
                recent_window = argv.get(i).ok_or("--recent-window 缺少数字")?.parse::<usize>().map_err(|_| "--recent-window 需要 usize")?;
            }
            "--parameter-dtype" => {
                i += 1;
                parameter_dtype = parse_dtype_flag("--parameter-dtype", argv.get(i).ok_or("--parameter-dtype 缺少 dtype")?, true)?;
            }
            "--runtime-dtype" => {
                i += 1;
                runtime_dtype = parse_dtype_flag("--runtime-dtype", argv.get(i).ok_or("--runtime-dtype 缺少 dtype")?, false)?;
            }
            "--activation-dtype" => {
                i += 1;
                activation_dtype = Some(parse_dtype_flag("--activation-dtype", argv.get(i).ok_or("--activation-dtype 缺少 dtype")?, true)?);
            }
            "--kv-cache-dtype" => {
                i += 1;
                kv_cache_dtype = Some(parse_dtype_flag("--kv-cache-dtype", argv.get(i).ok_or("--kv-cache-dtype 缺少 dtype")?, false)?);
            }
            "--quantize" => {
                i += 1;
                let raw = argv.get(i).ok_or("--quantize 缺少 dtype，支持 off/i8")?.to_ascii_lowercase();
                quantize_dtype = match raw.as_str() {
                    "off" | "none" | "disabled" => None,
                    _ => Some(parse_dtype_flag("--quantize", &raw, true)?),
                };
            }
            "--quant-scale" => {
                i += 1;
                quant_scale = Some(argv.get(i).ok_or("--quant-scale 缺少数字")?.parse::<f32>().map_err(|_| "--quant-scale 需要 f32")?);
            }
            "--allow-parameter-copies" => allow_parameter_copies = true,
            "--stream-weights" => stream_weights = true,
            "--max-seq-len" => {
                i += 1;
                max_seq_len = argv.get(i).ok_or("--max-seq-len 缺少数字")?.parse::<usize>().map_err(|_| "--max-seq-len 需要 usize")?;
                if max_seq_len == 0 { return Err("--max-seq-len 必须 > 0".to_string()); }
            }
            "--stop-on-eos" => stop_on_eos = true,
            "--decode-text-each-step" => decode_text_each_step = true,
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }

    if let Some(path) = prompt_file {
        prompt = std::fs::read_to_string(&path).map_err(|e| format!("读取 --prompt-file 失败: {e}"))?;
    }
    if !(0.0..=1.0).contains(&top_p) {
        return Err("--top-p 必须在 [0, 1] 范围内".to_string());
    }
    if temperature < 0.0 {
        return Err("--temperature 不能小于 0".to_string());
    }
    if repetition_penalty < 1.0 {
        return Err("--repetition-penalty 不能小于 1.0".to_string());
    }
    if let Some(scale) = quant_scale {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("--quant-scale 必须是有限且 > 0 的 f32".to_string());
        }
    }
    if let Some(dtype) = quantize_dtype {
        if !dtype.is_integer() {
            return Err(format!("--quantize 当前只支持整数存储 dtype，收到 {:?}", dtype));
        }
        if parameter_dtype != DType::F32 && parameter_dtype != dtype {
            return Err(format!(
                "--parameter-dtype={:?} 与 --quantize={:?} 冲突；量化开启时请设为默认 f32 或与量化 dtype 一致",
                parameter_dtype, dtype
            ));
        }
    } else if quant_scale.is_some() {
        return Err("--quant-scale 只有在 --quantize 开启时才可使用".to_string());
    }
    let parameter_quantization = match quantize_dtype {
        Some(dtype) => {
            let quant = ParameterQuantization::new(dtype);
            match quant_scale {
                Some(scale) => quant.with_scale(scale),
                None => quant,
            }
        }
        None => ParameterQuantization::Disabled,
    };

    let activation_dtype = activation_dtype.unwrap_or(runtime_dtype);
    let kv_cache_dtype = kv_cache_dtype.unwrap_or(runtime_dtype);
    Ok(Args {
        weights: weights.ok_or("必须提供 --weights")?,
        tokenizer: tokenizer.ok_or("必须提供 --tokenizer")?,
        prompt,
        system,
        runs,
        warmup,
        max_gen,
        mode,
        temperature,
        top_p,
        repetition_penalty,
        recent_window,
        parameter_dtype,
        runtime_dtype,
        activation_dtype,
        kv_cache_dtype,
        allow_parameter_copies,
        parameter_quantization,
        stream_weights,
        max_seq_len,
        stop_on_eos,
        decode_text_each_step,
    })
}

fn build_first_turn_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n",
        system, user
    )
}

#[inline]
fn rand01() -> f32 {
    Array1::<f32>::random(1, Uniform::new(0.0f32, 1.0f32))[0]
}

fn sample_top_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    recent_tokens: &[usize],
) -> usize {
    let mut adjusted = logits
        .iter()
        .map(|&v| if v.is_finite() { v } else { f32::NEG_INFINITY })
        .collect::<Vec<_>>();

    if adjusted.is_empty() {
        return 0;
    }

    if repetition_penalty > 1.0 {
        for &t in recent_tokens {
            if t < adjusted.len() {
                adjusted[t] /= repetition_penalty;
            }
        }
    }

    if temperature <= 1e-5 {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in adjusted.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        return best_i;
    }

    for v in adjusted.iter_mut() {
        *v /= temperature;
    }

    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in adjusted.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    if !best_v.is_finite() {
        return best_i;
    }

    let maxv = best_v;
    let mut probs: Vec<f32> = adjusted
        .iter()
        .map(|&x| if x.is_finite() { (x - maxv).exp() } else { 0.0 })
        .collect();
    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return best_i;
    }
    let inv = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv;
    }

    let mut idxs: Vec<usize> = (0..probs.len()).collect();
    idxs.sort_by(|&i, &j| probs[j].total_cmp(&probs[i]));

    let mut cumulative = 0.0f32;
    let mut cut = 0usize;
    let target_p = top_p.clamp(0.0, 1.0).max(1e-6);
    for (rank, &i) in idxs.iter().enumerate() {
        cumulative += probs[i];
        cut = rank + 1;
        if cumulative >= target_p {
            break;
        }
    }
    idxs.truncate(cut.max(1));

    let r = rand01();
    let mut acc = 0.0f32;
    for &i in &idxs {
        acc += probs[i] / cumulative;
        if r <= acc {
            return i;
        }
    }
    *idxs.last().unwrap()
}

fn tensor_from_token_ids(ids: &[usize]) -> Tensor {
    Tensor::from_array_no_grad(
        Array::from_shape_vec((1, ids.len()), ids.iter().map(|&x| x as f32).collect())
            .unwrap()
            .into_dyn(),
    )
}

fn last_step_logits_vec(logits: &Tensor) -> Vec<f32> {
    let logits_ref = logits.data_ref();
    let l3 = logits_ref
        .view()
        .into_dimensionality::<Ix3>()
        .expect("logits must be 3D [B,S,V]");
    let t = l3.shape()[1] - 1;
    l3.slice(s![0, t, ..]).iter().copied().collect()
}

fn build_stop_ids(tokenizer: &LlamaTokenizer) -> Vec<usize> {
    let mut stop_ids = Vec::new();
    for t in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"] {
        if let Some(id) = tokenizer.token_to_id(t) {
            stop_ids.push(id);
        }
    }
    if let Some(id) = tokenizer.eos_id() {
        stop_ids.push(id);
    }
    if let Some(id) = tokenizer.eot_id() {
        stop_ids.push(id);
    }
    stop_ids.sort_unstable();
    stop_ids.dedup();
    stop_ids
}

fn load_model(args: &Args, config: &LlamaConfig) -> Result<LlamaModel, Box<dyn std::error::Error>> {
    let precision_config = PrecisionConfig {
        parameter_dtype: args.parameter_dtype,
        runtime_dtype: args.runtime_dtype,
        allow_parameter_dtype_copies: args.allow_parameter_copies,
    };
    let load_options = WeightLoadOptions {
        float_source_quantization: args.parameter_quantization,
        stream_from_disk: args.stream_weights,
    };
    with_precision_config(precision_config, || {
        with_runtime_component_dtypes(Some(args.activation_dtype), Some(args.kv_cache_dtype), || {
            with_parameter_quantization(args.parameter_quantization, || {
                let model = with_parameter_init_mode(ParameterInitMode::Placeholder, || {
                    LlamaModel::new(config.clone())
                });
                ModelLoader::load_llama_weights_with_options(
                    &args.weights,
                    &model.named_parameters(),
                    load_options,
                )?;
                Ok::<LlamaModel, Box<dyn std::error::Error>>(model)
            })
        })
    })
}

fn run_once(
    model: &LlamaModel,
    tokenizer: &LlamaTokenizer,
    config: &LlamaConfig,
    stop_ids: &[usize],
    prompt_tokens: &[usize],
    args: &Args,
) -> RunStats {
    let mut stats = RunStats {
        prompt_tokens: prompt_tokens.len(),
        ..Default::default()
    };
    let mut kv_caches = model.init_kv_caches(1);
    model.reset_kv_caches(&mut kv_caches);
    let mut all_tokens = prompt_tokens.to_vec();
    let assistant_start = all_tokens.len();

    no_grad(|| {
        let total_start = Instant::now();

        let build_start = Instant::now();
        let prefill_input = tensor_from_token_ids(prompt_tokens);
        stats.prefill_input_build = build_start.elapsed();

        let forward_start = Instant::now();
        let prefill_logits = model.forward_last_logits(prefill_input, &mut kv_caches, 0);
        stats.prefill_forward = forward_start.elapsed();

        let logits_extract_start = Instant::now();
        let mut logits_vec = last_step_logits_vec(&prefill_logits);
        stats.prefill_logits_extract = logits_extract_start.elapsed();

        for _ in 0..args.max_gen {
            let next_token = match args.mode {
                DecodeMode::Greedy => {
                    let mut best_i = 0usize;
                    let mut best_v = f32::NEG_INFINITY;
                    for (i, &v) in logits_vec.iter().enumerate() {
                        if v > best_v {
                            best_v = v;
                            best_i = i;
                        }
                    }
                    best_i
                }
                DecodeMode::Sample => {
                    let start = all_tokens.len().saturating_sub(args.recent_window);
                    let recent = &all_tokens[start..];
                    let sample_start = Instant::now();
                    let sampled = sample_top_p(
                        &logits_vec,
                        args.temperature,
                        args.top_p,
                        args.repetition_penalty,
                        recent,
                    );
                    stats.sampling += sample_start.elapsed();
                    sampled
                }
            };

            if args.stop_on_eos && stop_ids.contains(&next_token) {
                break;
            }
            all_tokens.push(next_token);
            stats.generated_tokens += 1;

            if args.decode_text_each_step {
                let decode_start = Instant::now();
                let _ = tokenizer.decode(&all_tokens[assistant_start..], true);
                stats.tokenizer_decode += decode_start.elapsed();
            }

            let build_start = Instant::now();
            let decode_input = tensor_from_token_ids(&[next_token]);
            stats.decode_input_build += build_start.elapsed();

            match args.mode {
                DecodeMode::Greedy => {
                    let forward_start = Instant::now();
                    let next = model.forward_last_argmax(decode_input, &mut kv_caches, 0);
                    stats.decode_forward += forward_start.elapsed();
                    logits_vec.fill(f32::NEG_INFINITY);
                    if next < logits_vec.len() {
                        logits_vec[next] = 0.0;
                    }
                }
                DecodeMode::Sample => {
                    let forward_start = Instant::now();
                    let logits = model.forward_last_logits(decode_input, &mut kv_caches, 0);
                    stats.decode_forward += forward_start.elapsed();
                    let extract_start = Instant::now();
                    logits_vec = last_step_logits_vec(&logits);
                    stats.decode_logits_extract += extract_start.elapsed();
                }
            }

            if kv_caches[0].borrow().len + 2 >= config.max_seq_len {
                break;
            }
        }

        stats.total = total_start.elapsed();
    });

    stats
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_us_per_token(field: impl Fn(&RunStats) -> Duration, runs: &[RunStats]) -> f64 {
    let mut vals = Vec::new();
    for run in runs {
        let denom = run.generated_tokens.max(1) as f64;
        vals.push(field(run).as_secs_f64() * 1e6 / denom);
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[vals.len() / 2]
}

fn median_us_per_prompt_token(field: impl Fn(&RunStats) -> Duration, runs: &[RunStats]) -> f64 {
    let mut vals = Vec::new();
    for run in runs {
        let denom = run.prompt_tokens.max(1) as f64;
        vals.push(field(run).as_secs_f64() * 1e6 / denom);
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[vals.len() / 2]
}

fn median_scalar(field: impl Fn(&RunStats) -> usize, runs: &[RunStats]) -> usize {
    let mut vals = runs.iter().map(field).collect::<Vec<_>>();
    vals.sort_unstable();
    vals[vals.len() / 2]
}

fn print_stage(label: &str, total_us: f64, per_unit_us: f64, unit: &str, share: f64) {
    println!(
        "{label:<20} total={total_us:>10.2} us  per_{unit}={per_unit_us:>10.2} us  share={share:>6.2}%",
        share = share * 100.0
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("参数错误: {e}"))?;
    let config = model_config(args.max_seq_len);

    if !Path::new(&args.tokenizer).exists() {
        return Err(format!("tokenizer 文件不存在: {}", args.tokenizer).into());
    }
    if !Path::new(&args.weights).exists() {
        return Err(format!("weights 文件不存在: {}", args.weights).into());
    }

    let prompt = build_first_turn_prompt(&args.system, &args.prompt);
    let tokenizer = LlamaTokenizer::from_file(&args.tokenizer)?;
    let prompt_tokens = tokenizer.encode(&prompt, false)?;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenization 结果为空".into());
    }
    if prompt_tokens.len() + args.max_gen + 8 >= config.max_seq_len {
        return Err(format!(
            "prompt_tokens={} + max_gen={} 超过 max_seq_len={}，请调大 --max-seq-len 或减小 prompt/max-gen",
            prompt_tokens.len(),
            args.max_gen,
            config.max_seq_len
        )
        .into());
    }

    println!(
        "prefill+decode bench: runs={} warmup={} prompt_tokens={} max_gen={} mode={:?}",
        args.runs,
        args.warmup,
        prompt_tokens.len(),
        args.max_gen,
        args.mode
    );
    println!(
        "backend: float={} int8={}",
        active_float_backend_name(),
        active_int8_backend_name()
    );
    println!(
        "dtype: parameter={:?} runtime={:?} activation={:?} kv_cache={:?} quantization={:?} allow_parameter_copies={} stream_weights={} decode_text_each_step={} stop_on_eos={}",
        args.parameter_dtype,
        args.runtime_dtype,
        args.activation_dtype,
        args.kv_cache_dtype,
        args.parameter_quantization,
        args.allow_parameter_copies,
        args.stream_weights,
        args.decode_text_each_step,
        args.stop_on_eos
    );

    let model = load_model(&args, &config)?;
    let stop_ids = build_stop_ids(&tokenizer);

    for _ in 0..args.warmup {
        let _ = run_once(&model, &tokenizer, &config, &stop_ids, &prompt_tokens, &args);
    }

    let mut runs = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        runs.push(run_once(&model, &tokenizer, &config, &stop_ids, &prompt_tokens, &args));
    }

    let prompt_tok_med = median_scalar(|r| r.prompt_tokens, &runs);
    let gen_tok_med = median_scalar(|r| r.generated_tokens, &runs);
    let total_med = median_duration(runs.iter().map(|r| r.total).collect());
    let measured_med = median_duration(runs.iter().map(|r| r.measured_total()).collect());
    let overhead_med = median_duration(runs.iter().map(|r| r.overhead()).collect());

    let prefill_tok_s = prompt_tok_med as f64 / median_duration(runs.iter().map(|r| r.prefill_forward).collect()).as_secs_f64();
    let decode_tok_s = gen_tok_med.max(1) as f64 / median_duration(runs.iter().map(|r| r.decode_forward).collect()).as_secs_f64();
    let end_to_end_tok_s = gen_tok_med.max(1) as f64 / total_med.as_secs_f64();

    println!();
    println!(
        "summary: prompt_tokens={} generated_tokens={} total={:.2} ms measured={:.2} ms overhead={:.2} ms",
        prompt_tok_med,
        gen_tok_med,
        total_med.as_secs_f64() * 1e3,
        measured_med.as_secs_f64() * 1e3,
        overhead_med.as_secs_f64() * 1e3,
    );
    println!(
        "throughput: prefill_forward={:.2} tok/s  decode_forward={:.2} tok/s  end_to_end_decode={:.2} tok/s",
        prefill_tok_s,
        decode_tok_s,
        end_to_end_tok_s,
    );
    println!();

    let total_secs = total_med.as_secs_f64().max(1e-12);
    print_stage(
        "prefill_input_build",
        median_duration(runs.iter().map(|r| r.prefill_input_build).collect()).as_secs_f64() * 1e6,
        median_us_per_prompt_token(|r| r.prefill_input_build, &runs),
        "prompt_tok",
        median_duration(runs.iter().map(|r| r.prefill_input_build).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "prefill_forward",
        median_duration(runs.iter().map(|r| r.prefill_forward).collect()).as_secs_f64() * 1e6,
        median_us_per_prompt_token(|r| r.prefill_forward, &runs),
        "prompt_tok",
        median_duration(runs.iter().map(|r| r.prefill_forward).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "prefill_logits_extract",
        median_duration(runs.iter().map(|r| r.prefill_logits_extract).collect()).as_secs_f64() * 1e6,
        median_us_per_prompt_token(|r| r.prefill_logits_extract, &runs),
        "prompt_tok",
        median_duration(runs.iter().map(|r| r.prefill_logits_extract).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "decode_input_build",
        median_duration(runs.iter().map(|r| r.decode_input_build).collect()).as_secs_f64() * 1e6,
        median_us_per_token(|r| r.decode_input_build, &runs),
        "gen_tok",
        median_duration(runs.iter().map(|r| r.decode_input_build).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "decode_forward",
        median_duration(runs.iter().map(|r| r.decode_forward).collect()).as_secs_f64() * 1e6,
        median_us_per_token(|r| r.decode_forward, &runs),
        "gen_tok",
        median_duration(runs.iter().map(|r| r.decode_forward).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "decode_logits_extract",
        median_duration(runs.iter().map(|r| r.decode_logits_extract).collect()).as_secs_f64() * 1e6,
        median_us_per_token(|r| r.decode_logits_extract, &runs),
        "gen_tok",
        median_duration(runs.iter().map(|r| r.decode_logits_extract).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "sampling",
        median_duration(runs.iter().map(|r| r.sampling).collect()).as_secs_f64() * 1e6,
        median_us_per_token(|r| r.sampling, &runs),
        "gen_tok",
        median_duration(runs.iter().map(|r| r.sampling).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "tokenizer_decode",
        median_duration(runs.iter().map(|r| r.tokenizer_decode).collect()).as_secs_f64() * 1e6,
        median_us_per_token(|r| r.tokenizer_decode, &runs),
        "gen_tok",
        median_duration(runs.iter().map(|r| r.tokenizer_decode).collect()).as_secs_f64() / total_secs,
    );
    print_stage(
        "unattributed_overhead",
        overhead_med.as_secs_f64() * 1e6,
        overhead_med.as_secs_f64() * 1e6 / gen_tok_med.max(1) as f64,
        "gen_tok",
        overhead_med.as_secs_f64() / total_secs,
    );

    Ok(())
}
