use crate::autograd::{
    StoragePreference, Tensor, TensorStorageView, TensorStorageViewMut, is_no_grad,
};
use crate::layers::Linear;
use crate::layers::attention::encoding::RotaryEmbedding;
use crate::module::Module;
use crate::ops::fused::{fused_qkv_decode_infer_into, fused_softmax};
use crate::ops::matmul::{batch_matmul, dot_unrolled};
use crate::ops::shape::{permute, reshape};
use crate::precision::{DType, default_runtime_dtype};

use ndarray::Array4;
use ndarray::linalg::general_mat_mul;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    // attention scores buffer: S * L
    static ATT_SCORES_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // attention ctx buffer: S * D
    static ATT_CTX_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // decode(S=1) q RoPE buffer: D
    static ATT_Q_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // decode(S=1) full attention output buffer: H * D
    static ATT_OUT_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // decode(S=1) fused projection scratch buffers
    static ATT_QPROJ_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static ATT_KPROJ_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static ATT_VPROJ_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn with_attention_work_buffers<R>(
    scores_len: usize,
    ctx_len: usize,
    f: impl FnOnce(&mut [f32], &mut [f32]) -> R,
) -> R {
    ATT_SCORES_BUF.with(|sb| {
        ATT_CTX_BUF.with(|cb| match (sb.try_borrow_mut(), cb.try_borrow_mut()) {
            (Ok(mut scores_buf), Ok(mut ctx_buf)) => {
                if scores_buf.len() < scores_len {
                    scores_buf.resize(scores_len, 0.0);
                }
                if ctx_buf.len() < ctx_len {
                    ctx_buf.resize(ctx_len, 0.0);
                }
                f(&mut scores_buf[..scores_len], &mut ctx_buf[..ctx_len])
            }
            _ => {
                let mut scores_buf = vec![0.0f32; scores_len];
                let mut ctx_buf = vec![0.0f32; ctx_len];
                f(&mut scores_buf, &mut ctx_buf)
            }
        })
    })
}

fn write_cache_token_row(
    dst: TensorStorageViewMut<'_>,
    bb: usize,
    hk: usize,
    pos: usize,
    src: &[f32],
) {
    match dst {
        TensorStorageViewMut::F32(view) => {
            let mut view4 = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]");
            let mut row = view4.slice_mut(ndarray::s![bb, hk, pos, ..]);
            row.as_slice_mut()
                .expect("KV cache row must be contiguous")
                .copy_from_slice(src);
        }
        TensorStorageViewMut::F16(view) => {
            let mut view4 = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]");
            let mut row = view4.slice_mut(ndarray::s![bb, hk, pos, ..]);
            let row_slice = row.as_slice_mut().expect("KV cache row must be contiguous");
            for (dst, &value) in row_slice.iter_mut().zip(src.iter()) {
                *dst = half::f16::from_f32(value);
            }
        }
        TensorStorageViewMut::BF16(view) => {
            let mut view4 = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]");
            let mut row = view4.slice_mut(ndarray::s![bb, hk, pos, ..]);
            let row_slice = row.as_slice_mut().expect("KV cache row must be contiguous");
            for (dst, &value) in row_slice.iter_mut().zip(src.iter()) {
                *dst = half::bf16::from_f32(value);
            }
        }
        TensorStorageViewMut::I8(_, _) => {
            panic!("i8 KV cache writes are not supported yet");
        }
    }
}

fn write_cache_range(
    dst: TensorStorageViewMut<'_>,
    start: usize,
    end: usize,
    src: &ndarray::ArrayView4<'_, f32>,
) {
    match dst {
        TensorStorageViewMut::F32(view) => {
            view.into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]")
                .slice_mut(ndarray::s![.., .., start..end, ..])
                .assign(src);
        }
        TensorStorageViewMut::F16(view) => {
            let mut dst4 = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]");
            for bb in 0..src.dim().0 {
                for hk in 0..src.dim().1 {
                    for ss in 0..src.dim().2 {
                        let src_row = src.slice(ndarray::s![bb, hk, ss, ..]);
                        let mut dst_row = dst4.slice_mut(ndarray::s![bb, hk, start + ss, ..]);
                        let dst_slice = dst_row
                            .as_slice_mut()
                            .expect("KV cache row must be contiguous");
                        for (dst, &value) in dst_slice.iter_mut().zip(src_row.iter()) {
                            *dst = half::f16::from_f32(value);
                        }
                    }
                }
            }
        }
        TensorStorageViewMut::BF16(view) => {
            let mut dst4 = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]");
            for bb in 0..src.dim().0 {
                for hk in 0..src.dim().1 {
                    for ss in 0..src.dim().2 {
                        let src_row = src.slice(ndarray::s![bb, hk, ss, ..]);
                        let mut dst_row = dst4.slice_mut(ndarray::s![bb, hk, start + ss, ..]);
                        let dst_slice = dst_row
                            .as_slice_mut()
                            .expect("KV cache row must be contiguous");
                        for (dst, &value) in dst_slice.iter_mut().zip(src_row.iter()) {
                            *dst = half::bf16::from_f32(value);
                        }
                    }
                }
            }
        }
        TensorStorageViewMut::I8(_, _) => {
            panic!("i8 KV cache writes are not supported yet");
        }
    }
}

fn with_cache_f32_views<R>(
    k: &Tensor,
    v: &Tensor,
    f: impl FnOnce(ndarray::ArrayView4<'_, f32>, ndarray::ArrayView4<'_, f32>) -> R,
) -> R {
    k.with_storage_view_preferring(StoragePreference::F32Compute, |k_view| {
        v.with_storage_view_preferring(StoragePreference::F32Compute, |v_view| {
            let k4 = match k_view {
                TensorStorageView::F32(view) => view.into_dimensionality::<ndarray::Ix4>().unwrap(),
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            };
            let v4 = match v_view {
                TensorStorageView::F32(view) => view.into_dimensionality::<ndarray::Ix4>().unwrap(),
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            };
            f(k4, v4)
        })
    })
}

pub struct KVCacheInner {
    pub k: Tensor,  // [B, H_kv, max_seq, D]
    pub v: Tensor,  // [B, H_kv, max_seq, D]
    pub len: usize, // 当前已写入的长度
    pub dtype: DType,
    pub follows_global_dtype: bool,
}

pub type KVCache = Rc<RefCell<KVCacheInner>>;

impl KVCacheInner {
    pub fn new(b: usize, h_kv: usize, max_seq: usize, d: usize) -> Self {
        Self::new_impl(b, h_kv, max_seq, d, default_runtime_dtype(), true)
    }

    fn new_impl(
        b: usize,
        h_kv: usize,
        max_seq: usize,
        d: usize,
        dtype: DType,
        follows_global_dtype: bool,
    ) -> Self {
        assert!(b > 0, "KV cache batch size must be > 0");
        assert!(h_kv > 0, "KV cache head count must be > 0");
        assert!(max_seq > 0, "KV cache max_seq must be > 0");
        assert!(d > 0, "KV cache head dim must be > 0");
        assert!(
            dtype.is_float(),
            "KV cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        let k = Tensor::from_array_no_grad(Array4::<f32>::zeros((b, h_kv, max_seq, d)).into_dyn());
        let v = Tensor::from_array_no_grad(Array4::<f32>::zeros((b, h_kv, max_seq, d)).into_dyn());
        k.cast_inplace(dtype);
        v.cast_inplace(dtype);
        Self {
            k,
            v,
            len: 0,
            dtype,
            follows_global_dtype,
        }
    }

    pub fn new_with_dtype(b: usize, h_kv: usize, max_seq: usize, d: usize, dtype: DType) -> Self {
        Self::new_impl(b, h_kv, max_seq, d, dtype, false)
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn cast_inplace(&mut self, dtype: DType) {
        assert!(
            dtype.is_float(),
            "KV cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        self.k.cast_inplace(dtype);
        self.v.cast_inplace(dtype);
        self.dtype = dtype;
        self.follows_global_dtype = false;
    }

    fn validate_layout(
        &self,
        expected_b: usize,
        expected_h_kv: usize,
        expected_max_seq: usize,
        expected_d: usize,
        context: &str,
    ) {
        let expected_shape = vec![expected_b, expected_h_kv, expected_max_seq, expected_d];
        let k_shape = self.k.shape_vec();
        let v_shape = self.v.shape_vec();
        assert_eq!(
            k_shape.len(),
            4,
            "{} must store K as [B,H,S,D], got {:?}",
            context,
            k_shape
        );
        assert_eq!(
            v_shape.len(),
            4,
            "{} must store V as [B,H,S,D], got {:?}",
            context,
            v_shape
        );
        assert_eq!(
            k_shape, expected_shape,
            "{} shape mismatch: expected {:?}, got {:?}",
            context, expected_shape, k_shape
        );
        assert_eq!(
            v_shape, expected_shape,
            "{} value shape mismatch: expected {:?}, got {:?}",
            context, expected_shape, v_shape
        );
        assert!(
            self.dtype.is_float(),
            "{} currently only supports floating dtypes, got {:?}",
            context,
            self.dtype
        );
        assert_eq!(
            self.k.dtype(),
            self.dtype,
            "{} metadata/tensor dtype mismatch for K: {:?} vs {:?}",
            context,
            self.dtype,
            self.k.dtype()
        );
        assert_eq!(
            self.v.dtype(),
            self.dtype,
            "{} metadata/tensor dtype mismatch for V: {:?} vs {:?}",
            context,
            self.dtype,
            self.v.dtype()
        );
        assert_eq!(
            self.k.dtype(),
            self.v.dtype(),
            "{} K/V dtype mismatch: {:?} vs {:?}",
            context,
            self.k.dtype(),
            self.v.dtype()
        );
        assert!(
            self.len <= expected_max_seq,
            "{} current length out of bounds: {} > {}",
            context,
            self.len,
            expected_max_seq
        );
    }
}

pub struct SelfAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    rope: RotaryEmbedding,
    n_head: usize,
    pub n_kv_head: usize,
    head_dim: usize,
    scale: f32,
    pub causal: bool,
    max_seq: usize, // 为 cache 预分配用
    runtime_dtype: DType,
}

impl SelfAttention {
    fn new_impl(
        embed_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_seq_len: usize,
        rope_theta: f32,
        causal: bool,
        parameter_dtype: DType,
        runtime_dtype: DType,
    ) -> Self {
        assert!(embed_dim > 0, "embed_dim must be > 0");
        assert!(n_head > 0, "n_head must be > 0");
        assert!(n_kv_head > 0, "n_kv_head must be > 0");
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        assert_eq!(
            embed_dim % n_head,
            0,
            "Embed dim must be divisible by n_head"
        );
        assert_eq!(
            n_head % n_kv_head,
            0,
            "n_head must be divisible by n_kv_head"
        );
        assert!(
            runtime_dtype.is_float(),
            "SelfAttention runtime dtype currently only supports floating types, got {:?}",
            runtime_dtype
        );

        let head_dim = embed_dim / n_head;
        let kv_dim = n_kv_head * head_dim;

        let rope =
            RotaryEmbedding::new_with_dtype(head_dim, max_seq_len, rope_theta, runtime_dtype);

        Self {
            w_q: Linear::new_no_bias_with_dtype(embed_dim, embed_dim, parameter_dtype),
            w_k: Linear::new_no_bias_with_dtype(embed_dim, kv_dim, parameter_dtype),
            w_v: Linear::new_no_bias_with_dtype(embed_dim, kv_dim, parameter_dtype),
            w_o: Linear::new_no_bias_with_dtype(embed_dim, embed_dim, parameter_dtype),
            rope,
            n_head,
            n_kv_head,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            causal,
            max_seq: max_seq_len, // 存储正确的最大长度
            runtime_dtype,
        }
    }

    pub fn new_with_dtypes(
        embed_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_seq_len: usize,
        rope_theta: f32,
        causal: bool,
        parameter_dtype: DType,
        runtime_dtype: DType,
    ) -> Self {
        Self::new_impl(
            embed_dim,
            n_head,
            n_kv_head,
            max_seq_len,
            rope_theta,
            causal,
            parameter_dtype,
            runtime_dtype,
        )
    }

    pub fn new(
        embed_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_seq_len: usize,
        rope_theta: f32,
        causal: bool,
    ) -> Self {
        assert!(embed_dim > 0, "embed_dim must be > 0");
        assert!(n_head > 0, "n_head must be > 0");
        assert!(n_kv_head > 0, "n_kv_head must be > 0");
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        assert_eq!(
            embed_dim % n_head,
            0,
            "Embed dim must be divisible by n_head"
        );
        assert_eq!(
            n_head % n_kv_head,
            0,
            "n_head must be divisible by n_kv_head"
        );

        let runtime_dtype = default_runtime_dtype();
        assert!(
            runtime_dtype.is_float(),
            "SelfAttention runtime dtype currently only supports floating types, got {:?}",
            runtime_dtype
        );

        let head_dim = embed_dim / n_head;
        let kv_dim = n_kv_head * head_dim;
        let rope =
            RotaryEmbedding::new_with_dtype(head_dim, max_seq_len, rope_theta, runtime_dtype);

        Self {
            w_q: Linear::new_no_bias(embed_dim, embed_dim),
            w_k: Linear::new_no_bias(embed_dim, kv_dim),
            w_v: Linear::new_no_bias(embed_dim, kv_dim),
            w_o: Linear::new_no_bias(embed_dim, embed_dim),
            rope,
            n_head,
            n_kv_head,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            causal,
            max_seq: max_seq_len,
            runtime_dtype,
        }
    }

    pub fn new_with_dtype(
        embed_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_seq_len: usize,
        rope_theta: f32,
        causal: bool,
        dtype: DType,
    ) -> Self {
        Self::new_impl(
            embed_dim,
            n_head,
            n_kv_head,
            max_seq_len,
            rope_theta,
            causal,
            dtype,
            dtype,
        )
    }

    pub(crate) fn assert_cache_compatible(
        &self,
        cache: &KVCache,
        batch_size: usize,
        context: &str,
    ) {
        cache.borrow().validate_layout(
            batch_size,
            self.n_kv_head,
            self.max_seq,
            self.head_dim,
            context,
        );
    }

    // forward：eval 用预分配 cache；train 走原逻辑（cat + repeat_kv）
    pub fn forward(&self, x: Tensor, cache: Option<KVCache>) -> (Tensor, Option<KVCache>) {
        let x_shape = x.shape_vec();
        assert_eq!(x_shape.len(), 3, "attention input must be [B,S,H]");
        let (b, s, _) = (x_shape[0], x_shape[1], x_shape[2]);
        assert!(b > 0, "attention input batch size must be > 0");
        assert!(s > 0, "attention input sequence length must be > 0");
        assert!(self.n_kv_head > 0, "n_kv_head must be > 0");

        let h = self.n_head;
        let h_kv = self.n_kv_head;
        let d = self.head_dim;
        assert_eq!(h % h_kv, 0, "n_head must be divisible by n_kv_head");
        assert_eq!(
            x_shape[2], self.w_q.in_features,
            "attention input hidden size mismatch"
        );
        let n_rep = h / h_kv;

        // eval 路径：尽量绕开 Tensor shape ops（不产生中间 Tensor，也不触发 copy）
        if is_no_grad() {
            // ------------- decode(S=1) ultra hot-path -------------
            // Goal:
            // - Only rotate NEW token's Q/K (with offset=past_len)
            // - Write rotated K directly into KV cache (no k_rot tensor)
            // - Fuse RoPE(Q) + online-softmax attention (no scores buffer)
            // - Output BSHD layout to avoid permute copies
            let cache_handle: KVCache = match cache {
                Some(c) => c,
                None => Rc::new(RefCell::new(KVCacheInner::new_with_dtype(
                    b,
                    h_kv,
                    self.max_seq,
                    d,
                    self.runtime_dtype,
                ))),
            };
            self.assert_cache_compatible(&cache_handle, b, "SelfAttention KV cache");

            let past_len = cache_handle.borrow().len;
            if s == 1 {
                let q_proj_dim = h * d;
                let kv_proj_dim = h_kv * d;

                return ATT_QPROJ_BUF.with(|qpb| {
                    ATT_KPROJ_BUF.with(|kpb| {
                        ATT_VPROJ_BUF.with(|vpb| {
                            let mut qpb = qpb.borrow_mut();
                            let mut kpb = kpb.borrow_mut();
                            let mut vpb = vpb.borrow_mut();
                            if qpb.len() < q_proj_dim {
                                qpb.resize(q_proj_dim, 0.0);
                            }
                            if kpb.len() < kv_proj_dim {
                                kpb.resize(kv_proj_dim, 0.0);
                            }
                            if vpb.len() < kv_proj_dim {
                                vpb.resize(kv_proj_dim, 0.0);
                            }
                            {
                                let q_out = &mut qpb[..q_proj_dim];
                                let k_out = &mut kpb[..kv_proj_dim];
                                let v_out = &mut vpb[..kv_proj_dim];
                                fused_qkv_decode_infer_into(
                                    &x,
                                    &self.w_q.weight,
                                    &self.w_k.weight,
                                    &self.w_v.weight,
                                    q_out,
                                    k_out,
                                    v_out,
                                );
                            }
                            let q_all: &[f32] = &qpb[..q_proj_dim];
                            let k_new: &[f32] = &kpb[..kv_proj_dim];
                            let v_new: &[f32] = &vpb[..kv_proj_dim];

                            // 1) Write NEW token into KV cache. Rotate K on-the-fly into destination.
                            {
                                let mut c = cache_handle.borrow_mut();
                                let new_len = past_len + 1;
                                assert!(
                                    new_len <= self.max_seq,
                                    "KV cache overflow: new_len={} > max_seq={}",
                                    new_len,
                                    self.max_seq
                                );

                                let KVCacheInner { k, v, len, .. } = &mut *c;
                                for bb in 0..b {
                                    for hk in 0..h_kv {
                                        let src_off = bb * kv_proj_dim + hk * d;
                                        let src_k = &k_new[src_off..src_off + d];
                                        let src_v = &v_new[src_off..src_off + d];
                                        let mut rotated_k = vec![0.0f32; d];
                                        self.rope.rope_1token_copy(src_k, &mut rotated_k, past_len);
                                        k.with_native_storage_view_mut(|k_view| {
                                            write_cache_token_row(
                                                k_view, bb, hk, past_len, &rotated_k,
                                            );
                                        });
                                        v.with_native_storage_view_mut(|v_view| {
                                            write_cache_token_row(v_view, bb, hk, past_len, src_v);
                                        });
                                    }
                                }
                                *len = new_len;
                            }

                            // 2) Fused attention: RoPE(Q) on-the-fly + online softmax + weighted sum.
                            let total_len = past_len + 1;
                            let output = ATT_OUT_BUF.with(|ob| {
                                let c = cache_handle.borrow();
                                with_cache_f32_views(&c.k, &c.v, |k4, v4| {
                                    let mut ob = ob.borrow_mut();
                                    let out_len = b * h * d;
                                    if ob.len() < out_len {
                                        ob.resize(out_len, 0.0);
                                    }
                                    let out_vec = &mut ob[..out_len]; // [B,H,D] for S=1

                                    let scale = self.scale;
                                    let causal = self.causal;
                                    let q_batch_stride = h * d;
                                    let (cos_row_vec, sin_row_vec) =
                                        self.rope.cos_sin_row_vec(past_len);
                                    let half_d = d / 2;

                                    out_vec.par_chunks_mut(d).enumerate().for_each(
                                        |(row, out_row)| {
                                            let bb = row / h;
                                            let hh = row % h;
                                            let hk = hh / n_rep;

                                            ATT_Q_BUF.with(|qb| {
                                                let mut qb = qb.borrow_mut();
                                                if qb.len() < d {
                                                    qb.resize(d, 0.0);
                                                }
                                                let qbuf = &mut qb[..d];

                                                let q_off = bb * q_batch_stride + hh * d;
                                                let q_src = &q_all[q_off..q_off + d];
                                                for j in 0..half_d {
                                                    let x1 = q_src[j];
                                                    let x2 = q_src[j + half_d];
                                                    let c = cos_row_vec[j];
                                                    let s_val = sin_row_vec[j];
                                                    qbuf[j] = x1 * c - x2 * s_val;
                                                    qbuf[j + half_d] = x1 * s_val + x2 * c;
                                                }

                                                ATT_CTX_BUF.with(|cb| {
                                                    let mut cb = cb.borrow_mut();
                                                    if cb.len() < d {
                                                        cb.resize(d, 0.0);
                                                    }
                                                    let ctx = &mut cb[..d];
                                                    for value in ctx.iter_mut() {
                                                        *value = 0.0;
                                                    }

                                                    let mut m = f32::NEG_INFINITY;
                                                    let mut l = 0.0f32;

                                                    for j in 0..total_len {
                                                        if causal && j > past_len {
                                                            break;
                                                        }
                                                        let k_row =
                                                            k4.slice(ndarray::s![bb, hk, j, ..]);
                                                        let v_row =
                                                            v4.slice(ndarray::s![bb, hk, j, ..]);
                                                        let score = dot_unrolled(
                                                            qbuf,
                                                            k_row.as_slice().expect(
                                                                "K cache row must be contiguous",
                                                            ),
                                                        ) * scale;
                                                        if score > m {
                                                            let s = (m - score).exp();
                                                            for i in 0..d {
                                                                ctx[i] = ctx[i] * s + v_row[i];
                                                            }
                                                            l = l * s + 1.0;
                                                            m = score;
                                                        } else {
                                                            let w = (score - m).exp();
                                                            for i in 0..d {
                                                                ctx[i] += w * v_row[i];
                                                            }
                                                            l += w;
                                                        }
                                                    }

                                                    let inv = 1.0f32 / (l + 1e-9);
                                                    for i in 0..d {
                                                        out_row[i] = ctx[i] * inv;
                                                    }
                                                });
                                            });
                                        },
                                    );

                                    self.w_o.forward_decode_slice_no_bias(&ob[..out_len])
                                })
                            });

                            return (output, Some(cache_handle));
                        })
                    })
                });
            }
            let q = { self.w_q.forward(x.clone()) };
            let k = { self.w_k.forward(x.clone()) };
            let v = { self.w_v.forward(x) };

            let q = permute(
                &reshape(&q, vec![b as i32, s as i32, h as i32, d as i32]),
                vec![0, 2, 1, 3],
            );
            let k = permute(
                &reshape(&k, vec![b as i32, s as i32, h_kv as i32, d as i32]),
                vec![0, 2, 1, 3],
            );
            let v = permute(
                &reshape(&v, vec![b as i32, s as i32, h_kv as i32, d as i32]),
                vec![0, 2, 1, 3],
            );

            // 3) 初始化/取出 cache（预分配）
            // (cache_handle 已在上面创建)

            // 4) RoPE：offset = past_len（S>1 prefill 路径保持原实现）
            let q_rot = self.rope.forward(&q, past_len);
            let k_rot = self.rope.forward(&k, past_len);

            // 5) 写入 cache（不 cat）
            {
                k_rot.with_storage_view_preferring(StoragePreference::F32Compute, |k_view| {
                    v.with_storage_view_preferring(StoragePreference::F32Compute, |v_view| {
                        let k_src = match k_view {
                            TensorStorageView::F32(view) => {
                                view.into_dimensionality::<ndarray::Ix4>().unwrap()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute view expected for cache write")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute view expected for cache write")
                            }
                        };

                        let v_src = match v_view {
                            TensorStorageView::F32(view) => {
                                view.into_dimensionality::<ndarray::Ix4>().unwrap()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute view expected for cache write")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute view expected for cache write")
                            }
                        };

                        let mut c = cache_handle.borrow_mut();
                        let new_len = past_len + s;
                        assert!(
                            new_len <= self.max_seq,
                            "KV cache overflow: new_len={} > max_seq={}",
                            new_len,
                            self.max_seq
                        );

                        // KV cache 写入：
                        // - decode_step (S=1) 走更轻量的 slice + copy_from_slice（避免 assign 的逐元素/广播开销）
                        // - S>1 保持 assign（依然是 view 写入，不产生额外 cat/copy）
                        if s == 1 {
                            let d = k_src.dim().3;
                            let h_kv = k_src.dim().1;
                            for bb in 0..b {
                                for hk in 0..h_kv {
                                    let src_k = k_src.slice(ndarray::s![bb, hk, 0, ..]);
                                    let src_v = v_src.slice(ndarray::s![bb, hk, 0, ..]);
                                    c.k.with_native_storage_view_mut(|k_view| {
                                        write_cache_token_row(
                                            k_view,
                                            bb,
                                            hk,
                                            past_len,
                                            src_k.as_slice().expect("src_k not contiguous"),
                                        );
                                    });
                                    c.v.with_native_storage_view_mut(|v_view| {
                                        write_cache_token_row(
                                            v_view,
                                            bb,
                                            hk,
                                            past_len,
                                            src_v.as_slice().expect("src_v not contiguous"),
                                        );
                                    });
                                    debug_assert_eq!(src_k.len(), d);
                                    debug_assert_eq!(src_v.len(), d);
                                }
                            }
                        } else {
                            c.k.with_native_storage_view_mut(|k_view| {
                                write_cache_range(k_view, past_len, new_len, &k_src);
                            });
                            c.v.with_native_storage_view_mut(|v_view| {
                                write_cache_range(v_view, past_len, new_len, &v_src);
                            });
                        }
                        c.len = new_len;
                    })
                });
            }

            // 6) GQA attention（不 repeat_kv）
            // 6) GQA attention（不 repeat_kv）。为了绕开 eval 的 permute/reshape copy，
            // 这里直接产出 [B,S,H,D]（BSHD）布局，后续 reshape 到 [B,S,H*D] 可视为 view。
            let context_bshd = {
                let c = cache_handle.borrow();
                let total_len = c.len;
                with_cache_f32_views(&c.k, &c.v, |k4_full, v4_full| {
                    let k4 = k4_full.slice(ndarray::s![.., .., 0..total_len, ..]);
                    let v4 = v4_full.slice(ndarray::s![.., .., 0..total_len, ..]);
                    q_rot.with_storage_view_preferring(StoragePreference::F32Compute, |q_view| {
                        let q4 = match q_view {
                            TensorStorageView::F32(view) => {
                                view.into_dimensionality::<ndarray::Ix4>().unwrap()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute view expected for attention")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute view expected for attention")
                            }
                        };
                        gqa_attention_no_repeat_bshd_view(
                            &q4,
                            &k4,
                            &v4,
                            self.scale,
                            self.causal,
                            n_rep,
                            past_len,
                        )
                    })
                })
            };

            // 7) 输出投影：context [B,S,H,D] -> [B,S,H*D] -> w_o （全 view，不触发 copy）
            let context = Tensor::from_data_no_grad(context_bshd.into_dyn().into_shared());
            let context = reshape(&context, vec![b as i32, s as i32, (h * d) as i32]);
            let output = self.w_o.forward(context);

            return (output, Some(cache_handle));
        }
        let q = { self.w_q.forward(x.clone()) };
        let k = { self.w_k.forward(x.clone()) };
        let v = { self.w_v.forward(x) };

        // train 路径：走原来的逻辑（可导）
        let q = permute(
            &reshape(&q, vec![b as i32, s as i32, h as i32, d as i32]),
            vec![0, 2, 1, 3],
        );
        let k = permute(
            &reshape(&k, vec![b as i32, s as i32, h_kv as i32, d as i32]),
            vec![0, 2, 1, 3],
        );
        let v = permute(
            &reshape(&v, vec![b as i32, s as i32, h_kv as i32, d as i32]),
            vec![0, 2, 1, 3],
        );

        // 希望训练时禁止传 cache：
        if cache.is_some() {
            panic!("Train path does not accept eval KVCache. Use eval_mode + cache for decoding.");
        }

        // 3) RoPE（训练全序列 offset=0）
        let q_rot = self.rope.forward(&q, 0);
        let k_rot = self.rope.forward(&k, 0);

        // 4) Repeat KV heads
        let k_up = if n_rep > 1 {
            repeat_kv(k_rot.clone(), n_rep)
        } else {
            k_rot.clone()
        };

        let v_up = if n_rep > 1 {
            repeat_kv(v.clone(), n_rep)
        } else {
            v.clone()
        };

        // 5) Attention
        let k_t = permute(&k_up, vec![0, 1, 3, 2]);
        let scores = batch_matmul(&q_rot, &k_t);
        let attn_probs = fused_softmax(&scores, self.scale, self.causal);
        let context = batch_matmul(&attn_probs, &v_up);

        // 6) Output
        let context = permute(&context, vec![0, 2, 1, 3]);
        let context = reshape(&context, vec![b as i32, s as i32, (h * d) as i32]);
        let output = self.w_o.forward(context);

        // 训练路径默认不返回 cache
        (output, None)
    }
}

impl Module for SelfAttention {
    fn forward(&self, x: Tensor) -> Tensor {
        let (out, _) = self.forward(x, None);
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.w_q.parameters();
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p.extend(self.w_o.parameters());
        p
    }
}

// train 路径：repeat_kv（依赖 Tensor ops，可导）
// x: [B, H_kv, S, D] -> [B, H, S, D]

pub fn repeat_kv(x: Tensor, n_rep: usize) -> Tensor {
    let data_ref = x.data_ref();
    let shape = data_ref.shape();
    let (b, n_kv, s, d) = (shape[0], shape[1], shape[2], shape[3]);

    let contig_data = data_ref.as_standard_layout();

    let expanded = contig_data
        .into_shape((b, n_kv, 1, s, d))
        .expect("Failed to expand KV shape");

    let broadcasted = expanded
        .broadcast((b, n_kv, n_rep, s, d))
        .expect("Failed to broadcast KV");

    let res = broadcasted
        .to_owned()
        .into_shape((b, n_kv * n_rep, s, d))
        .expect("Failed to flatten repeated KV heads");

    Tensor::new(res.into_dyn())
}

// eval 路径核心：GQA attention（不 repeat_kv）
// q: [B, H, S, D]
// k/v: [B, H_kv, L, D]
// 返回 context: [B, S, H, D]（BSHD，便于后续 reshape 到 [B,S,H*D] 视为 view）
// past_len: cache 写入前的长度（用于 causal mask 的 absolute index）

fn gqa_attention_no_repeat_bshd_view(
    q: &ndarray::ArrayView4<f32>,
    k: &ndarray::ArrayView4<f32>,
    v: &ndarray::ArrayView4<f32>,
    scale: f32,
    causal: bool,
    n_rep: usize,
    past_len: usize,
) -> Array4<f32> {
    let (b, h, s, d) = q.dim();
    let (b2, h_kv, l, d2) = k.dim();
    assert_eq!(b, b2);
    assert_eq!(d, d2);
    assert_eq!(h, h_kv * n_rep);

    let mut out = Array4::<f32>::zeros((b, s, h, d));

    // 将 out 变换为 [H,B,S,D]，对 head 维并行写入（线程间互不重叠）
    let mut out_hbsd = out.view_mut().permuted_axes([2, 0, 1, 3]);
    out_hbsd
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(hq, mut out_for_head)| {
            let hk = hq / n_rep;

            // decode(S=1) 热路径：online softmax + 直接累加 ctx，完全不需要 scores/ctx buffer
            if s == 1 {
                for bb in 0..b {
                    let q_vec = q.slice(ndarray::s![bb, hq, 0, ..]); // [D]
                    let k_mat = k.slice(ndarray::s![bb, hk, .., ..]); // [L,D]
                    let v_mat = v.slice(ndarray::s![bb, hk, .., ..]); // [L,D]

                    let q_abs = past_len; // i == 0

                    // 1) max
                    let mut maxv = f32::NEG_INFINITY;
                    for j in 0..l {
                        // causal mask
                        if causal && j > q_abs {
                            continue;
                        }
                        let kj = k_mat.slice(ndarray::s![j, ..]);
                        let mut dot = 0.0f32;
                        for i in 0..d {
                            dot += q_vec[i] * kj[i];
                        }
                        let score = dot * scale;
                        if score > maxv {
                            maxv = score;
                        }
                    }

                    // 2) sum + ctx
                    let mut sum = 0.0f32;
                    // out_for_head: [B,S,D] and S==1
                    let mut out_row = out_for_head.slice_mut(ndarray::s![bb, 0, ..]);
                    out_row.fill(0.0);

                    for j in 0..l {
                        if causal && j > q_abs {
                            continue;
                        }
                        let kj = k_mat.slice(ndarray::s![j, ..]);
                        let mut dot = 0.0f32;
                        for i in 0..d {
                            dot += q_vec[i] * kj[i];
                        }
                        let w = (dot * scale - maxv).exp();
                        sum += w;

                        for i in 0..d {
                            out_row[i] += w * v_mat[[j, i]];
                        }
                    }

                    let inv = 1.0f32 / (sum + 1e-9);
                    for i in 0..d {
                        out_row[i] *= inv;
                    }
                }
                return;
            }

            // per-thread buffer reuse
            with_attention_work_buffers(s * l, s * d, |scores_buf, ctx_buf| {
                let mut scores = ndarray::ArrayViewMut2::from_shape((s, l), scores_buf)
                    .expect("scores buffer shape mismatch");
                let mut ctx = ndarray::ArrayViewMut2::from_shape((s, d), ctx_buf)
                    .expect("ctx buffer shape mismatch");

                for bb in 0..b {
                    let q_mat = q.slice(ndarray::s![bb, hq, .., ..]); // [S,D]
                    let k_mat = k.slice(ndarray::s![bb, hk, .., ..]); // [L,D]
                    let v_mat = v.slice(ndarray::s![bb, hk, .., ..]); // [L,D]

                    scores.fill(0.0);
                    general_mat_mul(1.0, &q_mat, &k_mat.t(), 0.0, &mut scores);
                    softmax_inplace_view(&mut scores, scale, causal, past_len);

                    ctx.fill(0.0);
                    general_mat_mul(1.0, &scores, &v_mat, 0.0, &mut ctx);

                    // out_for_head: [B,S,D]
                    out_for_head.slice_mut(ndarray::s![bb, .., ..]).assign(&ctx);
                }
            });
        });

    out
}

fn softmax_inplace_view(
    scores: &mut ndarray::ArrayViewMut2<f32>,
    scale: f32,
    causal: bool,
    past_len: usize,
) {
    let (s, l) = scores.dim();

    for i in 0..s {
        // query 的 absolute index
        let q_abs = past_len + i;

        // 1) scale + causal mask
        for j in 0..l {
            let mut val = scores[(i, j)] * scale;
            if causal && j > q_abs {
                val = f32::NEG_INFINITY;
            }
            scores[(i, j)] = val;
        }

        // 2) stable softmax
        let mut maxv = f32::NEG_INFINITY;
        for j in 0..l {
            let v = scores[(i, j)];
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..l {
            let e = (scores[(i, j)] - maxv).exp();
            scores[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0f32 / (sum + 1e-9);
        for j in 0..l {
            scores[(i, j)] *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    use crate::precision::{
        DType, PrecisionConfig, set_default_runtime_dtype, with_precision_config,
    };
    use ndarray::{Array, IxDyn};

    fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let t = Tensor::from_array_no_grad(
            Array::from_shape_vec(IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
        );
        t.cast_inplace(dtype);
        t
    }

    #[test]
    #[should_panic(expected = "n_head must be > 0")]
    fn self_attention_rejects_zero_query_heads() {
        let _ = SelfAttention::new(16, 0, 1, 8, 10000.0, true);
    }

    #[test]
    #[should_panic(expected = "n_kv_head must be > 0")]
    fn self_attention_rejects_zero_kv_heads() {
        let _ = SelfAttention::new(16, 4, 0, 8, 10000.0, true);
    }

    #[test]
    #[should_panic(expected = "n_head must be divisible by n_kv_head")]
    fn self_attention_rejects_non_divisible_gqa_ratio() {
        let _ = SelfAttention::new(12, 6, 4, 8, 10000.0, true);
    }

    #[test]
    #[should_panic(expected = "attention input must be [B,S,H]")]
    fn self_attention_rejects_non_3d_input() {
        let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
        let input = make_tensor(&[1, 8], vec![0.0; 8], DType::F32);
        no_grad(|| {
            let _ = attn.forward(input, None);
        });
    }

    #[test]
    fn no_grad_forward_keeps_bf16_input_storage() {
        let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
        let input = make_tensor(
            &[1, 2, 8],
            (0..16).map(|i| i as f32 * 0.1 - 0.5).collect(),
            DType::BF16,
        );

        no_grad(|| {
            let _ = attn.forward(input.clone(), None);
        });

        assert_eq!(input.dtype(), DType::BF16);
        input.with_storage_view(|view| match view {
            crate::autograd::TensorStorageView::BF16(_) => {}
            crate::autograd::TensorStorageView::F16(_) => {
                panic!("shape inspection should not materialize bf16 attention input")
            }
            crate::autograd::TensorStorageView::F32(_) => {
                panic!("shape inspection should not materialize bf16 attention input")
            }
        });
    }

    #[test]
    fn kv_cache_creation_follows_runtime_dtype() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let cache = KVCacheInner::new(1, 1, 4, 4);
                assert_eq!(cache.dtype, DType::BF16);
                assert!(cache.follows_global_dtype);
                cache.k.with_storage_view_preferring(
                    StoragePreference::Native,
                    |view| match view {
                        TensorStorageView::BF16(view) => assert_eq!(view.shape(), &[1, 1, 4, 4]),
                        TensorStorageView::F16(_) => {
                            panic!("kv cache should follow bf16 runtime dtype")
                        }
                        TensorStorageView::F32(_) => {
                            panic!("kv cache should follow bf16 runtime dtype")
                        }
                    },
                );
            },
        );
    }

    #[test]
    fn kv_cache_manual_cast_disables_global_following() {
        let mut cache = KVCacheInner::new(1, 1, 4, 4);
        assert!(cache.follows_global_dtype);
        cache.cast_inplace(DType::BF16);
        assert_eq!(cache.dtype, DType::BF16);
        assert!(!cache.follows_global_dtype);
        cache
            .v
            .with_storage_view_preferring(StoragePreference::Native, |view| match view {
                TensorStorageView::BF16(view) => assert_eq!(view.shape(), &[1, 1, 4, 4]),
                TensorStorageView::F16(_) => {
                    panic!("manual cast should switch cache to explicit bf16 storage")
                }
                TensorStorageView::F32(_) => {
                    panic!("manual cast should switch cache to explicit bf16 storage")
                }
            });
    }

    #[test]
    fn kv_cache_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let cache = KVCacheInner::new_with_dtype(1, 1, 4, 4, DType::F32);
                assert_eq!(cache.dtype, DType::F32);
                assert!(!cache.follows_global_dtype);
                cache.k.with_storage_view_preferring(
                    StoragePreference::Native,
                    |view| match view {
                        TensorStorageView::F32(view) => assert_eq!(view.shape(), &[1, 1, 4, 4]),
                        TensorStorageView::F16(_) => {
                            panic!(
                                "explicit kv cache dtype should override global bf16 runtime dtype"
                            )
                        }
                        TensorStorageView::BF16(_) => {
                            panic!(
                                "explicit kv cache dtype should override global bf16 runtime dtype"
                            )
                        }
                    },
                );
            },
        );
    }

    #[test]
    #[should_panic(expected = "KV cache batch size must be > 0")]
    fn kv_cache_rejects_zero_batch() {
        let _ = KVCacheInner::new(0, 1, 4, 4);
    }

    #[test]
    #[should_panic(expected = "SelfAttention KV cache shape mismatch")]
    fn self_attention_rejects_cache_shape_mismatch() {
        let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
        let input = make_tensor(&[1, 1, 8], vec![0.0; 8], DType::F32);
        let cache: KVCache = Rc::new(RefCell::new(KVCacheInner::new(2, 1, 8, 4)));
        no_grad(|| {
            let _ = attn.forward(input, Some(cache));
        });
    }

    #[test]
    #[should_panic(expected = "SelfAttention KV cache current length out of bounds")]
    fn self_attention_rejects_cache_len_overflow() {
        let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
        let input = make_tensor(&[1, 1, 8], vec![0.0; 8], DType::F32);
        let mut cache = KVCacheInner::new(1, 1, 8, 4);
        cache.len = 9;
        let cache: KVCache = Rc::new(RefCell::new(cache));
        no_grad(|| {
            let _ = attn.forward(input, Some(cache));
        });
    }

    #[test]
    fn self_attention_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::BF16,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies: false,
            },
            || {
                let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::F32);
                for weight in [
                    &attn.w_q.weight,
                    &attn.w_k.weight,
                    &attn.w_v.weight,
                    &attn.w_o.weight,
                ] {
                    assert_eq!(weight.dtype(), DType::F32);
                }
                assert_eq!(attn.rope.cache_dtype(), DType::F32);
            },
        );
    }

    #[test]
    fn self_attention_default_construction_splits_parameter_and_runtime_defaults() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
                for weight in [
                    &attn.w_q.weight,
                    &attn.w_k.weight,
                    &attn.w_v.weight,
                    &attn.w_o.weight,
                ] {
                    assert_eq!(weight.dtype(), DType::F32);
                }
                assert_eq!(attn.rope.cache_dtype(), DType::BF16);
            },
        );
    }

    #[test]
    fn self_attention_default_construction_captures_runtime_dtype_for_future_cache_creation() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
                set_default_runtime_dtype(DType::F32);
                let input = make_tensor(&[1, 1, 8], vec![0.0; 8], DType::F32);
                let (_out, cache) = no_grad(|| attn.forward(input, None));
                let cache = cache.expect("no_grad attention should create cache");
                assert_eq!(cache.borrow().dtype, DType::BF16);
            },
        );
    }
}
