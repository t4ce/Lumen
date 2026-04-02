use crate::autograd::{StoragePreference, Tensor, TensorStorageView, TensorStorageViewMut};
use crate::precision::{DType, default_runtime_dtype};
use half::f16;
use ndarray::{Array4, s};

#[derive(Clone)]
pub struct LlamaKVCache {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub max_seq_len: usize,
    pub dim: usize,
    pub head_num: usize,
    pub batch_size: usize,
    pub dtype: DType,
    pub follows_global_dtype: bool,
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
                .slice_mut(s![.., .., start..end, ..])
                .assign(src);
        }
        TensorStorageViewMut::F16(view) => {
            let mut dst4 = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]");
            for bb in 0..src.dim().0 {
                for hh in 0..src.dim().1 {
                    for ss in 0..src.dim().2 {
                        let src_row = src.slice(s![bb, hh, ss, ..]);
                        let mut dst_row = dst4.slice_mut(s![bb, hh, start + ss, ..]);
                        let dst_slice = dst_row
                            .as_slice_mut()
                            .expect("KV cache row must be contiguous");
                        for (dst, &value) in dst_slice.iter_mut().zip(src_row.iter()) {
                            *dst = f16::from_f32(value);
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
                for hh in 0..src.dim().1 {
                    for ss in 0..src.dim().2 {
                        let src_row = src.slice(s![bb, hh, ss, ..]);
                        let mut dst_row = dst4.slice_mut(s![bb, hh, start + ss, ..]);
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

fn native_cache_prefix(cache: &Tensor, current_len: usize) -> Tensor {
    cache.with_storage_view_preferring(StoragePreference::Native, |view| match view {
        TensorStorageView::F32(view) => {
            let sliced = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]")
                .slice(s![.., .., ..current_len, ..])
                .to_owned();
            Tensor::from_array_no_grad(sliced.into_dyn())
        }
        TensorStorageView::F16(view) => {
            let sliced = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]")
                .slice(s![.., .., ..current_len, ..])
                .to_owned()
                .into_dyn()
                .into_shared();
            Tensor::from_f16_data_no_grad(sliced)
        }
        TensorStorageView::BF16(view) => {
            let sliced = view
                .into_dimensionality::<ndarray::Ix4>()
                .expect("KV cache must be [B,H,S,D]")
                .slice(s![.., .., ..current_len, ..])
                .to_owned()
                .into_dyn()
                .into_shared();
            Tensor::from_bf16_data_no_grad(sliced)
        }
    })
}

impl LlamaKVCache {
    pub fn new(config: &crate::models::LlamaConfig) -> Self {
        Self::new_impl(config, default_runtime_dtype(), true)
    }

    fn new_impl(
        config: &crate::models::LlamaConfig,
        dtype: DType,
        follows_global_dtype: bool,
    ) -> Self {
        assert!(
            config.num_attention_heads > 0,
            "num_attention_heads must be > 0"
        );
        assert!(
            config.num_key_value_heads > 0,
            "num_key_value_heads must be > 0"
        );
        assert!(config.max_seq_len > 0, "max_seq_len must be > 0");
        assert!(
            dtype.is_float(),
            "KV cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        assert_eq!(
            config.hidden_size % config.num_attention_heads,
            0,
            "hidden_size must be divisible by num_attention_heads"
        );
        assert_eq!(
            config.num_attention_heads % config.num_key_value_heads,
            0,
            "num_attention_heads must be divisible by num_key_value_heads"
        );
        let max_seq_len = config.max_seq_len;
        let head_num = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let batch_size = 1;

        let k_cache = Tensor::from_array_no_grad(
            Array4::<f32>::zeros((batch_size, head_num, max_seq_len, head_dim)).into_dyn(),
        );
        let v_cache = Tensor::from_array_no_grad(
            Array4::<f32>::zeros((batch_size, head_num, max_seq_len, head_dim)).into_dyn(),
        );
        k_cache.cast_inplace(dtype);
        v_cache.cast_inplace(dtype);

        Self {
            k_cache,
            v_cache,
            max_seq_len,
            dim: head_dim,
            head_num,
            batch_size,
            dtype,
            follows_global_dtype,
        }
    }

    pub fn new_with_dtype(config: &crate::models::LlamaConfig, dtype: DType) -> Self {
        Self::new_impl(config, dtype, false)
    }

    pub fn cast_inplace(&mut self, dtype: DType) {
        assert!(
            dtype.is_float(),
            "KV cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        self.k_cache.cast_inplace(dtype);
        self.v_cache.cast_inplace(dtype);
        self.dtype = dtype;
        self.follows_global_dtype = false;
    }

    pub fn update(&mut self, k: &Tensor, v: &Tensor, start_pos: usize) {
        let k_shape = k.shape_vec();
        let v_shape = v.shape_vec();
        assert_eq!(k_shape.len(), 4, "K cache update expects [B,H,S,D]");
        assert_eq!(v_shape.len(), 4, "V cache update expects [B,H,S,D]");
        assert_eq!(k_shape, v_shape, "K/V cache update shape mismatch");
        assert_eq!(k_shape[0], self.batch_size, "KV cache batch size mismatch");
        assert_eq!(k_shape[1], self.head_num, "KV cache head count mismatch");
        assert_eq!(k_shape[3], self.dim, "KV cache head dim mismatch");
        assert!(
            start_pos <= self.max_seq_len,
            "KV cache start_pos out of bounds: {} > {}",
            start_pos,
            self.max_seq_len
        );
        let seq_len = k_shape[2];
        let end_pos = start_pos
            .checked_add(seq_len)
            .expect("KV cache end position overflow");
        assert!(
            end_pos <= self.max_seq_len,
            "KV Cache overflow! Max: {}, Current: {}",
            self.max_seq_len,
            end_pos
        );

        k.with_storage_view_preferring(StoragePreference::F32Compute, |k_view| {
            v.with_storage_view_preferring(StoragePreference::F32Compute, |v_view| {
                let k_view4 = match k_view {
                    TensorStorageView::F32(view) => view
                        .into_dimensionality::<ndarray::Ix4>()
                        .expect("K cache update expects [B,H,S,D]"),
                    TensorStorageView::F16(_) => {
                        unreachable!("f32 compute preference should expose f32 view")
                    }
                    TensorStorageView::BF16(_) => {
                        unreachable!("f32 compute preference should expose f32 view")
                    }
                };
                let v_view4 = match v_view {
                    TensorStorageView::F32(view) => view
                        .into_dimensionality::<ndarray::Ix4>()
                        .expect("V cache update expects [B,H,S,D]"),
                    TensorStorageView::F16(_) => {
                        unreachable!("f32 compute preference should expose f32 view")
                    }
                    TensorStorageView::BF16(_) => {
                        unreachable!("f32 compute preference should expose f32 view")
                    }
                };

                self.k_cache.with_native_storage_view_mut(|dst| {
                    write_cache_range(dst, start_pos, end_pos, &k_view4)
                });
                self.v_cache.with_native_storage_view_mut(|dst| {
                    write_cache_range(dst, start_pos, end_pos, &v_view4)
                });
            })
        });
    }

    pub fn get_view(&self, current_len: usize) -> (Tensor, Tensor) {
        assert!(
            current_len <= self.max_seq_len,
            "KV cache view length out of bounds: {} > {}",
            current_len,
            self.max_seq_len
        );

        (
            native_cache_prefix(&self.k_cache, current_len),
            native_cache_prefix(&self.v_cache, current_len),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LlamaConfig;
    use crate::precision::{PrecisionConfig, with_precision_config};
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

    fn test_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-5,
            max_seq_len: 4,
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn legacy_kv_cache_creation_follows_runtime_dtype() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let cache = LlamaKVCache::new(&test_config());
                assert_eq!(cache.dtype, DType::BF16);
                assert!(cache.follows_global_dtype);
                assert_eq!(cache.k_cache.dtype(), DType::BF16);
                assert_eq!(cache.v_cache.dtype(), DType::BF16);
            },
        );
    }

    #[test]
    fn legacy_kv_cache_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let cache = LlamaKVCache::new_with_dtype(&test_config(), DType::F32);
                assert_eq!(cache.dtype, DType::F32);
                assert!(!cache.follows_global_dtype);
                assert_eq!(cache.k_cache.dtype(), DType::F32);
                assert_eq!(cache.v_cache.dtype(), DType::F32);
            },
        );
    }

    #[test]
    fn legacy_kv_cache_manual_cast_disables_global_following() {
        let mut cache = LlamaKVCache::new(&test_config());
        assert!(cache.follows_global_dtype);
        cache.cast_inplace(DType::BF16);
        assert_eq!(cache.dtype, DType::BF16);
        assert!(!cache.follows_global_dtype);
        assert_eq!(cache.k_cache.dtype(), DType::BF16);
        assert_eq!(cache.v_cache.dtype(), DType::BF16);
    }

    #[test]
    fn legacy_kv_cache_get_view_preserves_native_dtype() {
        let mut cache = LlamaKVCache::new_with_dtype(&test_config(), DType::BF16);
        let k = make_tensor(
            &[1, 1, 2, 4],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            DType::BF16,
        );
        let v = make_tensor(
            &[1, 1, 2, 4],
            vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            DType::BF16,
        );
        cache.update(&k, &v, 0);

        let (k_view, v_view) = cache.get_view(2);
        assert_eq!(k_view.dtype(), DType::BF16);
        assert_eq!(v_view.dtype(), DType::BF16);

        let k_vals = k_view.data();
        let v_vals = v_view.data();
        assert_eq!(
            k_vals.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            v_vals.iter().copied().collect::<Vec<_>>(),
            vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        );
    }

    #[test]
    #[should_panic(expected = "K/V cache update shape mismatch")]
    fn kv_cache_rejects_mismatched_kv_shapes() {
        let mut cache = LlamaKVCache::new(&test_config());
        let k = make_tensor(&[1, 1, 1, 4], vec![0.0; 4], DType::F32);
        let v = make_tensor(&[1, 1, 2, 4], vec![0.0; 8], DType::F32);
        cache.update(&k, &v, 0);
    }

    #[test]
    #[should_panic(expected = "KV cache view length out of bounds")]
    fn kv_cache_rejects_oversized_view() {
        let cache = LlamaKVCache::new(&test_config());
        let _ = cache.get_view(5);
    }
}
