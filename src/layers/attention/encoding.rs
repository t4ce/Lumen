use crate::autograd::{StoragePreference, Tensor, TensorData, TensorStorageView, is_no_grad};
use crate::precision::{DType, default_runtime_dtype};
use half::{bf16, f16};
use ndarray::{Array, ArrayD, Ix2, Zip, s};
use std::cell::RefCell;
use std::rc::Rc;

pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    // 缓存预计算的 cos/sin
    // Shape: [1, 1, Max_Seq, Dim]
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self::new_with_dtype(dim, max_seq_len, theta, default_runtime_dtype())
    }

    pub fn new_with_dtype(dim: usize, max_seq_len: usize, theta: f32, dtype: DType) -> Self {
        assert!(
            dtype.is_float(),
            "RotaryEmbedding cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        let (cos, sin) = Self::precompute_freqs_cis(dim, max_seq_len, theta);
        let cos_cache = Tensor::from_array_no_grad(cos);
        let sin_cache = Tensor::from_array_no_grad(sin);
        cos_cache.cast_inplace(dtype);
        sin_cache.cast_inplace(dtype);

        Self {
            dim,
            max_seq_len,
            cos_cache,
            sin_cache,
        }
    }

    #[inline]
    pub fn cache_dtype(&self) -> DType {
        self.cos_cache.dtype()
    }

    fn precompute_freqs_cis(
        dim: usize,
        max_seq_len: usize,
        theta: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let half_d = dim / 2;
        let mut cos_arr = Array::zeros((1, 1, max_seq_len, dim));
        let mut sin_arr = Array::zeros((1, 1, max_seq_len, dim));

        for i in 0..max_seq_len {
            let pos = i as f32;
            for j in 0..half_d {
                let freq = 1.0 / theta.powf((j as f32 * 2.0) / dim as f32);
                let val = pos * freq;

                let c = val.cos();
                let s = val.sin();

                cos_arr[[0, 0, i, j]] = c;
                cos_arr[[0, 0, i, j + half_d]] = c;
                sin_arr[[0, 0, i, j]] = s;
                sin_arr[[0, 0, i, j + half_d]] = s;
            }
        }
        (cos_arr.into_dyn(), sin_arr.into_dyn())
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Tensor {
        let build_graph = !is_no_grad() && x.requires_grad();

        if !build_graph {
            return x.with_storage_view_preferring(StoragePreference::Native, |x_view| {
                let shape = match &x_view {
                    TensorStorageView::F32(x_view) => x_view.shape().to_vec(),
                    TensorStorageView::F16(x_view) => x_view.shape().to_vec(),
                    TensorStorageView::BF16(x_view) => x_view.shape().to_vec(),
                };
                assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
                let (_b, _h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
                assert_eq!(d, self.dim, "RoPE dimension mismatch");

                let end = offset + seq_len;
                if end > self.max_seq_len {
                    panic!(
                        "RoPE index out of range: offset {} + len {} > max {}",
                        offset, seq_len, self.max_seq_len
                    );
                }

                self.cos_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |cos_view| {
                        self.sin_cache.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |sin_view| {
                                let cos_4d = match cos_view {
                                    TensorStorageView::F32(view) => view
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .expect("RoPE Cache dimensionality mismatch"),
                                    TensorStorageView::F16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                    TensorStorageView::BF16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                };
                                let sin_4d = match sin_view {
                                    TensorStorageView::F32(view) => view
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .expect("RoPE Cache dimensionality mismatch"),
                                    TensorStorageView::F16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                    TensorStorageView::BF16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                };
                                let cos_slice_2d = cos_4d
                                    .slice(s![0, 0, offset..end, ..])
                                    .into_dimensionality::<Ix2>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                let sin_slice_2d = sin_4d
                                    .slice(s![0, 0, offset..end, ..])
                                    .into_dimensionality::<Ix2>()
                                    .expect("RoPE Cache dimensionality mismatch");

                                match x_view {
                                    TensorStorageView::F32(x_view) => {
                                        let mut out = Array::zeros(x_view.raw_dim());
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]];
                                                                let x2 = x_h[[ss, j + half]];
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] =
                                                                    x1 * c - x2 * s_val;
                                                                out_h[[ss, j + half]] =
                                                                    x2 * c + x1 * s_val;
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_array_no_grad(out.into_dyn())
                                    }
                                    TensorStorageView::F16(x_view) => {
                                        let mut out = ndarray::ArrayD::<f16>::from_elem(
                                            ndarray::IxDyn(&shape),
                                            f16::from_bits(0),
                                        );
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]].to_f32();
                                                                let x2 =
                                                                    x_h[[ss, j + half]].to_f32();
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] = f16::from_f32(
                                                                    x1 * c - x2 * s_val,
                                                                );
                                                                out_h[[ss, j + half]] =
                                                                    f16::from_f32(
                                                                        x2 * c + x1 * s_val,
                                                                    );
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_f16_data_no_grad(out.into_shared())
                                    }
                                    TensorStorageView::BF16(x_view) => {
                                        let mut out = ndarray::ArrayD::<bf16>::from_elem(
                                            ndarray::IxDyn(&shape),
                                            bf16::from_bits(0),
                                        );
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]].to_f32();
                                                                let x2 =
                                                                    x_h[[ss, j + half]].to_f32();
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] = bf16::from_f32(
                                                                    x1 * c - x2 * s_val,
                                                                );
                                                                out_h[[ss, j + half]] =
                                                                    bf16::from_f32(
                                                                        x2 * c + x1 * s_val,
                                                                    );
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_bf16_data_no_grad(out.into_shared())
                                    }
                                }
                            },
                        )
                    },
                )
            });
        }

        let x_data = x.data_ref();
        let shape = x_data.shape();
        assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
        let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(d, self.dim, "RoPE dimension mismatch");

        let end = offset + seq_len;
        if end > self.max_seq_len {
            panic!(
                "RoPE index out of range: offset {} + len {} > max {}",
                offset, seq_len, self.max_seq_len
            );
        }

        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_4d = match cos_view {
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

                        let sin_4d = match sin_view {
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
                        let cos_slice_2d = cos_4d
                            .slice(s![0, 0, offset..end, ..])
                            .into_dimensionality::<Ix2>()
                            .expect("RoPE Cache dimensionality mismatch");
                        let sin_slice_2d = sin_4d
                            .slice(s![0, 0, offset..end, ..])
                            .into_dimensionality::<Ix2>()
                            .expect("RoPE Cache dimensionality mismatch");

                        let mut out = Array::zeros(x_data.dim());

                        let x_view = x_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                        let mut out_view = out
                            .view_mut()
                            .into_dimensionality::<ndarray::Ix4>()
                            .unwrap();

                        Zip::from(out_view.outer_iter_mut())
                            .and(x_view.outer_iter())
                            .par_for_each(|mut out_b, x_b| {
                                Zip::from(out_b.outer_iter_mut())
                                    .and(x_b.outer_iter())
                                    .for_each(|mut out_h, x_h| {
                                        let half = d / 2;
                                        for ss in 0..seq_len {
                                            for j in 0..half {
                                                let x1 = x_h[[ss, j]];
                                                let x2 = x_h[[ss, j + half]];
                                                let c = cos_slice_2d[[ss, j]];
                                                let s_val = sin_slice_2d[[ss, j]];
                                                out_h[[ss, j]] = x1 * c - x2 * s_val;
                                                out_h[[ss, j + half]] = x2 * c + x1 * s_val;
                                            }
                                        }
                                    });
                            });

                        let x_clone = x.clone();
                        let cos_backward = cos_slice_2d.to_owned();
                        let sin_backward = sin_slice_2d.to_owned();

                        Tensor(Rc::new(RefCell::new(TensorData {
                            data: out.into_dyn().into_shared(),
                            f16_data: None,
                            bf16_data: None,
                            i8_data: None,
                            i8_scale: None,
                            has_f32_data: true,
                            storage_dtype: crate::precision::DType::F32,
                            cache_dirty: false,
                            is_parameter: false,
                            grad: None,
                            parents: vec![x.clone()],
                            backward_op: Some(std::rc::Rc::new(move |grad| {
                                let grad_view =
                                    grad.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                                let mut d_x = Array::zeros((b, h, seq_len, d));

                                Zip::from(d_x.outer_iter_mut())
                                    .and(grad_view.outer_iter())
                                    .par_for_each(|mut dx_b, g_b| {
                                        Zip::from(dx_b.outer_iter_mut())
                                            .and(g_b.outer_iter())
                                            .for_each(|mut dx_h, g_h| {
                                                let half = d / 2;
                                                for ss in 0..seq_len {
                                                    for j in 0..half {
                                                        let g1 = g_h[[ss, j]];
                                                        let g2 = g_h[[ss, j + half]];

                                                        let c = cos_backward[[ss, j]];
                                                        let s_val = sin_backward[[ss, j]];

                                                        dx_h[[ss, j]] = g1 * c + g2 * s_val;
                                                        dx_h[[ss, j + half]] = g2 * c - g1 * s_val;
                                                    }
                                                }
                                            });
                                    });

                                x_clone.add_grad(d_x.into_dyn());
                            })),
                            requires_grad: true,
                        })))
                    },
                )
            })
    }

    // Apply RoPE for a single token at absolute position `pos`.
    //
    // Decode (S=1) hot-path helper to avoid allocating intermediate q_rot/k_rot tensors.
    // `src` and `dst` must both have length == `self.dim`.
    #[inline]
    pub fn rope_1token_copy(&self, src: &[f32], dst: &mut [f32], pos: usize) {
        assert_eq!(src.len(), self.dim, "RoPE src len mismatch");
        assert_eq!(dst.len(), self.dim, "RoPE dst len mismatch");
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }

        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_row = match cos_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let sin_row = match sin_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let half = self.dim / 2;
                        for j in 0..half {
                            let x1 = src[j];
                            let x2 = src[j + half];
                            let c = cos_row[j];
                            let s_val = sin_row[j];
                            dst[j] = x1 * c - x2 * s_val;
                            dst[j + half] = x2 * c + x1 * s_val;
                        }
                    },
                )
            });
    }

    // Get (cos, sin) row at position `pos` as owned Vecs.
    // This is useful to pass into rayon-parallel decode kernels without capturing Tensor/Rc.
    pub fn cos_sin_row_vec(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }
        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_row = match cos_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let sin_row = match sin_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        (cos_row, sin_row)
                    },
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
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

    #[test]
    fn rope_no_grad_preserves_bf16_input_dtype() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let input_f32 = make_tensor(
            &[1, 1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            DType::F32,
        );
        let input_bf16 = make_tensor(
            &[1, 1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            DType::BF16,
        );

        let ref_out = no_grad(|| rope.forward(&input_f32, 0));
        let bf16_out = no_grad(|| rope.forward(&input_bf16, 0));

        assert_eq!(input_bf16.dtype(), DType::BF16);
        assert_eq!(bf16_out.dtype(), DType::BF16);

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        bf16_out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => panic!("bf16 RoPE output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 RoPE output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn rope_cache_creation_follows_runtime_dtype() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let rope = RotaryEmbedding::new(4, 8, 10000.0);
                assert_eq!(rope.cos_cache.dtype(), DType::BF16);
                assert_eq!(rope.sin_cache.dtype(), DType::BF16);
            },
        );
    }

    #[test]
    fn rope_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
                assert_eq!(rope.cos_cache.dtype(), DType::F32);
                assert_eq!(rope.sin_cache.dtype(), DType::F32);
            },
        );
    }
}
