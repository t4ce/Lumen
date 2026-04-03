use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView, is_no_grad,
};
use crate::init::{InitType, tensor_init, tensor_init_with_dtype};
use crate::module::Module;
use crate::precision::DType;
use half::{bf16, f16};
use ndarray::{Array, Zip};
use std::cell::RefCell;
use std::ops::AddAssign;
use std::rc::Rc;

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

#[inline]
fn parse_embedding_index(value: f32, position: usize, vocab_size: usize) -> usize {
    assert!(
        value.is_finite(),
        "Embedding index at position {} must be finite, got {}",
        position,
        value
    );
    assert!(
        value >= 0.0,
        "Embedding index at position {} must be >= 0, got {}",
        position,
        value
    );
    assert!(
        value.fract() == 0.0,
        "Embedding index at position {} must be an integer, got {}",
        position,
        value
    );
    let idx = value as usize;
    assert!(
        idx < vocab_size,
        "Embedding index out of bounds at position {}: {} >= {}",
        position,
        idx,
        vocab_size
    );
    idx
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let weight = tensor_init(vec![vocab_size, embed_dim], InitType::KaimingNormal);
        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    pub fn new_with_dtype(vocab_size: usize, embed_dim: usize, dtype: DType) -> Self {
        let weight =
            tensor_init_with_dtype(vec![vocab_size, embed_dim], InitType::KaimingNormal, dtype);
        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    pub fn forward(&self, indices: &Tensor) -> Tensor {
        let e_dim = self.embed_dim;
        let v_size = self.vocab_size;
        let build_graph = !is_no_grad() && self.weight.requires_grad();

        let mut out_shape = indices.shape_vec();
        out_shape.push(e_dim);

        let num_elements = indices.len();
        let idx_values = indices.with_storage_view(|idx_view| match idx_view {
            TensorStorageView::F32(idx_view) => idx_view
                .iter()
                .enumerate()
                .map(|(pos, &v)| parse_embedding_index(v, pos, v_size))
                .collect::<Vec<_>>(),
            TensorStorageView::F16(idx_view) => idx_view
                .iter()
                .enumerate()
                .map(|(pos, v)| parse_embedding_index(v.to_f32(), pos, v_size))
                .collect::<Vec<_>>(),
            TensorStorageView::BF16(idx_view) => idx_view
                .iter()
                .enumerate()
                .map(|(pos, v)| parse_embedding_index(v.to_f32(), pos, v_size))
                .collect::<Vec<_>>(),
        });

        if !build_graph {
            if self.weight.dtype() == DType::I8 {
                return match self.weight.native_storage_owned() {
                    TensorStorageOwned::I8(w_data, scale) => {
                        let mut out = ndarray::ArrayD::<i8>::zeros(ndarray::IxDyn(&out_shape));
                        let mut out_flat = out
                            .view_mut()
                            .into_shape((num_elements, e_dim))
                            .expect("Flatten output failed");
                        let w_2d = w_data
                            .view()
                            .into_dimensionality::<ndarray::Ix2>()
                            .expect("Embedding weight must be 2D");
                        Zip::from(out_flat.outer_iter_mut())
                            .and(&idx_values)
                            .par_for_each(|mut out_row, &idx| {
                                let w_row = w_2d.slice(ndarray::s![idx, ..]);
                                out_row.assign(&w_row);
                            });
                        Tensor::from_i8_data_no_grad(out.into_shared(), scale)
                    }
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_) => unreachable!("checked i8 weight above"),
                };
            }

            return self
                .weight
                .with_storage_view_preferring(StoragePreference::Native, |w_view| match w_view {
                    TensorStorageView::F32(w_view) => {
                        let mut out = Array::zeros(out_shape.clone());
                        let mut out_flat = out
                            .view_mut()
                            .into_shape((num_elements, e_dim))
                            .expect("Flatten output failed");
                        let w_2d = w_view
                            .into_dimensionality::<ndarray::Ix2>()
                            .expect("Embedding weight must be 2D");
                        Zip::from(out_flat.outer_iter_mut())
                            .and(&idx_values)
                            .par_for_each(|mut out_row, &idx| {
                                let w_row = w_2d.slice(ndarray::s![idx, ..]);
                                out_row.assign(&w_row);
                            });
                        Tensor::from_array_no_grad(out.into_dyn())
                    }
                    TensorStorageView::F16(w_view) => {
                        let mut out = ndarray::ArrayD::<f16>::from_elem(
                            ndarray::IxDyn(&out_shape),
                            f16::from_bits(0),
                        );
                        let mut out_flat = out
                            .view_mut()
                            .into_shape((num_elements, e_dim))
                            .expect("Flatten output failed");
                        let w_2d = w_view
                            .into_dimensionality::<ndarray::Ix2>()
                            .expect("Embedding weight must be 2D");
                        Zip::from(out_flat.outer_iter_mut())
                            .and(&idx_values)
                            .par_for_each(|mut out_row, &idx| {
                                let w_row = w_2d.slice(ndarray::s![idx, ..]);
                                out_row.assign(&w_row);
                            });
                        Tensor::from_f16_data_no_grad(out.into_shared())
                    }
                    TensorStorageView::BF16(w_view) => {
                        let mut out = ndarray::ArrayD::<bf16>::from_elem(
                            ndarray::IxDyn(&out_shape),
                            bf16::from_bits(0),
                        );
                        let mut out_flat = out
                            .view_mut()
                            .into_shape((num_elements, e_dim))
                            .expect("Flatten output failed");
                        let w_2d = w_view
                            .into_dimensionality::<ndarray::Ix2>()
                            .expect("Embedding weight must be 2D");
                        Zip::from(out_flat.outer_iter_mut())
                            .and(&idx_values)
                            .par_for_each(|mut out_row, &idx| {
                                let w_row = w_2d.slice(ndarray::s![idx, ..]);
                                out_row.assign(&w_row);
                            });
                        Tensor::from_bf16_data_no_grad(out.into_shared())
                    }
                });
        }

        let mut out = Array::zeros(out_shape);
        let mut out_flat = out
            .view_mut()
            .into_shape((num_elements, e_dim))
            .expect("Flatten output failed");

        self.weight.with_storage_view(|w_view| match w_view {
            TensorStorageView::F32(w_view) => {
                let w_2d = w_view
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("Embedding weight must be 2D");
                Zip::from(out_flat.outer_iter_mut())
                    .and(&idx_values)
                    .par_for_each(|mut out_row, &idx| {
                        let w_row = w_2d.slice(ndarray::s![idx, ..]);
                        out_row.assign(&w_row);
                    });
            }
            TensorStorageView::F16(w_view) => {
                let w_2d = w_view
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("Embedding weight must be 2D");
                Zip::from(out_flat.outer_iter_mut())
                    .and(&idx_values)
                    .par_for_each(|mut out_row, &idx| {
                        let w_row = w_2d.slice(ndarray::s![idx, ..]);
                        for (dst, &src) in out_row.iter_mut().zip(w_row.iter()) {
                            *dst = src.to_f32();
                        }
                    });
            }
            TensorStorageView::BF16(w_view) => {
                let w_2d = w_view
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("Embedding weight must be 2D");
                Zip::from(out_flat.outer_iter_mut())
                    .and(&idx_values)
                    .par_for_each(|mut out_row, &idx| {
                        let w_row = w_2d.slice(ndarray::s![idx, ..]);
                        for (dst, &src) in out_row.iter_mut().zip(w_row.iter()) {
                            *dst = src.to_f32();
                        }
                    });
            }
        });

        let out_dyn = out.into_dyn();

        let indices_clone = indices.clone();
        let w_clone = self.weight.clone();
        let v_snap = v_size;
        let e_snap = e_dim;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: out_dyn.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![indices.clone(), self.weight.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let binding = indices_clone.data_ref();
                let idx_flat = binding.view().into_shape(num_elements).unwrap();
                let grad_2d = grad.view().into_shape((num_elements, e_snap)).unwrap();

                let mut d_w = Array::zeros((v_snap, e_snap));
                for (i, &idx_f32) in idx_flat.iter().enumerate() {
                    let idx = parse_embedding_index(idx_f32, i, v_snap);
                    d_w.slice_mut(ndarray::s![idx, ..])
                        .add_assign(&grad_2d.slice(ndarray::s![i, ..]));
                }
                w_clone.add_grad(d_w.into_dyn());
            })),
            requires_grad: true,
        })))
    }
}

impl Module for Embedding {
    fn forward(&self, x: Tensor) -> Tensor {
        self.forward(&x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
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
    fn embedding_accepts_integer_indices() {
        let emb = Embedding::new(8, 4);
        let indices = make_tensor(&[1, 3], vec![0.0, 2.0, 7.0], DType::F32);
        let out = no_grad(|| emb.forward(&indices));
        assert_eq!(out.shape_vec(), vec![1, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "must be an integer")]
    fn embedding_rejects_fractional_indices() {
        let emb = Embedding::new(8, 4);
        let indices = make_tensor(&[1, 1], vec![1.5], DType::F32);
        no_grad(|| {
            let _ = emb.forward(&indices);
        });
    }

    #[test]
    #[should_panic(expected = "must be >= 0")]
    fn embedding_rejects_negative_indices() {
        let emb = Embedding::new(8, 4);
        let indices = make_tensor(&[1, 1], vec![-1.0], DType::BF16);
        no_grad(|| {
            let _ = emb.forward(&indices);
        });
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn embedding_rejects_nan_indices() {
        let emb = Embedding::new(8, 4);
        let indices = make_tensor(&[1, 1], vec![f32::NAN], DType::F32);
        no_grad(|| {
            let _ = emb.forward(&indices);
        });
    }

    #[test]
    fn embedding_no_grad_preserves_bf16_weight_dtype() {
        let emb = Embedding::new(4, 2);
        emb.weight.set_array_f32_with_dtype(
            Array::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .expect("weight shape mismatch")
                .into_dyn(),
            DType::BF16,
        );
        let indices = make_tensor(&[1, 2], vec![1.0, 3.0], DType::F32);

        let out = no_grad(|| emb.forward(&indices));
        assert_eq!(out.dtype(), DType::BF16);
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, vec![2.0, 3.0, 6.0, 7.0]);
            }
            TensorStorageView::F16(_) => {
                panic!("bf16 embedding output should stay bf16 in no-grad")
            }
            TensorStorageView::F32(_) => {
                panic!("bf16 embedding output should stay bf16 in no-grad")
            }
        });
    }

    #[test]
    fn embedding_no_grad_preserves_i8_weight_dtype() {
        let emb = Embedding::new(4, 2);
        emb.weight.set_array_f32_with_dtype(
            Array::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .expect("weight shape mismatch")
                .into_dyn(),
            DType::I8,
        );
        let indices = make_tensor(&[1, 2], vec![1.0, 3.0], DType::F32);

        let out = no_grad(|| emb.forward(&indices));
        assert_eq!(out.dtype(), DType::I8);
        let ref_vals = no_grad(|| {
            let ref_emb = Embedding::new(4, 2);
            ref_emb.weight.set_array_f32_with_dtype(
                Array::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                    .expect("weight shape mismatch")
                    .into_dyn(),
                DType::I8,
            );
            ref_emb
                .forward(&indices)
                .data_ref()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        });
        let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(out_vals, ref_vals);
    }

    #[test]
    fn embedding_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::BF16,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies: false,
            },
            || {
                let emb = Embedding::new_with_dtype(8, 4, DType::F32);
                assert_eq!(emb.weight.dtype(), DType::F32);
            },
        );
    }
}
