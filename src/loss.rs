use crate::autograd::{StoragePreference, Tensor, TensorData, TensorStorageView, is_no_grad};
use crate::precision::DType;
use ndarray::{Array2, Zip, arr0};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc; // 引入并行迭代

// --- MSE Loss ---
pub struct MSELoss;
impl MSELoss {
    pub fn apply(output: &Tensor, target: &Tensor) -> Tensor {
        let build_graph = !is_no_grad() && (output.requires_grad() || target.requires_grad());
        let loss_val =
            output.with_storage_view_preferring(StoragePreference::F32Compute, |out_view| {
                target.with_storage_view_preferring(StoragePreference::F32Compute, |tar_view| {
                    let out_ref = match out_view {
                        TensorStorageView::F32(view) => view,
                        TensorStorageView::F16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                        TensorStorageView::BF16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                    };
                    let tar_ref = match tar_view {
                        TensorStorageView::F32(view) => view,
                        TensorStorageView::F16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                        TensorStorageView::BF16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                    };
                    let n = out_ref.len() as f32;
                    let sum_sq: f32 = Zip::from(&out_ref)
                        .and(&tar_ref)
                        .par_map_collect(|&o, &t| (o - t).powi(2))
                        .sum();
                    sum_sq / n
                })
            });

        if !build_graph {
            return Tensor::from_f32_data_no_grad_with_dtype(arr0(loss_val).into_dyn(), DType::F32);
        }

        let output_clone = output.clone();
        let target_clone = target.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: arr0(loss_val).into_dyn().into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![output.clone(), target.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let grad_val = grad_output.first().unwrap();
                let (grad_out, grad_target) = {
                    let out_d = output_clone.data_ref();
                    let tar_d = target_clone.data_ref();
                    let n = out_d.len() as f32;
                    let factor = 2.0 / n * grad_val;

                    let grad = Zip::from(&*out_d)
                        .and(&*tar_d)
                        .par_map_collect(|&o, &t| (o - t) * factor);
                    (grad.clone(), grad.mapv(|x| -x))
                };

                output_clone.add_grad(grad_out);
                target_clone.add_grad(grad_target);
            })),
            requires_grad: true,
        })))
    }
}

// --- Cross Entropy Loss ---
// 针对 Batch 进行行级并行优化
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn apply(input_logits: &Tensor, target_onehot: &Tensor) -> Tensor {
        let build_graph =
            !is_no_grad() && (input_logits.requires_grad() || target_onehot.requires_grad());
        // Forward
        let (loss_val, softmax_output) = input_logits.with_storage_view_preferring(
            StoragePreference::F32Compute,
            |logits_view| {
                target_onehot.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |targets_view| {
                        let logits_ref = match logits_view {
                            TensorStorageView::F32(view) => view,
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let targets_ref = match targets_view {
                            TensorStorageView::F32(view) => view,
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let batch_size = logits_ref.shape()[0];
                        let dim = logits_ref.shape()[1];

                        let logits_2d = logits_ref.view().into_shape((batch_size, dim)).unwrap();
                        let targets_2d = targets_ref.view().into_shape((batch_size, dim)).unwrap();

                        let mut softmax_out_flat = Array2::<f32>::zeros((batch_size, dim));
                        let total_loss: f32 = Zip::from(softmax_out_flat.outer_iter_mut())
                            .and(logits_2d.outer_iter())
                            .and(targets_2d.outer_iter())
                            .into_par_iter()
                            .map(|(mut sm_row, l_row, t_row)| {
                                let max_val = l_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let mut sum_exp = 0.0f32;

                                for (s_val, &l_val) in sm_row.iter_mut().zip(l_row.iter()) {
                                    let e = (l_val - max_val).exp();
                                    *s_val = e;
                                    sum_exp += e;
                                }

                                let inv_sum = 1.0 / sum_exp;
                                let epsilon = 1e-9;
                                let mut row_loss = 0.0;

                                for (s_val, &t_val) in sm_row.iter_mut().zip(t_row.iter()) {
                                    *s_val *= inv_sum;
                                    if t_val > 0.0 {
                                        row_loss -= t_val * (*s_val + epsilon).ln();
                                    }
                                }
                                row_loss
                            })
                            .sum();

                        (total_loss / batch_size as f32, softmax_out_flat.into_dyn())
                    },
                )
            },
        );

        if !build_graph {
            return Tensor::from_f32_data_no_grad_with_dtype(arr0(loss_val).into_dyn(), DType::F32);
        }

        let input_clone = input_logits.clone();
        let target_clone = target_onehot.clone();
        let softmax_cache = softmax_output;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: arr0(loss_val).into_dyn().into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input_logits.clone(), target_onehot.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let grad_val = grad_output.first().unwrap();
                let grad = {
                    let targets_ref = target_clone.data_ref();
                    let batch_size = targets_ref.shape()[0] as f32;
                    let factor = grad_val / batch_size;

                    Zip::from(&softmax_cache)
                        .and(&*targets_ref)
                        .par_map_collect(|&p, &t| (p - t) * factor)
                };

                input_clone.add_grad(grad);
            })),
            requires_grad: true,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
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
    fn mse_loss_no_grad_returns_f32_scalar_without_materializing_bf16_input() {
        let output = make_tensor(&[2], vec![1.0, 3.0], DType::BF16);
        let target = make_tensor(&[2], vec![2.0, 1.0], DType::BF16);

        let loss = no_grad(|| MSELoss::apply(&output, &target));

        assert_eq!(loss.dtype(), DType::F32);
        assert!(!loss.requires_grad());
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(target.dtype(), DType::BF16);
        assert!((loss.data_ref().first().copied().unwrap_or_default() - 2.5).abs() <= 1e-6);
    }

    #[test]
    fn cross_entropy_no_grad_accepts_i8_logits_and_returns_f32_scalar() {
        let logits = make_tensor(&[2, 3], vec![2.0, 0.0, -1.0, -1.0, 0.0, 2.0], DType::I8);
        let target = make_tensor(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], DType::F32);

        let loss = no_grad(|| CrossEntropyLoss::apply(&logits, &target));

        assert_eq!(loss.dtype(), DType::F32);
        assert!(!loss.requires_grad());
        assert_eq!(logits.dtype(), DType::I8);
        assert!(
            loss.data_ref()
                .first()
                .copied()
                .unwrap_or_default()
                .is_finite()
        );
    }
}
