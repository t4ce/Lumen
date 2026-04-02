use crate::autograd::{StoragePreference, Tensor, TensorData, TensorStorageView, is_no_grad};
use crate::module::Module;
use crate::precision::DType;
use half::{bf16, f16};
use ndarray::{Array1, Array2, Zip};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let data = Array1::ones(dim).into_dyn();
        Self {
            weight: Tensor::parameter(data),
            eps,
        }
    }

    pub fn new_with_dtype(dim: usize, eps: f32, dtype: DType) -> Self {
        let data = Array1::ones(dim).into_dyn();
        Self {
            weight: Tensor::parameter_with_dtype(data, dtype),
            eps,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && (input.requires_grad() || self.weight.requires_grad());

        if !build_graph {
            return input.with_storage_view_preferring(StoragePreference::Native, |input_view| {
                self.weight.with_storage_view(|weight_view| {
                    let shape = match &input_view {
                        TensorStorageView::F32(input_view) => input_view.shape().to_vec(),
                        TensorStorageView::F16(input_view) => input_view.shape().to_vec(),
                        TensorStorageView::BF16(input_view) => input_view.shape().to_vec(),
                    };
                    let dim = shape[shape.len() - 1];
                    let rows = shape.iter().product::<usize>() / dim;
                    let eps = self.eps;

                    match input_view {
                        TensorStorageView::F32(input_view) => {
                            let x_cow = input_view.as_standard_layout();
                            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                            let mut output_flat = Array2::<f32>::zeros((rows, dim));

                            match weight_view {
                                TensorStorageView::F32(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");
                                    let w_slice = w_1d
                                        .as_slice()
                                        .expect("RMSNorm weight should be contiguous");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq =
                                                x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in
                                                out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                            {
                                                *o = xi * inv_rms * wi;
                                            }
                                        });
                                }
                                TensorStorageView::F16(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq =
                                                x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = xi * inv_rms * wi.to_f32();
                                            }
                                        });
                                }
                                TensorStorageView::BF16(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq =
                                                x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = xi * inv_rms * wi.to_f32();
                                            }
                                        });
                                }
                            }

                            Tensor::from_array_no_grad(
                                output_flat.into_shape(shape).unwrap().into_dyn(),
                            )
                        }
                        TensorStorageView::F16(input_view) => {
                            let x_cow = input_view.as_standard_layout();
                            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                            let mut output_flat =
                                Array2::<f16>::from_elem((rows, dim), f16::from_bits(0));

                            match weight_view {
                                TensorStorageView::F32(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");
                                    let w_slice = w_1d
                                        .as_slice()
                                        .expect("RMSNorm weight should be contiguous");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in
                                                out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                            {
                                                *o = f16::from_f32(xi.to_f32() * inv_rms * wi);
                                            }
                                        });
                                }
                                TensorStorageView::F16(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = f16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                                TensorStorageView::BF16(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = f16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                            }

                            Tensor::from_f16_data_no_grad(
                                output_flat
                                    .into_shape(shape)
                                    .unwrap()
                                    .into_dyn()
                                    .into_shared(),
                            )
                        }
                        TensorStorageView::BF16(input_view) => {
                            let x_cow = input_view.as_standard_layout();
                            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                            let mut output_flat =
                                Array2::<bf16>::from_elem((rows, dim), bf16::from_bits(0));

                            match weight_view {
                                TensorStorageView::F32(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");
                                    let w_slice = w_1d
                                        .as_slice()
                                        .expect("RMSNorm weight should be contiguous");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in
                                                out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                            {
                                                *o = bf16::from_f32(xi.to_f32() * inv_rms * wi);
                                            }
                                        });
                                }
                                TensorStorageView::F16(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = bf16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                                TensorStorageView::BF16(weight_view) => {
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = bf16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                            }

                            Tensor::from_bf16_data_no_grad(
                                output_flat
                                    .into_shape(shape)
                                    .unwrap()
                                    .into_dyn()
                                    .into_shared(),
                            )
                        }
                    }
                })
            });
        }

        let (output_data, rows, dim, shape) = self.weight.with_storage_view(|weight_view| {
            let input_ref = input.data_ref();
            let x = &*input_ref;

            let shape = x.shape().to_vec();
            let dim = shape[shape.len() - 1];
            let rows = x.len() / dim;

            let x_cow = x.as_standard_layout();
            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
            let mut output_flat = Array2::<f32>::zeros((rows, dim));
            let eps = self.eps;

            match weight_view {
                TensorStorageView::F32(weight_view) => {
                    let w_1d = weight_view
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("RMSNorm weight must be 1D");
                    let w_slice = w_1d
                        .as_slice()
                        .expect("RMSNorm weight should be contiguous");

                    Zip::from(output_flat.outer_iter_mut())
                        .and(x_2d.outer_iter())
                        .par_for_each(|mut out_row, x_row| {
                            let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                            let inv_rms = 1.0 / rms;

                            for (o, (&xi, &wi)) in out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                            {
                                *o = xi * inv_rms * wi;
                            }
                        });
                }
                TensorStorageView::F16(weight_view) => {
                    let w_1d = weight_view
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("RMSNorm weight must be 1D");

                    Zip::from(output_flat.outer_iter_mut())
                        .and(x_2d.outer_iter())
                        .par_for_each(|mut out_row, x_row| {
                            let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                            let inv_rms = 1.0 / rms;

                            for (o, (&xi, &wi)) in
                                out_row.iter_mut().zip(x_row.iter().zip(w_1d.iter()))
                            {
                                *o = xi * inv_rms * wi.to_f32();
                            }
                        });
                }
                TensorStorageView::BF16(weight_view) => {
                    let w_1d = weight_view
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("RMSNorm weight must be 1D");

                    Zip::from(output_flat.outer_iter_mut())
                        .and(x_2d.outer_iter())
                        .par_for_each(|mut out_row, x_row| {
                            let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                            let inv_rms = 1.0 / rms;

                            for (o, (&xi, &wi)) in
                                out_row.iter_mut().zip(x_row.iter().zip(w_1d.iter()))
                            {
                                *o = xi * inv_rms * wi.to_f32();
                            }
                        });
                }
            }

            let out_d = output_flat.into_shape(shape.clone()).unwrap().into_dyn();
            (out_d, rows, dim, shape)
        });

        let input_clone = input.clone();
        let weight_clone = self.weight.clone();
        let eps = self.eps;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: output_data.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone(), self.weight.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let (d_input, d_weight) = {
                    let x_ref = input_clone.data_ref();
                    let x_cow = x_ref.as_standard_layout();
                    let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();

                    let w_ref = weight_clone.data_ref();
                    let w_slice = w_ref.as_slice().unwrap();

                    let g_cow = grad_output.as_standard_layout();
                    let g_2d = g_cow.view().into_shape((rows, dim)).unwrap();

                    let mut d_input_flat = Array2::<f32>::zeros((rows, dim));
                    Zip::from(d_input_flat.outer_iter_mut())
                        .and(x_2d.outer_iter())
                        .and(g_2d.outer_iter())
                        .par_for_each(|mut dx_row, x_row, g_row| {
                            let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                            let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
                            let inv_dim = 1.0 / dim as f32;

                            let mut dot = 0.0f32;
                            for (&gi, (&wi, &xi)) in
                                g_row.iter().zip(w_slice.iter().zip(x_row.iter()))
                            {
                                dot += (gi * wi) * (xi * inv_rms);
                            }

                            let mean_dot = dot * inv_dim;

                            for (dxi, (&gi, (&wi, &xi))) in dx_row
                                .iter_mut()
                                .zip(g_row.iter().zip(w_slice.iter().zip(x_row.iter())))
                            {
                                let term1 = gi * wi;
                                let x_norm = xi * inv_rms;
                                *dxi = inv_rms * (term1 - x_norm * mean_dot);
                            }
                        });

                    let dw_accum = (0..rows)
                        .into_par_iter()
                        .fold(
                            || Array1::<f32>::zeros(dim),
                            |mut acc, r| {
                                let x_row = x_2d.row(r);
                                let g_row = g_2d.row(r);

                                let sum_sq = x_row.fold(0.0f32, |a, &v| a + v * v);
                                let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

                                for (a, (&gi, &xi)) in
                                    acc.iter_mut().zip(g_row.iter().zip(x_row.iter()))
                                {
                                    *a += gi * xi * inv_rms;
                                }
                                acc
                            },
                        )
                        .reduce(
                            || Array1::<f32>::zeros(dim),
                            |mut a, b| {
                                a += &b;
                                a
                            },
                        );

                    (
                        d_input_flat.into_shape(shape.clone()).unwrap().into_dyn(),
                        dw_accum.into_dyn(),
                    )
                };

                input_clone.add_grad(d_input);
                weight_clone.add_grad(d_weight);
            })),
            requires_grad: true,
        })))
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
    fn rms_norm_no_grad_preserves_bf16_input_dtype() {
        let norm = RMSNorm::new(4, 1e-5);
        norm.weight.set_array_f32_with_dtype(
            Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
                .expect("weight shape mismatch")
                .into_dyn(),
            DType::BF16,
        );

        let input_f32 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::F32);
        let input_bf16 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::BF16);

        let ref_out = no_grad(|| norm.forward(input_f32));
        let bf16_out = no_grad(|| norm.forward(input_bf16));

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
            TensorStorageView::F16(_) => panic!("bf16 RMSNorm output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 RMSNorm output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn rms_norm_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::BF16,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies: false,
            },
            || {
                let norm = RMSNorm::new_with_dtype(4, 1e-5, DType::F32);
                assert_eq!(norm.weight.dtype(), DType::F32);
            },
        );
    }
}
