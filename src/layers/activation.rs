use crate::autograd::{StoragePreference, Tensor, TensorData, TensorStorageView, is_no_grad};
use crate::module::Module;
use crate::precision::DType;
use ndarray::{Array2, Zip};
use std::cell::RefCell;
use std::rc::Rc;

fn unary_no_grad(input: &Tensor, op: impl Fn(f32) -> f32 + Copy + Send + Sync) -> Tensor {
    let output_dtype = input.dtype();
    input.with_storage_view_preferring(StoragePreference::F32Compute, |input_view| {
        let input_f32 = match input_view {
            TensorStorageView::F32(view) => view,
            TensorStorageView::F16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
            TensorStorageView::BF16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
        };
        let data = Zip::from(&input_f32).par_map_collect(|&x| op(x)).into_dyn();
        Tensor::from_f32_data_no_grad_with_dtype(data, output_dtype)
    })
}

fn softmax_no_grad(input: &Tensor, axis: usize) -> Tensor {
    let output_dtype = if input.dtype().is_float() {
        input.dtype()
    } else {
        DType::F32
    };

    input.with_storage_view_preferring(StoragePreference::F32Compute, |input_view| {
        let input_view = match input_view {
            TensorStorageView::F32(view) => view,
            TensorStorageView::F16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
            TensorStorageView::BF16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
        };
        let shape = input_view.shape().to_vec();
        assert!(!shape.is_empty(), "Softmax expects at least 1D input");
        assert_eq!(
            axis,
            shape.len() - 1,
            "Softmax currently only supports the last dimension in this implementation"
        );
        let last_dim = shape[axis];
        let outer_dim = shape.iter().product::<usize>() / last_dim;

        let x_cow = input_view.as_standard_layout();
        let x_2d = x_cow.view().into_shape((outer_dim, last_dim)).unwrap();
        let mut y_flat = Array2::<f32>::zeros((outer_dim, last_dim));
        Zip::from(y_flat.outer_iter_mut())
            .and(x_2d.outer_iter())
            .par_for_each(|mut y_row, x_row| {
                let max_val = x_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;

                for (y_val, &x_val) in y_row.iter_mut().zip(x_row.iter()) {
                    let e = (x_val - max_val).exp();
                    *y_val = e;
                    sum += e;
                }

                let inv_sum = 1.0 / sum;
                for y_val in y_row.iter_mut() {
                    *y_val *= inv_sum;
                }
            });

        Tensor::from_f32_data_no_grad_with_dtype(
            y_flat.into_shape(shape).unwrap().into_dyn(),
            output_dtype,
        )
    })
}

pub struct ReLU;
impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && input.requires_grad();

        if !build_graph {
            return unary_no_grad(&input, |x| x.max(0.0));
        }

        let data = {
            let input_ref = input.data_ref();
            Zip::from(&*input_ref).par_map_collect(|&x| x.max(0.0))
        };

        let input_clone = input.clone();
        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let grad_input = {
                    let input_d = input_clone.data_ref();
                    let mut grad_input = grad.to_owned().into_dyn();
                    Zip::from(grad_input.view_mut())
                        .and(&*input_d)
                        .par_for_each(|g, &x| {
                            if x <= 0.0 {
                                *g = 0.0;
                            }
                        });
                    grad_input
                };
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && input.requires_grad();

        if !build_graph {
            return unary_no_grad(&input, |x| 1.0 / (1.0 + (-x).exp()));
        }

        let data = {
            let input_ref = input.data_ref();
            Zip::from(&*input_ref).par_map_collect(|&x| 1.0 / (1.0 + (-x).exp()))
        };

        let output_data = data.clone();
        let input_clone = input.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let mut grad_input = grad.to_owned().into_dyn();
                Zip::from(grad_input.view_mut())
                    .and(&output_data)
                    .par_for_each(|g, &y| {
                        *g = *g * y * (1.0 - y);
                    });
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Tanh;
impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Module for Tanh {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && input.requires_grad();

        if !build_graph {
            return unary_no_grad(&input, |x| x.tanh());
        }

        let data = {
            let input_ref = input.data_ref();
            Zip::from(&*input_ref).par_map_collect(|&x| x.tanh())
        };

        let output_data = data.clone();
        let input_clone = input.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let mut grad_input = grad.to_owned().into_dyn();
                Zip::from(grad_input.view_mut())
                    .and(&output_data)
                    .par_for_each(|g, &y| {
                        *g = *g * (1.0 - y * y);
                    });
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct SiLU;
impl SiLU {
    pub fn new() -> Self {
        SiLU
    }
}

impl Module for SiLU {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && input.requires_grad();

        if !build_graph {
            return unary_no_grad(&input, |x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            });
        }

        let data = {
            let input_ref = input.data_ref();
            Zip::from(&*input_ref).par_map_collect(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
        };

        let input_clone = input.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let grad_input = {
                    let x_ref = input_clone.data_ref();
                    let dx = Zip::from(&*x_ref).par_map_collect(|&x| {
                        let sig = 1.0 / (1.0 + (-x).exp());
                        sig + x * sig * (1.0 - sig)
                    });
                    (&dx * grad).into_dyn()
                };
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Softmax {
    axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Self {
        Softmax { axis }
    }
}

impl Module for Softmax {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && input.requires_grad();

        if !build_graph {
            return softmax_no_grad(&input, self.axis);
        }

        let (y, output_data) = {
            let input_ref = input.data_ref();
            let x = &*input_ref;
            let shape = x.shape();
            assert!(!shape.is_empty(), "Softmax expects at least 1D input");
            assert_eq!(
                self.axis,
                shape.len() - 1,
                "Softmax currently only supports the last dimension in this implementation"
            );

            let axis = self.axis;

            let last_dim = shape[axis];
            let outer_dim = x.len() / last_dim;
            let x_cow = x.as_standard_layout();
            let x_2d = x_cow.view().into_shape((outer_dim, last_dim)).unwrap();

            let mut y_flat = Array2::<f32>::zeros((outer_dim, last_dim));
            Zip::from(y_flat.outer_iter_mut())
                .and(x_2d.outer_iter())
                .par_for_each(|mut y_row, x_row| {
                    let max_val = x_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum = 0.0f32;

                    for (y_val, &x_val) in y_row.iter_mut().zip(x_row.iter()) {
                        let e = (x_val - max_val).exp();
                        *y_val = e;
                        sum += e;
                    }

                    let inv_sum = 1.0 / sum;
                    for y_val in y_row.iter_mut() {
                        *y_val *= inv_sum;
                    }
                });

            let y = y_flat.into_shape(shape).unwrap();
            (y.clone(), y)
        };

        let input_clone = input.clone();
        let axis_idx = self.axis;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: y.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let grad_shape = grad_output.shape();
                let dim = grad_shape[axis_idx];
                let outer = grad_output.len() / dim;
                let g_cow = grad_output.as_standard_layout();
                let g_2d = g_cow.view().into_shape((outer, dim)).unwrap();

                let y_cow = output_data.as_standard_layout();
                let y_2d = y_cow.view().into_shape((outer, dim)).unwrap();

                let mut d_input_flat = Array2::<f32>::zeros((outer, dim));

                Zip::from(d_input_flat.outer_iter_mut())
                    .and(y_2d.outer_iter())
                    .and(g_2d.outer_iter())
                    .par_for_each(|mut di_row, y_row, g_row| {
                        let dot: f32 = y_row.iter().zip(g_row.iter()).map(|(&y, &g)| y * g).sum();

                        for (di, (&y, &g)) in di_row.iter_mut().zip(y_row.iter().zip(g_row.iter()))
                        {
                            *di = y * (g - dot);
                        }
                    });

                let d_input = d_input_flat.into_shape(grad_shape).unwrap();
                input_clone.add_grad(d_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Gelu;
impl Gelu {
    pub fn new() -> Self {
        Gelu
    }
}

impl Module for Gelu {
    fn forward(&self, input: Tensor) -> Tensor {
        const C: f32 = 0.7978845608;
        const K: f32 = 0.044715;

        let build_graph = !is_no_grad() && input.requires_grad();

        if !build_graph {
            return unary_no_grad(&input, |x| {
                let x3 = x * x * x;
                0.5 * x * (1.0 + (C * (x + K * x3)).tanh())
            });
        }

        let output = {
            let input_ref = input.data_ref();
            Zip::from(&*input_ref).par_map_collect(|&x| {
                let x3 = x * x * x;
                0.5 * x * (1.0 + (C * (x + K * x3)).tanh())
            })
        };

        let input_clone = input.clone();
        Tensor(Rc::new(RefCell::new(TensorData {
            data: output.into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let grad_input = {
                    let x_ref = input_clone.data_ref();
                    let dx = Zip::from(&*x_ref).par_map_collect(|&x| {
                        let x3 = x * x * x;
                        let inner = C * (x + K * x3);
                        let tanh_i = inner.tanh();
                        let sech2 = 1.0 - tanh_i * tanh_i;
                        0.5 * (1.0 + tanh_i) + 0.5 * x * sech2 * C * (1.0 + 3.0 * K * x * x)
                    });
                    (&dx * grad).into_dyn()
                };
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    use crate::precision::DType;
    use half::bf16;
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
    fn relu_no_grad_preserves_bf16_dtype() {
        let input = make_tensor(&[3], vec![-1.0, 0.5, 2.0], DType::BF16);
        let out = no_grad(|| ReLU::new().forward(input.clone()));

        assert_eq!(input.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn softmax_no_grad_preserves_bf16_dtype() {
        let input = make_tensor(&[1, 4], vec![1.0, 2.0, 3.0, 4.0], DType::BF16);
        let ref_out = no_grad(|| {
            Softmax::new(1).forward(make_tensor(&[1, 4], vec![1.0, 2.0, 3.0, 4.0], DType::F32))
        });
        let out = no_grad(|| Softmax::new(1).forward(input.clone()));

        assert_eq!(input.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);
        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => panic!("bf16 softmax output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 softmax output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn relu_no_grad_preserves_i8_dtype() {
        let input = make_tensor(&[3], vec![-1.0, 0.5, 2.0], DType::I8);
        let out = no_grad(|| ReLU::new().forward(input.clone()));

        assert_eq!(input.dtype(), DType::I8);
        assert_eq!(out.dtype(), DType::I8);
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert!(vals[0].abs() <= 0.02);
        assert!((vals[1] - 0.5).abs() <= 0.02);
        assert!((vals[2] - 2.0).abs() <= 0.02);
    }

    #[test]
    fn softmax_no_grad_promotes_i8_to_f32() {
        let input = make_tensor(&[1, 4], vec![1.0, 2.0, 3.0, 4.0], DType::I8);
        let out = no_grad(|| Softmax::new(1).forward(input.clone()));

        assert_eq!(input.dtype(), DType::I8);
        assert_eq!(out.dtype(), DType::F32);
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-5);
    }

    #[test]
    #[should_panic(expected = "only supports the last dimension")]
    fn softmax_rejects_non_last_axis() {
        let input = make_tensor(&[2, 3], (0..6).map(|v| v as f32).collect(), DType::F32);
        no_grad(|| {
            let _ = Softmax::new(0).forward(input);
        });
    }
}
