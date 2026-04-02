// src/ops/arithmetic.rs
use crate::autograd::{StoragePreference, Tensor, TensorData, TensorStorageView, is_no_grad};
use crate::precision::DType;
use ndarray::{ArrayD, ArrayViewD, Zip};
use std::cell::RefCell;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

fn apply_binary_views(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    op: BinaryOp,
) -> ArrayD<f32> {
    match op {
        BinaryOp::Add => (&lhs + &rhs).into_dyn(),
        BinaryOp::Sub => (&lhs - &rhs).into_dyn(),
        BinaryOp::Mul => (&lhs * &rhs).into_dyn(),
    }
}

fn binary_no_grad(lhs: &Tensor, rhs: &Tensor, op: BinaryOp) -> Tensor {
    let output_dtype = if lhs.dtype() == rhs.dtype() {
        lhs.dtype()
    } else {
        DType::F32
    };

    lhs.with_storage_view_preferring(StoragePreference::F32Compute, |lhs_view| {
        rhs.with_storage_view_preferring(StoragePreference::F32Compute, |rhs_view| {
            let lhs_f32 = match lhs_view {
                TensorStorageView::F32(view) => view,
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            };
            let rhs_f32 = match rhs_view {
                TensorStorageView::F32(view) => view,
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            };

            Tensor::from_f32_data_no_grad_with_dtype(
                apply_binary_views(lhs_f32, rhs_f32, op),
                output_dtype,
            )
        })
    })
}

fn reduce_gradient(grad: ArrayViewD<'_, f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if grad.shape() == target_shape {
        return grad.to_owned().into_dyn();
    }

    let mut res = grad.to_owned().into_dyn();
    let g_ndim = res.ndim();
    let t_ndim = target_shape.len();

    if g_ndim > t_ndim {
        for _ in 0..(g_ndim - t_ndim) {
            res = res.sum_axis(ndarray::Axis(0));
        }
    }

    for i in 0..res.ndim() {
        if target_shape[i] == 1 && res.shape()[i] > 1 {
            let summed = res.sum_axis(ndarray::Axis(i));
            res = summed.insert_axis(ndarray::Axis(i));
        } else if target_shape[i] != res.shape()[i] {
            panic!(
                "Gradient shape mismatch. Grad: {:?}, Target: {:?}",
                grad.shape(),
                target_shape
            );
        }
    }

    if res.shape() != target_shape {
        if res.len() == target_shape.iter().product::<usize>() {
            return res.into_shape(target_shape).unwrap();
        }
        panic!("Reduction failed.");
    }

    res
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());

        if !build_graph {
            return binary_no_grad(&self, &rhs, BinaryOp::Add);
        }

        let data = (&*self.data_ref() + &*rhs.data_ref()).into_dyn();

        let lhs = self.clone();
        let rhs = rhs.clone();

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
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let l_shape = lhs.shape_vec();
                let r_shape = rhs.shape_vec();
                lhs.add_grad(reduce_gradient(grad.view(), &l_shape));
                rhs.add_grad(reduce_gradient(grad.view(), &r_shape));
            })),
            requires_grad: true,
        })))
    }
}
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        self.clone() + rhs.clone()
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());

        if !build_graph {
            return binary_no_grad(&self, &rhs, BinaryOp::Sub);
        }

        let data = (&*self.data_ref() - &*rhs.data_ref()).into_dyn();

        let lhs = self.clone();
        let rhs = rhs.clone();

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
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let l_shape = lhs.shape_vec();
                let r_shape = rhs.shape_vec();
                lhs.add_grad(reduce_gradient(grad.view(), &l_shape));

                let grad_neg = Zip::from(grad).par_map_collect(|&x| -x);
                rhs.add_grad(reduce_gradient(grad_neg.view(), &r_shape));
            })),
            requires_grad: true,
        })))
    }
}
impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        self.clone() - rhs.clone()
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());

        if !build_graph {
            return binary_no_grad(&self, &rhs, BinaryOp::Mul);
        }

        let data = (&*self.data_ref() * &*rhs.data_ref()).into_dyn();

        let lhs = self.clone();
        let rhs = rhs.clone();

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
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let (g_lhs, g_rhs, l_shape, r_shape) = {
                    let a_data = lhs.data_ref();
                    let b_data = rhs.data_ref();

                    let (g_lhs, g_rhs) =
                        if grad.shape() == a_data.shape() && grad.shape() == b_data.shape() {
                            let gl = Zip::from(grad)
                                .and(&*b_data)
                                .par_map_collect(|&g, &b| g * b);
                            let gr = Zip::from(grad)
                                .and(&*a_data)
                                .par_map_collect(|&g, &a| g * a);
                            (gl, gr)
                        } else {
                            (grad * &*b_data, grad * &*a_data)
                        };

                    (
                        g_lhs,
                        g_rhs,
                        a_data.shape().to_vec(),
                        b_data.shape().to_vec(),
                    )
                };
                lhs.add_grad(reduce_gradient(g_lhs.view(), &l_shape));
                rhs.add_grad(reduce_gradient(g_rhs.view(), &r_shape));
            })),
            requires_grad: true,
        })))
    }
}
impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        self.clone() * rhs.clone()
    }
}

pub fn sum(input: &Tensor) -> Tensor {
    let build_graph = !is_no_grad() && input.requires_grad();

    if !build_graph {
        let sum_val =
            input.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
                TensorStorageView::F32(view) => view.sum(),
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            });
        let result = ndarray::arr0(sum_val).into_dyn();
        return Tensor::from_array_no_grad(result);
    }

    let sum_val = input.data_ref().sum();
    let result = ndarray::arr0(sum_val).into_dyn();

    let input_clone = input.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
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
            let g = grad.first().copied().unwrap_or(0.0);
            let input_shape = input_clone.shape_vec();
            let grad_input = ndarray::ArrayD::from_elem(input_shape, g);
            input_clone.add_grad(grad_input);
        })),
        requires_grad: true,
    })))
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
    fn bf16_add_no_grad_preserves_dtype_and_inputs() {
        let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::BF16);
        let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::BF16);

        let out = no_grad(|| lhs.clone() + rhs.clone());

        assert_eq!(lhs.dtype(), DType::BF16);
        assert_eq!(rhs.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                let expected = vec![bf16::from_f32(1.5).to_f32(), bf16::from_f32(1.0).to_f32()];
                assert_eq!(vals, expected);
            }
            TensorStorageView::F16(_) => panic!("bf16 add output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 add output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn mixed_add_no_grad_promotes_to_f32_without_mutating_bf16_input() {
        let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::BF16);
        let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::F32);

        let out = no_grad(|| lhs.clone() + rhs.clone());

        assert_eq!(lhs.dtype(), DType::BF16);
        assert_eq!(rhs.dtype(), DType::F32);
        assert_eq!(out.dtype(), DType::F32);
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(vals, vec![1.5, 1.0]);
    }

    #[test]
    fn bf16_mul_no_grad_preserves_dtype() {
        let lhs = make_tensor(&[2], vec![2.0, -1.5], DType::BF16);
        let rhs = make_tensor(&[2], vec![0.25, 2.0], DType::BF16);

        let out = no_grad(|| lhs * rhs);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn bf16_sum_no_grad_keeps_input_dtype() {
        let input = make_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::BF16);
        let out = no_grad(|| sum(&input));
        assert_eq!(input.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.data_ref().first().copied(), Some(10.0));
    }

    #[test]
    fn i8_add_no_grad_preserves_dtype() {
        let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::I8);
        let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::I8);

        let out = no_grad(|| lhs.clone() + rhs.clone());

        assert_eq!(lhs.dtype(), DType::I8);
        assert_eq!(rhs.dtype(), DType::I8);
        assert_eq!(out.dtype(), DType::I8);
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert!((vals[0] - 1.5).abs() <= 0.02);
        assert!((vals[1] - 1.0).abs() <= 0.02);
    }

    #[test]
    fn i8_mul_no_grad_preserves_dtype() {
        let lhs = make_tensor(&[2], vec![2.0, -1.5], DType::I8);
        let rhs = make_tensor(&[2], vec![0.25, 2.0], DType::I8);

        let out = no_grad(|| lhs * rhs);
        assert_eq!(out.dtype(), DType::I8);
    }
}
