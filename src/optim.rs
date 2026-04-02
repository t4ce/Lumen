use crate::autograd::Tensor;
use crate::precision::{DType, default_runtime_dtype};
use ndarray::Zip;
use ndarray::prelude::*;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&self) {
        for param in self.params() {
            param.zero_grad();
        }
    }
    fn params(&self) -> &Vec<Tensor>;
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    state_dtype: DType,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new_with_dtype(params, lr, default_runtime_dtype())
    }

    pub fn new_with_dtype(params: Vec<Tensor>, lr: f32, state_dtype: DType) -> Self {
        assert!(
            state_dtype.is_float(),
            "Optimizer state currently only supports floating dtypes, got {:?}",
            state_dtype
        );
        let len = params.len();
        SGD {
            params,
            lr,
            momentum: 0.0, // 默认无动量
            state_dtype,
            velocities: vec![None; len],
        }
    }

    #[inline]
    pub fn state_dtype(&self) -> DType {
        self.state_dtype
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for SGD {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad_arc() {
                Some(g) => g,
                None => continue,
            };

            if self.momentum == 0.0 {
                let lr = self.lr;
                let mut data = param.data_mut();
                Zip::from(data.view_mut())
                    .and(grad.view())
                    .for_each(|w, g| {
                        *w -= lr * *g;
                    });
            } else {
                if self.velocities[i].is_none() {
                    let state =
                        Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&param.shape_vec())));
                    state.cast_inplace(self.state_dtype);
                    self.velocities[i] = Some(state);
                }

                let m = self.momentum;
                let lr = self.lr;
                let v_buf = self.velocities[i].as_ref().unwrap();
                let mut next_v = v_buf.data();

                Zip::from(next_v.view_mut())
                    .and(grad.view())
                    .for_each(|v, g| {
                        *v = m * (*v) + *g;
                    });

                let mut data = param.data_mut();
                Zip::from(data.view_mut())
                    .and(next_v.view())
                    .for_each(|w, vv| {
                        *w -= lr * *vv;
                    });
                v_buf.set_array_f32_with_dtype(next_v, self.state_dtype);
            }
        }
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,

    // 状态
    step_count: usize,
    state_dtype: DType,
    exp_avg: Vec<Option<Tensor>>,    // m (一阶矩)
    exp_avg_sq: Vec<Option<Tensor>>, // v (二阶矩)
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new_with_dtype(params, lr, default_runtime_dtype())
    }

    pub fn new_with_dtype(params: Vec<Tensor>, lr: f32, state_dtype: DType) -> Self {
        assert!(
            state_dtype.is_float(),
            "Optimizer state currently only supports floating dtypes, got {:?}",
            state_dtype
        );
        let len = params.len();
        Adam {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            step_count: 0,
            state_dtype,
            exp_avg: vec![None; len],
            exp_avg_sq: vec![None; len],
        }
    }

    #[inline]
    pub fn state_dtype(&self) -> DType {
        self.state_dtype
    }
}

impl Optimizer for Adam {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        self.step_count += 1;
        let (beta1, beta2) = self.betas;

        // 预计算 Bias Correction
        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad_arc() {
                Some(g) => g,
                None => continue,
            };

            if self.exp_avg[i].is_none() {
                let exp_avg = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&param.shape_vec())));
                exp_avg.cast_inplace(self.state_dtype);
                let exp_avg_sq =
                    Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&param.shape_vec())));
                exp_avg_sq.cast_inplace(self.state_dtype);
                self.exp_avg[i] = Some(exp_avg);
                self.exp_avg_sq[i] = Some(exp_avg_sq);
            }

            let lr = self.lr;
            let eps = self.eps;
            let m_buf = self.exp_avg[i].as_ref().unwrap();
            let v_buf = self.exp_avg_sq[i].as_ref().unwrap();
            let mut m_next = m_buf.data();
            let mut v_next = v_buf.data();
            let mut data = param.data_mut();

            Zip::from(data.view_mut())
                .and(m_next.view_mut())
                .and(v_next.view_mut())
                .and(grad.view())
                .for_each(|w, m, v, g| {
                    *m = beta1 * (*m) + (1.0 - beta1) * g;
                    *v = beta2 * (*v) + (1.0 - beta2) * g * g;
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    *w -= lr * (m_hat / (v_hat.sqrt() + eps));
                });
            m_buf.set_array_f32_with_dtype(m_next, self.state_dtype);
            v_buf.set_array_f32_with_dtype(v_next, self.state_dtype);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::{PrecisionConfig, set_default_runtime_dtype, with_precision_config};

    #[test]
    fn sgd_default_construction_captures_runtime_dtype_for_future_state() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
                let mut opt = SGD::new(vec![param.clone()], 0.1).with_momentum(0.9);
                set_default_runtime_dtype(DType::F32);
                param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.5));
                opt.step();

                assert_eq!(opt.state_dtype(), DType::BF16);
                assert_eq!(
                    opt.velocities[0].as_ref().expect("velocity state").dtype(),
                    DType::BF16
                );
            },
        );
    }

    #[test]
    fn sgd_explicit_state_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
                let mut opt =
                    SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32).with_momentum(0.9);
                param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.5));
                opt.step();

                assert_eq!(opt.state_dtype(), DType::F32);
                assert_eq!(
                    opt.velocities[0].as_ref().expect("velocity state").dtype(),
                    DType::F32
                );
            },
        );
    }

    #[test]
    fn adam_default_construction_captures_runtime_dtype_for_future_state() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
                let mut opt = Adam::new(vec![param.clone()], 0.1);
                set_default_runtime_dtype(DType::F32);
                param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.25));
                opt.step();

                assert_eq!(opt.state_dtype(), DType::BF16);
                assert_eq!(
                    opt.exp_avg[0].as_ref().expect("exp_avg").dtype(),
                    DType::BF16
                );
                assert_eq!(
                    opt.exp_avg_sq[0].as_ref().expect("exp_avg_sq").dtype(),
                    DType::BF16
                );
            },
        );
    }

    #[test]
    #[should_panic(expected = "Optimizer state currently only supports floating dtypes")]
    fn optimizer_state_rejects_integer_dtype() {
        let _ = SGD::new_with_dtype(Vec::new(), 0.1, DType::I8);
    }
}
