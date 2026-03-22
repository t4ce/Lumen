// src/layers/linear.rs
use crate::autograd::{Tensor, is_no_grad};
use crate::init::{tensor_init, InitType}; // 引入 Init
use crate::module::Module;
use ndarray::{Axis, Ix1, IxDyn, ArrayD};
use crate::ops::matmul::{matmul, matvec_rowmajor_parallel};

pub struct Linear {
    pub weight: Tensor,       // shape: [out_features, in_features]
    pub bias: Option<Tensor>, // shape: [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // 注意：为了对齐 PyTorch/HF nn.Linear.weight 的布局，weight 存成 [out, in]
        let weight = tensor_init(vec![out_features, in_features], InitType::KaimingNormal);

        let bias = tensor_init(vec![out_features], InitType::Zeros);

        Linear {
            weight,
            bias: Some(bias),
            in_features,
            out_features,
        }
    }

    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let weight = tensor_init(vec![out_features, in_features], InitType::KaimingNormal);

        Linear { weight, bias: None, in_features, out_features }
    }


    #[inline]
    pub fn forward_decode_slice_no_bias_into(&self, input: &[f32], out: &mut [f32]) {
        assert!(is_no_grad(), "forward_decode_slice_no_bias_into is inference-only");
        assert!(self.bias.is_none(), "forward_decode_slice_no_bias_into currently expects no bias");
        assert_eq!(input.len(), self.in_features, "input width mismatch");
        assert_eq!(out.len(), self.out_features, "output width mismatch");

        let weight_guard = self.weight.data_ref();
        let weight2 = weight_guard
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Linear weight must be 2D [out,in]");

        let weight_owned;
        let weight_slice: &[f32] = if let Some(s) = weight2.as_slice() {
            s
        } else {
            weight_owned = weight2.iter().copied().collect::<Vec<f32>>();
            weight_owned.as_slice()
        };

        matvec_rowmajor_parallel(input, weight_slice, self.out_features, self.in_features, out);
    }

    pub fn forward_decode_slice_no_bias(&self, input: &[f32]) -> Tensor {
        assert!(is_no_grad(), "forward_decode_slice_no_bias is inference-only");
        assert!(self.bias.is_none(), "forward_decode_slice_no_bias currently expects no bias");
        assert_eq!(input.len(), self.in_features, "input width mismatch");

        let mut data = ArrayD::<f32>::zeros(IxDyn(&[1, 1, self.out_features])).into_shared();
        let out_slice = data
            .as_slice_mut()
            .expect("decode linear output should be contiguous");
        self.forward_decode_slice_no_bias_into(input, out_slice);
        Tensor::from_data_no_grad(data)
    }
}

impl Module for Linear {
    fn forward(&self, input: Tensor) -> Tensor {
        let y = matmul(&input, &self.weight);

        if let Some(bias) = &self.bias {
            if is_no_grad() {
                let bias_guard = bias.data_ref();
                let bias_1d = bias_guard
                    .view()
                    .into_dimensionality::<Ix1>()
                    .expect("Linear bias must be 1D [out]");
                let bias_owned;
                let bias_slice: &[f32] = if let Some(s) = bias_1d.as_slice() {
                    s
                } else {
                    bias_owned = bias_1d.iter().copied().collect::<Vec<f32>>();
                    bias_owned.as_slice()
                };

                {
                    let mut y_data = y.data_mut();
                    let last_axis = Axis(y_data.ndim() - 1);
                    for mut lane in y_data.lanes_mut(last_axis) {
                        for (dst, &b) in lane.iter_mut().zip(bias_slice.iter()) {
                            *dst += b;
                        }
                    }
                }
                y
            } else {
                y + bias.clone()
            }
        } else {
            y
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}
