use crate::autograd::{is_no_grad, Tensor, TensorData};
use crate::init::{tensor_init, InitType};
use crate::module::Module;
use ndarray::{Array, Zip};
use std::cell::RefCell;
use std::ops::AddAssign;
use std::rc::Rc;

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let weight = tensor_init(vec![vocab_size, embed_dim], InitType::KaimingNormal);
        Self { weight, vocab_size, embed_dim }
    }

    pub fn forward(&self, indices: &Tensor) -> Tensor {
        let w_data = self.weight.data_arc();
        let idx_data = indices.data_arc();
        let e_dim = self.embed_dim;
        let v_size = self.vocab_size;

        let mut out_shape = idx_data.shape().to_vec();
        out_shape.push(e_dim);
        let mut out = Array::zeros(out_shape);

        let num_elements = idx_data.len();
        let idx_flat = idx_data
            .view()
            .into_shape(num_elements)
            .expect("Flatten indices failed");
        let mut out_flat = out
            .view_mut()
            .into_shape((num_elements, e_dim))
            .expect("Flatten output failed");

        let w_2d = w_data
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Embedding weight must be 2D");

        Zip::from(out_flat.outer_iter_mut())
            .and(&idx_flat)
            .par_for_each(|mut out_row, &idx_f32| {
                let idx = idx_f32 as usize;
                if idx < v_size {
                    let w_row = w_2d.slice(ndarray::s![idx, ..]);
                    out_row.assign(&w_row);
                } else {
                    panic!("Embedding index out of bounds: {} >= {}", idx, v_size);
                }
            });

        let out_dyn = out.into_dyn();
        let build_graph = !is_no_grad() && self.weight.requires_grad();

        if !build_graph {
            return Tensor::from_array_no_grad(out_dyn);
        }

        let indices_clone = indices.clone();
        let w_clone = self.weight.clone();
        let v_snap = v_size;
        let e_snap = e_dim;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: out_dyn.into_shared(),
            grad: None,
            parents: vec![indices.clone(), self.weight.clone()],
            backward_op: Some(Box::new(move |grad| {
                let binding = indices_clone.data_ref();
                let idx_flat = binding.view().into_shape(num_elements).unwrap();
                let grad_2d = grad.view().into_shape((num_elements, e_snap)).unwrap();

                let mut d_w = Array::zeros((v_snap, e_snap));
                for (i, &idx_f32) in idx_flat.iter().enumerate() {
                    let idx = idx_f32 as usize;
                    if idx < v_snap {
                        d_w.slice_mut(ndarray::s![idx, ..])
                            .add_assign(&grad_2d.slice(ndarray::s![i, ..]));
                    }
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
