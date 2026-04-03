use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView, is_no_grad,
};
use crate::ops::matmul::{
    SliceRef, dual_matvec_rowmajor_parallel_mixed, dual_matvec_silu_mul_rowmajor_parallel_f32_bf16,
    dual_matvec_silu_mul_rowmajor_parallel_f32_f16, dual_matvec_silu_mul_rowmajor_parallel_f32_i8,
    dual_matvec_silu_mul_rowmajor_parallel_mixed, matvec_rowmajor_parallel_mixed,
    qkv_matvec_rowmajor_parallel, qkv_matvec_rowmajor_parallel_f32_bf16,
    qkv_matvec_rowmajor_parallel_f32_f16, qkv_matvec_rowmajor_parallel_f32_i8,
    with_bf16_input_as_f32, with_f16_input_as_f32,
};
use crate::precision::DType;
use half::{bf16, f16};
use ndarray::{Array, Array2, Array3, Axis, Ix2, Zip};
use std::cell::RefCell;
use std::rc::Rc;

#[inline]
fn slice_ref_dtype(slice: SliceRef<'_>) -> DType {
    match slice {
        SliceRef::F32(_) => DType::F32,
        SliceRef::F16(_) => DType::F16,
        SliceRef::BF16(_) => DType::BF16,
        SliceRef::I8(_, _) => DType::I8,
    }
}

fn with_decode_input_as_slice_ref<R>(input: &Tensor, f: impl FnOnce(SliceRef<'_>) -> R) -> R {
    if input.dtype() == DType::I8 {
        return match input.native_storage_owned() {
            TensorStorageOwned::I8(data, scale) => {
                if let Some(slice) = data.as_slice() {
                    f(SliceRef::I8(slice, scale))
                } else {
                    let owned = data.iter().copied().collect::<Vec<_>>();
                    f(SliceRef::I8(owned.as_slice(), scale))
                }
            }
            TensorStorageOwned::F32(_)
            | TensorStorageOwned::F16(_)
            | TensorStorageOwned::BF16(_) => {
                unreachable!("checked i8 decode input above")
            }
        };
    }

    input.with_storage_view_preferring(StoragePreference::Native, |x_view| match x_view {
        TensorStorageView::F32(x_view) => {
            if let Some(x_slice) = x_view.as_slice() {
                f(SliceRef::F32(x_slice))
            } else {
                let x_owned = x_view.iter().copied().collect::<Vec<f32>>();
                f(SliceRef::F32(x_owned.as_slice()))
            }
        }
        TensorStorageView::F16(x_view) => {
            if let Some(x_slice) = x_view.as_slice() {
                f(SliceRef::F16(x_slice))
            } else {
                let x_owned = x_view.iter().copied().collect::<Vec<_>>();
                f(SliceRef::F16(x_owned.as_slice()))
            }
        }
        TensorStorageView::BF16(x_view) => {
            if let Some(x_slice) = x_view.as_slice() {
                f(SliceRef::BF16(x_slice))
            } else {
                let x_owned = x_view.iter().copied().collect::<Vec<_>>();
                f(SliceRef::BF16(x_owned.as_slice()))
            }
        }
    })
}

fn validate_gate_up_shapes(k_dim: usize, gate_weight: &Tensor, up_weight: &Tensor) -> usize {
    assert!(k_dim > 0, "input hidden dim must be > 0");
    let gate_shape = gate_weight.shape_vec();
    let up_shape = up_weight.shape_vec();
    assert_eq!(gate_shape.len(), 2, "gate weight must be 2D [N, K]");
    assert_eq!(up_shape.len(), 2, "up weight must be 2D [N, K]");
    let n_dim = gate_shape[0];
    assert_eq!(gate_shape[1], k_dim, "gate weight K mismatch");
    assert_eq!(up_shape[0], n_dim, "gate/up out dim mismatch");
    assert_eq!(up_shape[1], k_dim, "up weight K mismatch");
    n_dim
}

fn validate_qkv_shapes(
    k_dim: usize,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
) -> (usize, usize, usize) {
    assert!(k_dim > 0, "input hidden dim must be > 0");
    let q_shape = q_weight.shape_vec();
    let k_shape = k_weight.shape_vec();
    let v_shape = v_weight.shape_vec();
    assert_eq!(q_shape.len(), 2, "Q weight must be 2D [Nq, K]");
    assert_eq!(k_shape.len(), 2, "K weight must be 2D [Nk, K]");
    assert_eq!(v_shape.len(), 2, "V weight must be 2D [Nv, K]");
    let q_n = q_shape[0];
    let k_n = k_shape[0];
    let v_n = v_shape[0];
    assert_eq!(q_shape[1], k_dim, "Q weight K mismatch");
    assert_eq!(k_shape[1], k_dim, "K weight K mismatch");
    assert_eq!(v_shape[1], k_dim, "V weight K mismatch");
    assert_eq!(v_n, k_n, "K/V dim mismatch");
    (q_n, k_n, v_n)
}

enum SliceRef2D<'a> {
    F32Borrowed(ndarray::ArrayViewD<'a, f32>),
    F32Owned(Vec<f32>, usize, usize),
    F16Borrowed(ndarray::ArrayViewD<'a, f16>),
    F16Owned(Vec<f16>, usize, usize),
    BF16Borrowed(ndarray::ArrayViewD<'a, bf16>),
    BF16Owned(Vec<bf16>, usize, usize),
}

impl<'a> SliceRef2D<'a> {
    fn as_slice_ref(&self) -> SliceRef<'_> {
        match self {
            Self::F32Borrowed(view) => SliceRef::F32(
                view.as_slice()
                    .expect("borrowed f32 view should remain contiguous"),
            ),
            Self::F32Owned(slice, _, _) => SliceRef::F32(slice.as_slice()),
            Self::F16Borrowed(view) => SliceRef::F16(
                view.as_slice()
                    .expect("borrowed f16 view should remain contiguous"),
            ),
            Self::F16Owned(slice, _, _) => SliceRef::F16(slice.as_slice()),
            Self::BF16Borrowed(view) => SliceRef::BF16(
                view.as_slice()
                    .expect("borrowed bf16 view should remain contiguous"),
            ),
            Self::BF16Owned(slice, _, _) => SliceRef::BF16(slice.as_slice()),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::F32Borrowed(view) => view.shape()[0],
            Self::F32Owned(_, rows, _)
            | Self::F16Owned(_, rows, _)
            | Self::BF16Owned(_, rows, _) => *rows,
            Self::F16Borrowed(view) => view.shape()[0],
            Self::BF16Borrowed(view) => view.shape()[0],
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::F32Borrowed(view) => view.shape()[1],
            Self::F32Owned(_, _, cols)
            | Self::F16Owned(_, _, cols)
            | Self::BF16Owned(_, _, cols) => *cols,
            Self::F16Borrowed(view) => view.shape()[1],
            Self::BF16Borrowed(view) => view.shape()[1],
        }
    }
}

fn storage_view_2d_as_slice_ref<'a>(view: TensorStorageView<'a>, label: &str) -> SliceRef2D<'a> {
    match view {
        TensorStorageView::F32(view) => {
            let shape = view.shape();
            assert_eq!(shape.len(), 2, "{label} weight must be 2D [N, K]");
            let (rows, cols) = (shape[0], shape[1]);
            if view.as_slice().is_some() {
                SliceRef2D::F32Borrowed(view)
            } else {
                SliceRef2D::F32Owned(view.iter().copied().collect::<Vec<_>>(), rows, cols)
            }
        }
        TensorStorageView::F16(view) => {
            let shape = view.shape();
            assert_eq!(shape.len(), 2, "{label} weight must be 2D [N, K]");
            let (rows, cols) = (shape[0], shape[1]);
            if view.as_slice().is_some() {
                SliceRef2D::F16Borrowed(view)
            } else {
                SliceRef2D::F16Owned(view.iter().copied().collect::<Vec<_>>(), rows, cols)
            }
        }
        TensorStorageView::BF16(view) => {
            let shape = view.shape();
            assert_eq!(shape.len(), 2, "{label} weight must be 2D [N, K]");
            let (rows, cols) = (shape[0], shape[1]);
            if view.as_slice().is_some() {
                SliceRef2D::BF16Borrowed(view)
            } else {
                SliceRef2D::BF16Owned(view.iter().copied().collect::<Vec<_>>(), rows, cols)
            }
        }
    }
}

fn for_each_decode_input_row<'a>(
    x_view: TensorStorageView<'a>,
    rows: usize,
    k_dim: usize,
    mut f: impl FnMut(usize, SliceRef<'_>),
) {
    match x_view {
        TensorStorageView::F32(view) => {
            if let Ok(x_2d) = view.clone().into_shape((rows, k_dim)) {
                for row_idx in 0..rows {
                    let x_row = x_2d.row(row_idx);
                    let x_owned;
                    let x_slice = if let Some(s) = x_row.as_slice() {
                        SliceRef::F32(s)
                    } else {
                        x_owned = x_row.iter().copied().collect::<Vec<_>>();
                        SliceRef::F32(x_owned.as_slice())
                    };
                    f(row_idx, x_slice);
                }
            } else {
                let x_2d = view
                    .to_owned()
                    .into_shape((rows, k_dim))
                    .expect("decode input reshape failed");
                for row_idx in 0..rows {
                    let x_row = x_2d.row(row_idx);
                    let x_slice = SliceRef::F32(
                        x_row
                            .as_slice()
                            .expect("owned decode input row must be contiguous"),
                    );
                    f(row_idx, x_slice);
                }
            }
        }
        TensorStorageView::F16(view) => {
            if let Ok(x_2d) = view.clone().into_shape((rows, k_dim)) {
                for row_idx in 0..rows {
                    let x_row = x_2d.row(row_idx);
                    let x_owned;
                    let x_slice = if let Some(s) = x_row.as_slice() {
                        SliceRef::F16(s)
                    } else {
                        x_owned = x_row.iter().copied().collect::<Vec<_>>();
                        SliceRef::F16(x_owned.as_slice())
                    };
                    f(row_idx, x_slice);
                }
            } else {
                let x_2d = view
                    .to_owned()
                    .into_shape((rows, k_dim))
                    .expect("decode input reshape failed");
                for row_idx in 0..rows {
                    let x_row = x_2d.row(row_idx);
                    let x_slice = SliceRef::F16(
                        x_row
                            .as_slice()
                            .expect("owned decode input row must be contiguous"),
                    );
                    f(row_idx, x_slice);
                }
            }
        }
        TensorStorageView::BF16(view) => {
            if let Ok(x_2d) = view.clone().into_shape((rows, k_dim)) {
                for row_idx in 0..rows {
                    let x_row = x_2d.row(row_idx);
                    let x_owned;
                    let x_slice = if let Some(s) = x_row.as_slice() {
                        SliceRef::BF16(s)
                    } else {
                        x_owned = x_row.iter().copied().collect::<Vec<_>>();
                        SliceRef::BF16(x_owned.as_slice())
                    };
                    f(row_idx, x_slice);
                }
            } else {
                let x_2d = view
                    .to_owned()
                    .into_shape((rows, k_dim))
                    .expect("decode input reshape failed");
                for row_idx in 0..rows {
                    let x_row = x_2d.row(row_idx);
                    let x_slice = SliceRef::BF16(
                        x_row
                            .as_slice()
                            .expect("owned decode input row must be contiguous"),
                    );
                    f(row_idx, x_slice);
                }
            }
        }
    }
}

fn run_qkv_slices(
    x_slice: SliceRef<'_>,
    q_slice: SliceRef<'_>,
    k_slice: SliceRef<'_>,
    v_slice: SliceRef<'_>,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    match (x_slice, q_slice, k_slice, v_slice) {
        (
            SliceRef::F32(x_f32),
            SliceRef::F32(q_slice),
            SliceRef::F32(k_slice),
            SliceRef::F32(v_slice),
        ) => {
            qkv_matvec_rowmajor_parallel(
                x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
            );
        }
        (
            SliceRef::F32(x_f32),
            SliceRef::F16(q_slice),
            SliceRef::F16(k_slice),
            SliceRef::F16(v_slice),
        ) => {
            qkv_matvec_rowmajor_parallel_f32_f16(
                x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
            );
        }
        (
            SliceRef::F16(x_f16),
            SliceRef::F32(q_slice),
            SliceRef::F32(k_slice),
            SliceRef::F32(v_slice),
        ) => {
            with_f16_input_as_f32(x_f16, |x_f32| {
                qkv_matvec_rowmajor_parallel(
                    x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
                );
            });
        }
        (
            SliceRef::F16(x_f16),
            SliceRef::F16(q_slice),
            SliceRef::F16(k_slice),
            SliceRef::F16(v_slice),
        ) => {
            with_f16_input_as_f32(x_f16, |x_f32| {
                qkv_matvec_rowmajor_parallel_f32_f16(
                    x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
                );
            });
        }
        (
            SliceRef::BF16(x_bf16),
            SliceRef::F32(q_slice),
            SliceRef::F32(k_slice),
            SliceRef::F32(v_slice),
        ) => {
            with_bf16_input_as_f32(x_bf16, |x_f32| {
                qkv_matvec_rowmajor_parallel(
                    x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
                );
            });
        }
        (
            SliceRef::F32(x_f32),
            SliceRef::BF16(q_slice),
            SliceRef::BF16(k_slice),
            SliceRef::BF16(v_slice),
        ) => {
            qkv_matvec_rowmajor_parallel_f32_bf16(
                x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
            );
        }
        (
            SliceRef::BF16(x_bf16),
            SliceRef::BF16(q_slice),
            SliceRef::BF16(k_slice),
            SliceRef::BF16(v_slice),
        ) => {
            with_bf16_input_as_f32(x_bf16, |x_f32| {
                qkv_matvec_rowmajor_parallel_f32_bf16(
                    x_f32, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
                );
            });
        }
        (
            SliceRef::F32(x_f32),
            SliceRef::I8(q_slice, q_scale),
            SliceRef::I8(k_slice, k_scale),
            SliceRef::I8(v_slice, v_scale),
        ) => {
            qkv_matvec_rowmajor_parallel_f32_i8(
                x_f32, q_slice, q_scale, k_slice, k_scale, v_slice, v_scale, q_n, k_n, k_dim,
                q_out, k_out, v_out,
            );
        }
        (
            SliceRef::BF16(x_bf16),
            SliceRef::I8(q_slice, q_scale),
            SliceRef::I8(k_slice, k_scale),
            SliceRef::I8(v_slice, v_scale),
        ) => {
            with_bf16_input_as_f32(x_bf16, |x_f32| {
                qkv_matvec_rowmajor_parallel_f32_i8(
                    x_f32, q_slice, q_scale, k_slice, k_scale, v_slice, v_scale, q_n, k_n, k_dim,
                    q_out, k_out, v_out,
                );
            });
        }
        (
            SliceRef::F16(x_f16),
            SliceRef::I8(q_slice, q_scale),
            SliceRef::I8(k_slice, k_scale),
            SliceRef::I8(v_slice, v_scale),
        ) => {
            with_f16_input_as_f32(x_f16, |x_f32| {
                qkv_matvec_rowmajor_parallel_f32_i8(
                    x_f32, q_slice, q_scale, k_slice, k_scale, v_slice, v_scale, q_n, k_n, k_dim,
                    q_out, k_out, v_out,
                );
            });
        }
        (x_slice, q_slice, k_slice, v_slice) => {
            matvec_rowmajor_parallel_mixed(x_slice, q_slice, q_n, k_dim, q_out);
            dual_matvec_rowmajor_parallel_mixed(
                x_slice, k_slice, v_slice, k_n, k_dim, k_out, v_out,
            );
        }
    }
}

fn run_gate_up_slice(
    x_slice: SliceRef<'_>,
    gate_slice: SliceRef<'_>,
    up_slice: SliceRef<'_>,
    n_dim: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    match (x_slice, gate_slice, up_slice) {
        (SliceRef::F32(x_f32), SliceRef::F16(gate_slice), SliceRef::F16(up_slice)) => {
            dual_matvec_silu_mul_rowmajor_parallel_f32_f16(
                x_f32, gate_slice, up_slice, n_dim, k_dim, out,
            );
        }
        (SliceRef::F16(x_f16), SliceRef::F16(gate_slice), SliceRef::F16(up_slice)) => {
            with_f16_input_as_f32(x_f16, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_f16(
                    x_f32, gate_slice, up_slice, n_dim, k_dim, out,
                );
            });
        }
        (SliceRef::F32(x_f32), SliceRef::BF16(gate_slice), SliceRef::BF16(up_slice)) => {
            dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(
                x_f32, gate_slice, up_slice, n_dim, k_dim, out,
            );
        }
        (SliceRef::BF16(x_bf16), SliceRef::BF16(gate_slice), SliceRef::BF16(up_slice)) => {
            with_bf16_input_as_f32(x_bf16, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(
                    x_f32, gate_slice, up_slice, n_dim, k_dim, out,
                );
            });
        }
        (
            SliceRef::F32(x_f32),
            SliceRef::I8(gate_slice, gate_scale),
            SliceRef::I8(up_slice, up_scale),
        ) => {
            dual_matvec_silu_mul_rowmajor_parallel_f32_i8(
                x_f32, gate_slice, gate_scale, up_slice, up_scale, n_dim, k_dim, out,
            );
        }
        (
            SliceRef::BF16(x_bf16),
            SliceRef::I8(gate_slice, gate_scale),
            SliceRef::I8(up_slice, up_scale),
        ) => {
            with_bf16_input_as_f32(x_bf16, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_i8(
                    x_f32, gate_slice, gate_scale, up_slice, up_scale, n_dim, k_dim, out,
                );
            });
        }
        (
            SliceRef::F16(x_f16),
            SliceRef::I8(gate_slice, gate_scale),
            SliceRef::I8(up_slice, up_scale),
        ) => {
            with_f16_input_as_f32(x_f16, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_i8(
                    x_f32, gate_slice, gate_scale, up_slice, up_scale, n_dim, k_dim, out,
                );
            });
        }
        (x_slice, gate_slice, up_slice) => {
            dual_matvec_silu_mul_rowmajor_parallel_mixed(
                x_slice, gate_slice, up_slice, n_dim, k_dim, out,
            );
        }
    }
}

pub fn fused_softmax(input: &Tensor, scale: f32, is_causal: bool) -> Tensor {
    let build_graph = !is_no_grad() && input.requires_grad();

    if !build_graph {
        return input.with_storage_view_preferring(
            StoragePreference::Native,
            |x_view| match x_view {
                TensorStorageView::F32(x_view) => {
                    let shape = x_view.shape().to_vec();
                    if shape.len() != 4 {
                        panic!("Fused Softmax expects 4D input [B, H, Q, K]");
                    }

                    let q_len = shape[2];
                    let k_len = shape[3];

                    let mut out = Array::zeros(x_view.raw_dim());
                    let x_view = x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
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
                                    for i in 0..q_len {
                                        let row_in = x_h.slice(ndarray::s![i, ..]);
                                        let mut row_out = out_h.slice_mut(ndarray::s![i, ..]);

                                        let mut max_val = f32::NEG_INFINITY;
                                        for j in 0..k_len {
                                            let is_masked =
                                                if is_causal && q_len > 1 { j > i } else { false };
                                            if !is_masked {
                                                let val = row_in[j] * scale;
                                                if val > max_val {
                                                    max_val = val;
                                                }
                                            }
                                        }

                                        let mut sum_exp = 0.0;
                                        for j in 0..k_len {
                                            let is_masked =
                                                if is_causal && q_len > 1 { j > i } else { false };
                                            if is_masked {
                                                row_out[j] = 0.0;
                                            } else {
                                                let val = (row_in[j] * scale - max_val).exp();
                                                row_out[j] = val;
                                                sum_exp += val;
                                            }
                                        }

                                        let inv_sum = 1.0 / (sum_exp + 1e-10);
                                        for j in 0..k_len {
                                            row_out[j] *= inv_sum;
                                        }
                                    }
                                });
                        });

                    Tensor::from_array_no_grad(out.into_dyn())
                }
                TensorStorageView::F16(x_view) => {
                    let shape = x_view.shape().to_vec();
                    if shape.len() != 4 {
                        panic!("Fused Softmax expects 4D input [B, H, Q, K]");
                    }

                    let q_len = shape[2];
                    let k_len = shape[3];

                    let mut out = ndarray::ArrayD::<f16>::from_elem(
                        ndarray::IxDyn(&shape),
                        f16::from_bits(0),
                    );
                    let x_view = x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
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
                                    for i in 0..q_len {
                                        let row_in = x_h.slice(ndarray::s![i, ..]);
                                        let mut row_out = out_h.slice_mut(ndarray::s![i, ..]);
                                        let mut row_exp = vec![0.0f32; k_len];

                                        let mut max_val = f32::NEG_INFINITY;
                                        for j in 0..k_len {
                                            let is_masked =
                                                if is_causal && q_len > 1 { j > i } else { false };
                                            if !is_masked {
                                                let val = row_in[j].to_f32() * scale;
                                                if val > max_val {
                                                    max_val = val;
                                                }
                                            }
                                        }

                                        let mut sum_exp = 0.0;
                                        for j in 0..k_len {
                                            let is_masked =
                                                if is_causal && q_len > 1 { j > i } else { false };
                                            if is_masked {
                                                row_out[j] = f16::from_bits(0);
                                                row_exp[j] = 0.0;
                                            } else {
                                                let val =
                                                    (row_in[j].to_f32() * scale - max_val).exp();
                                                row_exp[j] = val;
                                                sum_exp += val;
                                            }
                                        }

                                        let inv_sum = 1.0 / (sum_exp + 1e-10);
                                        for j in 0..k_len {
                                            row_out[j] = f16::from_f32(row_exp[j] * inv_sum);
                                        }
                                    }
                                });
                        });

                    Tensor::from_f16_data_no_grad(out.into_shared())
                }
                TensorStorageView::BF16(x_view) => {
                    let shape = x_view.shape().to_vec();
                    if shape.len() != 4 {
                        panic!("Fused Softmax expects 4D input [B, H, Q, K]");
                    }

                    let q_len = shape[2];
                    let k_len = shape[3];

                    let mut out = ndarray::ArrayD::<bf16>::from_elem(
                        ndarray::IxDyn(&shape),
                        bf16::from_bits(0),
                    );
                    let x_view = x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
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
                                    for i in 0..q_len {
                                        let row_in = x_h.slice(ndarray::s![i, ..]);
                                        let mut row_out = out_h.slice_mut(ndarray::s![i, ..]);
                                        let mut row_exp = vec![0.0f32; k_len];

                                        let mut max_val = f32::NEG_INFINITY;
                                        for j in 0..k_len {
                                            let is_masked =
                                                if is_causal && q_len > 1 { j > i } else { false };
                                            if !is_masked {
                                                let val = row_in[j].to_f32() * scale;
                                                if val > max_val {
                                                    max_val = val;
                                                }
                                            }
                                        }

                                        let mut sum_exp = 0.0;
                                        for j in 0..k_len {
                                            let is_masked =
                                                if is_causal && q_len > 1 { j > i } else { false };
                                            if is_masked {
                                                row_out[j] = bf16::from_bits(0);
                                                row_exp[j] = 0.0;
                                            } else {
                                                let val =
                                                    (row_in[j].to_f32() * scale - max_val).exp();
                                                row_exp[j] = val;
                                                sum_exp += val;
                                            }
                                        }

                                        let inv_sum = 1.0 / (sum_exp + 1e-10);
                                        for j in 0..k_len {
                                            row_out[j] = bf16::from_f32(row_exp[j] * inv_sum);
                                        }
                                    }
                                });
                        });

                    Tensor::from_bf16_data_no_grad(out.into_shared())
                }
            },
        );
    }

    let (output, output_data) = {
        let x = input.data_ref();
        let shape = x.shape();
        if shape.len() != 4 {
            panic!("Fused Softmax expects 4D input [B, H, Q, K]");
        }

        let q_len = shape[2];
        let k_len = shape[3];

        let mut out = Array::zeros(x.dim());
        let x_view = x.view().into_dimensionality::<ndarray::Ix4>().unwrap();
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
                        for i in 0..q_len {
                            let row_in = x_h.slice(ndarray::s![i, ..]);
                            let mut row_out = out_h.slice_mut(ndarray::s![i, ..]);

                            let mut max_val = f32::NEG_INFINITY;
                            for j in 0..k_len {
                                let is_masked = if is_causal && q_len > 1 { j > i } else { false };
                                if !is_masked {
                                    let val = row_in[j] * scale;
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }

                            let mut sum_exp = 0.0;
                            for j in 0..k_len {
                                let is_masked = if is_causal && q_len > 1 { j > i } else { false };
                                if is_masked {
                                    row_out[j] = 0.0;
                                } else {
                                    let val = (row_in[j] * scale - max_val).exp();
                                    row_out[j] = val;
                                    sum_exp += val;
                                }
                            }

                            let inv_sum = 1.0 / (sum_exp + 1e-10);
                            for j in 0..k_len {
                                row_out[j] *= inv_sum;
                            }
                        }
                    });
            });

        (out.clone().into_dyn(), out)
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
            let y = &output_data;
            let y_grad = y * grad;
            let sum_y_grad = y_grad.sum_axis(Axis(3)).insert_axis(Axis(3));
            let dx = y * (grad - &sum_y_grad) * scale;
            input_clone.add_grad(dx);
        })),
        requires_grad: true,
    })))
}

pub fn fused_gate_up_silu_infer(
    input: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
) -> Tensor {
    assert!(is_no_grad(), "fused_gate_up_silu_infer is inference-only");

    let x_shape = input.shape_vec();
    let k_dim = *x_shape.last().expect("input must have last dim");
    let n_dim = validate_gate_up_shapes(k_dim, gate_weight, up_weight);
    let m_dim = input.len() / k_dim;
    if m_dim == 1 {
        let mut out = vec![0.0f32; n_dim];
        fused_gate_up_silu_infer_into(input, gate_weight, up_weight, &mut out);
        let mut out_shape = x_shape;
        let last = out_shape.len() - 1;
        out_shape[last] = n_dim;
        return Tensor::from_array_no_grad(
            Array2::from_shape_vec((1, n_dim), out)
                .expect("decode gate/up output shape build failed")
                .into_shape(out_shape)
                .expect("decode gate/up reshape failed")
                .into_dyn(),
        );
    }

    let (out, n_dim) = input.with_storage_view_preferring(StoragePreference::Native, |x_view| {
        let input_dtype = match &x_view {
            TensorStorageView::F32(_) => DType::F32,
            TensorStorageView::F16(_) => DType::F16,
            TensorStorageView::BF16(_) => DType::BF16,
        };
        gate_weight.with_storage_view_for_input_dtype(input_dtype, |gate_view| {
            up_weight.with_storage_view_for_input_dtype(input_dtype, |up_view| {
                macro_rules! run_gate_up {
                    ($gate_slice:expr, $up_slice:expr, $n_dim:expr) => {{
                        let n_dim = $n_dim;
                        let gate_slice = $gate_slice;
                        let up_slice = $up_slice;
                        let mut out = Array2::<f32>::zeros((m_dim, n_dim));
                        for_each_decode_input_row(x_view, m_dim, k_dim, |row_idx, x_slice| {
                            let mut out_row = out.slice_mut(ndarray::s![row_idx, ..]);
                            let out_slice = out_row
                                .as_slice_mut()
                                .expect("output row should be contiguous");
                            dual_matvec_silu_mul_rowmajor_parallel_mixed(
                                x_slice, gate_slice, up_slice, n_dim, k_dim, out_slice,
                            );
                        });
                        (out, n_dim)
                    }};
                }

                match (gate_view, up_view) {
                    (TensorStorageView::F32(gate_view), TensorStorageView::F32(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        let n_dim = gate_2d.nrows();
                        assert_eq!(gate_2d.ncols(), k_dim, "gate weight K mismatch");
                        assert_eq!(up_2d.nrows(), n_dim, "gate/up out dim mismatch");
                        assert_eq!(up_2d.ncols(), k_dim, "up weight K mismatch");
                        run_gate_up!(
                            SliceRef::F32(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::F32(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        )
                    }
                    (TensorStorageView::F32(gate_view), TensorStorageView::BF16(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        let n_dim = gate_2d.nrows();
                        assert_eq!(gate_2d.ncols(), k_dim, "gate weight K mismatch");
                        assert_eq!(up_2d.nrows(), n_dim, "gate/up out dim mismatch");
                        assert_eq!(up_2d.ncols(), k_dim, "up weight K mismatch");
                        run_gate_up!(
                            SliceRef::F32(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::BF16(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        )
                    }
                    (TensorStorageView::BF16(gate_view), TensorStorageView::F32(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        let n_dim = gate_2d.nrows();
                        assert_eq!(gate_2d.ncols(), k_dim, "gate weight K mismatch");
                        assert_eq!(up_2d.nrows(), n_dim, "gate/up out dim mismatch");
                        assert_eq!(up_2d.ncols(), k_dim, "up weight K mismatch");
                        run_gate_up!(
                            SliceRef::BF16(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::F32(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        )
                    }
                    (TensorStorageView::BF16(gate_view), TensorStorageView::BF16(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        let n_dim = gate_2d.nrows();
                        assert_eq!(gate_2d.ncols(), k_dim, "gate weight K mismatch");
                        assert_eq!(up_2d.nrows(), n_dim, "gate/up out dim mismatch");
                        assert_eq!(up_2d.ncols(), k_dim, "up weight K mismatch");
                        run_gate_up!(
                            SliceRef::BF16(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::BF16(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        )
                    }
                    (gate_view, up_view) => {
                        let gate_slice = storage_view_2d_as_slice_ref(gate_view, "gate");
                        let up_slice = storage_view_2d_as_slice_ref(up_view, "up");
                        assert_eq!(gate_slice.cols(), k_dim, "gate weight K mismatch");
                        assert_eq!(
                            up_slice.rows(),
                            gate_slice.rows(),
                            "gate/up out dim mismatch"
                        );
                        assert_eq!(up_slice.cols(), k_dim, "up weight K mismatch");
                        run_gate_up!(
                            gate_slice.as_slice_ref(),
                            up_slice.as_slice_ref(),
                            gate_slice.rows()
                        )
                    }
                }
            })
        })
    });

    let mut out_shape = x_shape;
    let last = out_shape.len() - 1;
    out_shape[last] = n_dim;
    Tensor::from_array_no_grad(out.into_shape(out_shape).unwrap().into_dyn())
}

pub fn fused_gate_up_silu_infer_into(
    input: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    out: &mut [f32],
) {
    assert!(
        is_no_grad(),
        "fused_gate_up_silu_infer_into is inference-only"
    );

    let x_shape = input.shape_vec();
    let k_dim = *x_shape.last().expect("input must have last dim");
    let n_dim = validate_gate_up_shapes(k_dim, gate_weight, up_weight);
    let m_dim = input.len() / k_dim;
    assert_eq!(
        m_dim, 1,
        "fused_gate_up_silu_infer_into currently expects single-token decode input"
    );
    assert_eq!(out.len(), n_dim, "output size mismatch");
    if gate_weight.dtype() == crate::precision::DType::I8
        && up_weight.dtype() == crate::precision::DType::I8
    {
        let gate_owned = gate_weight.native_storage_owned();
        let up_owned = up_weight.native_storage_owned();
        return with_decode_input_as_slice_ref(input, |x_slice| match (gate_owned, up_owned) {
            (
                TensorStorageOwned::I8(gate_data, gate_scale),
                TensorStorageOwned::I8(up_data, up_scale),
            ) => {
                let gate_2d = gate_data
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("gate weight must be 2D [N, K]");
                let up_2d = up_data
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("up weight must be 2D [N, K]");
                run_gate_up_slice(
                    x_slice,
                    SliceRef::I8(
                        gate_2d.as_slice().expect("gate weight must be contiguous"),
                        gate_scale,
                    ),
                    SliceRef::I8(
                        up_2d.as_slice().expect("up weight must be contiguous"),
                        up_scale,
                    ),
                    n_dim,
                    k_dim,
                    out,
                );
            }
            _ => unreachable!("checked I8 weights above"),
        });
    }
    with_decode_input_as_slice_ref(input, |x_slice| {
        let input_dtype = slice_ref_dtype(x_slice);
        gate_weight.with_storage_view_for_input_dtype(input_dtype, |gate_view| {
            up_weight.with_storage_view_for_input_dtype(input_dtype, |up_view| {
                macro_rules! run_gate_up_into {
                    ($gate_slice:expr, $up_slice:expr, $n_dim:expr) => {{
                        let n_dim = $n_dim;
                        let gate_slice = $gate_slice;
                        let up_slice = $up_slice;
                        assert_eq!(out.len(), n_dim, "output size mismatch");
                        run_gate_up_slice(x_slice, gate_slice, up_slice, n_dim, k_dim, out);
                    }};
                }

                match (gate_view, up_view) {
                    (TensorStorageView::F32(gate_view), TensorStorageView::F32(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        run_gate_up_into!(
                            SliceRef::F32(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::F32(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        );
                    }
                    (TensorStorageView::F32(gate_view), TensorStorageView::BF16(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        run_gate_up_into!(
                            SliceRef::F32(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::BF16(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        );
                    }
                    (TensorStorageView::BF16(gate_view), TensorStorageView::F32(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        run_gate_up_into!(
                            SliceRef::BF16(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::F32(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        );
                    }
                    (TensorStorageView::BF16(gate_view), TensorStorageView::BF16(up_view)) => {
                        let gate_2d = gate_view
                            .into_dimensionality::<Ix2>()
                            .expect("gate weight must be 2D [N, K]");
                        let up_2d = up_view
                            .into_dimensionality::<Ix2>()
                            .expect("up weight must be 2D [N, K]");
                        run_gate_up_into!(
                            SliceRef::BF16(
                                gate_2d.as_slice().expect("gate weight must be contiguous")
                            ),
                            SliceRef::BF16(up_2d.as_slice().expect("up weight must be contiguous")),
                            n_dim
                        );
                    }
                    (gate_view, up_view) => {
                        let gate_slice = storage_view_2d_as_slice_ref(gate_view, "gate");
                        let up_slice = storage_view_2d_as_slice_ref(up_view, "up");
                        assert_eq!(gate_slice.cols(), k_dim, "gate weight K mismatch");
                        assert_eq!(up_slice.cols(), k_dim, "up weight K mismatch");
                        run_gate_up_into!(
                            gate_slice.as_slice_ref(),
                            up_slice.as_slice_ref(),
                            n_dim
                        );
                    }
                }
            })
        })
    });
}

pub fn fused_qkv_decode_infer_into(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert!(
        is_no_grad(),
        "fused_qkv_decode_infer_into is inference-only"
    );

    let x_shape = input.shape_vec();
    assert_eq!(x_shape.len(), 3, "decode input must be [B, S, K]");
    let (b, s, k_dim) = (x_shape[0], x_shape[1], x_shape[2]);
    let (q_n, k_n, v_n) = validate_qkv_shapes(k_dim, q_weight, k_weight, v_weight);
    assert_eq!(
        b, 1,
        "fused_qkv_decode_infer_into currently expects batch size 1"
    );
    assert_eq!(s, 1, "fused_qkv_decode_infer_into only supports S=1 decode");
    assert_eq!(q_out.len(), q_n, "Q output size mismatch");
    assert_eq!(k_out.len(), k_n, "K output size mismatch");
    assert_eq!(v_out.len(), v_n, "V output size mismatch");
    if q_weight.dtype() == crate::precision::DType::I8
        && k_weight.dtype() == crate::precision::DType::I8
        && v_weight.dtype() == crate::precision::DType::I8
    {
        let q_owned = q_weight.native_storage_owned();
        let k_owned = k_weight.native_storage_owned();
        let v_owned = v_weight.native_storage_owned();
        return with_decode_input_as_slice_ref(input, |x_slice| {
            match (q_owned, k_owned, v_owned) {
                (
                    TensorStorageOwned::I8(q_data, q_scale),
                    TensorStorageOwned::I8(k_data, k_scale),
                    TensorStorageOwned::I8(v_data, v_scale),
                ) => {
                    let q_2d = q_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("Q weight must be 2D [Nq, K]");
                    let k_2d = k_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("K weight must be 2D [Nk, K]");
                    let v_2d = v_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("V weight must be 2D [Nv, K]");
                    run_qkv_slices(
                        x_slice,
                        SliceRef::I8(
                            q_2d.as_slice().expect("Q weight must be contiguous"),
                            q_scale,
                        ),
                        SliceRef::I8(
                            k_2d.as_slice().expect("K weight must be contiguous"),
                            k_scale,
                        ),
                        SliceRef::I8(
                            v_2d.as_slice().expect("V weight must be contiguous"),
                            v_scale,
                        ),
                        q_n,
                        k_n,
                        k_dim,
                        q_out,
                        k_out,
                        v_out,
                    );
                }
                _ => unreachable!("checked I8 weights above"),
            }
        });
    }
    with_decode_input_as_slice_ref(input, |x_slice| {
        let input_dtype = slice_ref_dtype(x_slice);
        q_weight.with_storage_view_for_input_dtype(input_dtype, |q_view| {
            k_weight.with_storage_view_for_input_dtype(input_dtype, |k_view| {
                v_weight.with_storage_view_for_input_dtype(input_dtype, |v_view| {
                    macro_rules! run_qkv {
                        ($q_slice:expr, $k_slice:expr, $v_slice:expr, $q_n:expr, $k_n:expr, $v_n:expr) => {{
                            let q_n = $q_n;
                            let k_n = $k_n;
                            let v_n = $v_n;
                            let q_slice = $q_slice;
                            let k_slice = $k_slice;
                            let v_slice = $v_slice;
                            assert_eq!(q_out.len(), q_n, "Q output size mismatch");
                            assert_eq!(k_out.len(), k_n, "K output size mismatch");
                            assert_eq!(v_out.len(), v_n, "V output size mismatch");
                            assert_eq!(v_n, k_n, "K/V dim mismatch");
                            run_qkv_slices(
                                x_slice, q_slice, k_slice, v_slice, q_n, k_n, k_dim, q_out, k_out, v_out,
                            );
                        }};
                    }

                    let q_slice = storage_view_2d_as_slice_ref(q_view, "Q");
                    let k_slice = storage_view_2d_as_slice_ref(k_view, "K");
                    let v_slice = storage_view_2d_as_slice_ref(v_view, "V");
                    assert_eq!(q_slice.cols(), k_dim, "Q weight K mismatch");
                    assert_eq!(k_slice.cols(), k_dim, "K weight K mismatch");
                    assert_eq!(v_slice.cols(), k_dim, "V weight K mismatch");
                    run_qkv!(
                        q_slice.as_slice_ref(),
                        k_slice.as_slice_ref(),
                        v_slice.as_slice_ref(),
                        q_slice.rows(),
                        k_slice.rows(),
                        v_slice.rows()
                    );
                })
            })
        })
    });
}
pub fn fused_qkv_decode_infer(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    n_head: usize,
    n_kv_head: usize,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    assert!(is_no_grad(), "fused_qkv_decode_infer is inference-only");

    let x_shape = input.shape_vec();
    assert_eq!(x_shape.len(), 3, "decode input must be [B, S, K]");
    let (b, s, k_dim) = (x_shape[0], x_shape[1], x_shape[2]);
    assert_eq!(s, 1, "fused_qkv_decode_infer only supports S=1 decode");
    assert!(n_head > 0, "n_head must be > 0");
    assert!(n_kv_head > 0, "n_kv_head must be > 0");
    let (q_n, k_n, v_n) = validate_qkv_shapes(k_dim, q_weight, k_weight, v_weight);
    assert_eq!(q_n % n_head, 0, "Q dim must be divisible by n_head");
    assert_eq!(k_n % n_kv_head, 0, "K dim must be divisible by n_kv_head");

    let d = q_n / n_head;
    assert_eq!(k_n / n_kv_head, d, "Q/K head dim mismatch");

    if b == 1 {
        let mut q_out = vec![0.0f32; q_n];
        let mut k_out = vec![0.0f32; k_n];
        let mut v_out = vec![0.0f32; v_n];
        fused_qkv_decode_infer_into(
            input, q_weight, k_weight, v_weight, &mut q_out, &mut k_out, &mut v_out,
        );
        let q = Array3::from_shape_vec((1, n_head, d), q_out).expect("decode Q shape build failed");
        let k =
            Array3::from_shape_vec((1, n_kv_head, d), k_out).expect("decode K shape build failed");
        let v =
            Array3::from_shape_vec((1, n_kv_head, d), v_out).expect("decode V shape build failed");
        return (q, k, v);
    }

    let mut q_out = Array2::<f32>::zeros((b, q_n));
    let mut k_out = Array2::<f32>::zeros((b, k_n));
    let mut v_out = Array2::<f32>::zeros((b, v_n));
    if q_weight.dtype() == crate::precision::DType::I8
        && k_weight.dtype() == crate::precision::DType::I8
        && v_weight.dtype() == crate::precision::DType::I8
    {
        let q_owned = q_weight.native_storage_owned();
        let k_owned = k_weight.native_storage_owned();
        let v_owned = v_weight.native_storage_owned();
        input.with_storage_view_preferring(StoragePreference::Native, |x_view| {
            match (q_owned, k_owned, v_owned, x_view) {
                (
                    TensorStorageOwned::I8(q_data, q_scale),
                    TensorStorageOwned::I8(k_data, k_scale),
                    TensorStorageOwned::I8(v_data, v_scale),
                    TensorStorageView::F32(x_view),
                ) => {
                    let q_2d = q_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("Q weight must be 2D [Nq, K]");
                    let k_2d = k_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("K weight must be 2D [Nk, K]");
                    let v_2d = v_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("V weight must be 2D [Nv, K]");
                    for_each_decode_input_row(
                        TensorStorageView::F32(x_view),
                        b,
                        k_dim,
                        |bb, x_slice| {
                            let mut q_row_view = q_out.slice_mut(ndarray::s![bb, ..]);
                            let mut k_row_view = k_out.slice_mut(ndarray::s![bb, ..]);
                            let mut v_row_view = v_out.slice_mut(ndarray::s![bb, ..]);
                            run_qkv_slices(
                                x_slice,
                                SliceRef::I8(
                                    q_2d.as_slice().expect("Q weight must be contiguous"),
                                    q_scale,
                                ),
                                SliceRef::I8(
                                    k_2d.as_slice().expect("K weight must be contiguous"),
                                    k_scale,
                                ),
                                SliceRef::I8(
                                    v_2d.as_slice().expect("V weight must be contiguous"),
                                    v_scale,
                                ),
                                q_n,
                                k_n,
                                k_dim,
                                q_row_view
                                    .as_slice_mut()
                                    .expect("Q output row not contiguous"),
                                k_row_view
                                    .as_slice_mut()
                                    .expect("K output row not contiguous"),
                                v_row_view
                                    .as_slice_mut()
                                    .expect("V output row not contiguous"),
                            );
                        },
                    );
                }
                (
                    TensorStorageOwned::I8(q_data, q_scale),
                    TensorStorageOwned::I8(k_data, k_scale),
                    TensorStorageOwned::I8(v_data, v_scale),
                    TensorStorageView::F16(x_view),
                ) => {
                    let q_2d = q_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("Q weight must be 2D [Nq, K]");
                    let k_2d = k_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("K weight must be 2D [Nk, K]");
                    let v_2d = v_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("V weight must be 2D [Nv, K]");
                    for_each_decode_input_row(
                        TensorStorageView::F16(x_view),
                        b,
                        k_dim,
                        |bb, x_slice| {
                            let mut q_row_view = q_out.slice_mut(ndarray::s![bb, ..]);
                            let mut k_row_view = k_out.slice_mut(ndarray::s![bb, ..]);
                            let mut v_row_view = v_out.slice_mut(ndarray::s![bb, ..]);
                            run_qkv_slices(
                                x_slice,
                                SliceRef::I8(
                                    q_2d.as_slice().expect("Q weight must be contiguous"),
                                    q_scale,
                                ),
                                SliceRef::I8(
                                    k_2d.as_slice().expect("K weight must be contiguous"),
                                    k_scale,
                                ),
                                SliceRef::I8(
                                    v_2d.as_slice().expect("V weight must be contiguous"),
                                    v_scale,
                                ),
                                q_n,
                                k_n,
                                k_dim,
                                q_row_view
                                    .as_slice_mut()
                                    .expect("Q output row not contiguous"),
                                k_row_view
                                    .as_slice_mut()
                                    .expect("K output row not contiguous"),
                                v_row_view
                                    .as_slice_mut()
                                    .expect("V output row not contiguous"),
                            );
                        },
                    );
                }
                (
                    TensorStorageOwned::I8(q_data, q_scale),
                    TensorStorageOwned::I8(k_data, k_scale),
                    TensorStorageOwned::I8(v_data, v_scale),
                    TensorStorageView::BF16(x_view),
                ) => {
                    let q_2d = q_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("Q weight must be 2D [Nq, K]");
                    let k_2d = k_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("K weight must be 2D [Nk, K]");
                    let v_2d = v_data
                        .view()
                        .into_dimensionality::<Ix2>()
                        .expect("V weight must be 2D [Nv, K]");
                    for_each_decode_input_row(
                        TensorStorageView::BF16(x_view),
                        b,
                        k_dim,
                        |bb, x_slice| {
                            let mut q_row_view = q_out.slice_mut(ndarray::s![bb, ..]);
                            let mut k_row_view = k_out.slice_mut(ndarray::s![bb, ..]);
                            let mut v_row_view = v_out.slice_mut(ndarray::s![bb, ..]);
                            run_qkv_slices(
                                x_slice,
                                SliceRef::I8(
                                    q_2d.as_slice().expect("Q weight must be contiguous"),
                                    q_scale,
                                ),
                                SliceRef::I8(
                                    k_2d.as_slice().expect("K weight must be contiguous"),
                                    k_scale,
                                ),
                                SliceRef::I8(
                                    v_2d.as_slice().expect("V weight must be contiguous"),
                                    v_scale,
                                ),
                                q_n,
                                k_n,
                                k_dim,
                                q_row_view
                                    .as_slice_mut()
                                    .expect("Q output row not contiguous"),
                                k_row_view
                                    .as_slice_mut()
                                    .expect("K output row not contiguous"),
                                v_row_view
                                    .as_slice_mut()
                                    .expect("V output row not contiguous"),
                            );
                        },
                    );
                }
                _ => unreachable!("checked I8 weights above"),
            }
        });

        return (
            q_out
                .into_shape((b, n_head, d))
                .expect("Q output reshape failed"),
            k_out
                .into_shape((b, n_kv_head, d))
                .expect("K output reshape failed"),
            v_out
                .into_shape((b, n_kv_head, d))
                .expect("V output reshape failed"),
        );
    }
    input.with_storage_view_preferring(StoragePreference::Native, |x_view| {
        let input_dtype = match &x_view {
            TensorStorageView::F32(_) => DType::F32,
            TensorStorageView::F16(_) => DType::F16,
            TensorStorageView::BF16(_) => DType::BF16,
        };
        q_weight.with_storage_view_for_input_dtype(input_dtype, |q_view| {
            k_weight.with_storage_view_for_input_dtype(input_dtype, |k_view| {
                v_weight.with_storage_view_for_input_dtype(input_dtype, |v_view| {
                    macro_rules! run_qkv_rows {
                        ($q_slice:expr, $k_slice:expr, $v_slice:expr) => {{
                            let q_slice = $q_slice;
                            let k_slice = $k_slice;
                            let v_slice = $v_slice;
                            for_each_decode_input_row(x_view, b, k_dim, |bb, x_slice| {
                                let mut q_row_view = q_out.slice_mut(ndarray::s![bb, ..]);
                                let q_out_slice = q_row_view
                                    .as_slice_mut()
                                    .expect("Q output row not contiguous");
                                let mut k_row_view = k_out.slice_mut(ndarray::s![bb, ..]);
                                let k_out_slice = k_row_view
                                    .as_slice_mut()
                                    .expect("K output row not contiguous");
                                let mut v_row_view = v_out.slice_mut(ndarray::s![bb, ..]);
                                let v_out_slice = v_row_view
                                    .as_slice_mut()
                                    .expect("V output row not contiguous");
                                run_qkv_slices(
                                    x_slice,
                                    q_slice,
                                    k_slice,
                                    v_slice,
                                    q_n,
                                    k_n,
                                    k_dim,
                                    q_out_slice,
                                    k_out_slice,
                                    v_out_slice,
                                );
                            });
                        }};
                    }

                    let q_slice = storage_view_2d_as_slice_ref(q_view, "Q");
                    let k_slice = storage_view_2d_as_slice_ref(k_view, "K");
                    let v_slice = storage_view_2d_as_slice_ref(v_view, "V");
                    assert_eq!(q_slice.cols(), k_dim, "Q weight K mismatch");
                    assert_eq!(k_slice.cols(), k_dim, "K weight K mismatch");
                    assert_eq!(v_slice.cols(), k_dim, "V weight K mismatch");
                    run_qkv_rows!(
                        q_slice.as_slice_ref(),
                        k_slice.as_slice_ref(),
                        v_slice.as_slice_ref()
                    );
                })
            })
        })
    });

    (
        q_out
            .into_shape((b, n_head, d))
            .expect("Q output reshape failed"),
        k_out
            .into_shape((b, n_kv_head, d))
            .expect("K output reshape failed"),
        v_out
            .into_shape((b, n_kv_head, d))
            .expect("V output reshape failed"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    use crate::precision::DType;
    use ndarray::IxDyn;

    fn sample_f32(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 19 + 7) % 31) as f32) / 15.0 - 1.0)
            .collect()
    }

    fn quantize_bf16(src: &[f32]) -> Vec<f32> {
        src.iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect()
    }

    fn quantize_f16(src: &[f32]) -> Vec<f32> {
        src.iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect()
    }

    fn quantize_i8(shape: &[usize], src: &[f32]) -> Vec<f32> {
        let t = make_tensor(shape, src.to_vec(), DType::I8);
        t.data_ref().iter().copied().collect()
    }

    fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let t = Tensor::from_array_no_grad(
            Array::from_shape_vec(IxDyn(shape), data)
                .unwrap()
                .into_dyn(),
        );
        t.cast_inplace(dtype);
        t
    }

    fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "idx={idx}, lhs={a}, rhs={b}, tol={tol}"
            );
        }
    }

    #[test]
    fn fused_qkv_bf16_matches_quantized_reference() {
        let hidden = 8usize;
        let x = sample_f32(hidden);
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let x_q = quantize_bf16(&x);
        let q_q = quantize_bf16(&q);
        let k_q = quantize_bf16(&k);
        let v_q = quantize_bf16(&v);

        let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
        let input_bf16 = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
        let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
        let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
        let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
        let q_bf16 = make_tensor(&[hidden, hidden], q.clone(), DType::BF16);
        let k_bf16 = make_tensor(&[hidden, hidden], k.clone(), DType::BF16);
        let v_bf16 = make_tensor(&[hidden, hidden], v.clone(), DType::BF16);

        let mut q_ref = vec![0.0f32; hidden];
        let mut k_ref = vec![0.0f32; hidden];
        let mut v_ref = vec![0.0f32; hidden];
        let mut q_out = vec![0.0f32; hidden];
        let mut k_out = vec![0.0f32; hidden];
        let mut v_out = vec![0.0f32; hidden];

        no_grad(|| {
            fused_qkv_decode_infer_into(
                &input_f32, &q_f32, &k_f32, &v_f32, &mut q_ref, &mut k_ref, &mut v_ref,
            );
            fused_qkv_decode_infer_into(
                &input_bf16,
                &q_bf16,
                &k_bf16,
                &v_bf16,
                &mut q_out,
                &mut k_out,
                &mut v_out,
            );
        });

        assert_close(&q_ref, &q_out, 1e-4);
        assert_close(&k_ref, &k_out, 1e-4);
        assert_close(&v_ref, &v_out, 1e-4);
    }

    #[test]
    fn fused_qkv_decode_infer_bf16_matches_quantized_reference() {
        let hidden = 8usize;
        let x = sample_f32(hidden);
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let x_q = quantize_bf16(&x);
        let q_q = quantize_bf16(&q);
        let k_q = quantize_bf16(&k);
        let v_q = quantize_bf16(&v);

        let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
        let input_bf16 = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
        let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
        let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
        let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
        let q_bf16 = make_tensor(&[hidden, hidden], q.clone(), DType::BF16);
        let k_bf16 = make_tensor(&[hidden, hidden], k.clone(), DType::BF16);
        let v_bf16 = make_tensor(&[hidden, hidden], v.clone(), DType::BF16);

        let (q_ref, k_ref, v_ref) =
            no_grad(|| fused_qkv_decode_infer(&input_f32, &q_f32, &k_f32, &v_f32, 1, 1));
        let (q_out, k_out, v_out) =
            no_grad(|| fused_qkv_decode_infer(&input_bf16, &q_bf16, &k_bf16, &v_bf16, 1, 1));

        assert_close(q_ref.as_slice().unwrap(), q_out.as_slice().unwrap(), 1e-4);
        assert_close(k_ref.as_slice().unwrap(), k_out.as_slice().unwrap(), 1e-4);
        assert_close(v_ref.as_slice().unwrap(), v_out.as_slice().unwrap(), 1e-4);
    }

    #[test]
    fn fused_qkv_decode_infer_batch_bf16_matches_quantized_reference() {
        let batch = 2usize;
        let hidden = 8usize;
        let x = sample_f32(batch * hidden)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i / hidden) as f32 * 0.1)
            .collect::<Vec<_>>();
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let x_q = quantize_bf16(&x);
        let q_q = quantize_bf16(&q);
        let k_q = quantize_bf16(&k);
        let v_q = quantize_bf16(&v);

        let input_f32 = make_tensor(&[batch, 1, hidden], x_q.clone(), DType::F32);
        let input_bf16 = make_tensor(&[batch, 1, hidden], x.clone(), DType::BF16);
        let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
        let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
        let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
        let q_bf16 = make_tensor(&[hidden, hidden], q.clone(), DType::BF16);
        let k_bf16 = make_tensor(&[hidden, hidden], k.clone(), DType::BF16);
        let v_bf16 = make_tensor(&[hidden, hidden], v.clone(), DType::BF16);

        let (q_ref, k_ref, v_ref) =
            no_grad(|| fused_qkv_decode_infer(&input_f32, &q_f32, &k_f32, &v_f32, 1, 1));
        let (q_out, k_out, v_out) =
            no_grad(|| fused_qkv_decode_infer(&input_bf16, &q_bf16, &k_bf16, &v_bf16, 1, 1));

        assert_close(q_ref.as_slice().unwrap(), q_out.as_slice().unwrap(), 1e-4);
        assert_close(k_ref.as_slice().unwrap(), k_out.as_slice().unwrap(), 1e-4);
        assert_close(v_ref.as_slice().unwrap(), v_out.as_slice().unwrap(), 1e-4);
    }

    #[test]
    fn fused_qkv_decode_infer_batch_accepts_non_contiguous_input() {
        let batch = 2usize;
        let hidden = 8usize;
        let x = sample_f32(batch * hidden)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i / hidden) as f32 * 0.2)
            .collect::<Vec<_>>();
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let input_contig = make_tensor(&[batch, 1, hidden], x.clone(), DType::F32);
        let input_non_contig = Tensor::from_array_no_grad(
            Array3::from_shape_vec((1, batch, hidden), x)
                .expect("input shape")
                .permuted_axes([1, 0, 2])
                .into_dyn(),
        );
        let q_f32 = make_tensor(&[hidden, hidden], q, DType::F32);
        let k_f32 = make_tensor(&[hidden, hidden], k, DType::F32);
        let v_f32 = make_tensor(&[hidden, hidden], v, DType::F32);

        let (q_ref, k_ref, v_ref) =
            no_grad(|| fused_qkv_decode_infer(&input_contig, &q_f32, &k_f32, &v_f32, 1, 1));
        let (q_out, k_out, v_out) =
            no_grad(|| fused_qkv_decode_infer(&input_non_contig, &q_f32, &k_f32, &v_f32, 1, 1));

        assert_close(q_ref.as_slice().unwrap(), q_out.as_slice().unwrap(), 1e-5);
        assert_close(k_ref.as_slice().unwrap(), k_out.as_slice().unwrap(), 1e-5);
        assert_close(v_ref.as_slice().unwrap(), v_out.as_slice().unwrap(), 1e-5);
    }

    #[test]
    fn fused_qkv_f16_matches_quantized_reference() {
        let hidden = 8usize;
        let x = sample_f32(hidden);
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let x_q = quantize_f16(&x);
        let q_q = quantize_f16(&q);
        let k_q = quantize_f16(&k);
        let v_q = quantize_f16(&v);

        let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
        let input_f16 = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
        let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
        let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
        let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
        let q_f16 = make_tensor(&[hidden, hidden], q.clone(), DType::F16);
        let k_f16 = make_tensor(&[hidden, hidden], k.clone(), DType::F16);
        let v_f16 = make_tensor(&[hidden, hidden], v.clone(), DType::F16);

        let mut q_ref = vec![0.0f32; hidden];
        let mut k_ref = vec![0.0f32; hidden];
        let mut v_ref = vec![0.0f32; hidden];
        let mut q_out = vec![0.0f32; hidden];
        let mut k_out = vec![0.0f32; hidden];
        let mut v_out = vec![0.0f32; hidden];

        no_grad(|| {
            fused_qkv_decode_infer_into(
                &input_f32, &q_f32, &k_f32, &v_f32, &mut q_ref, &mut k_ref, &mut v_ref,
            );
            fused_qkv_decode_infer_into(
                &input_f16, &q_f16, &k_f16, &v_f16, &mut q_out, &mut k_out, &mut v_out,
            );
        });

        assert_close(&q_ref, &q_out, 1e-4);
        assert_close(&k_ref, &k_out, 1e-4);
        assert_close(&v_ref, &v_out, 1e-4);
    }

    #[test]
    fn fused_gateup_bf16_matches_quantized_reference() {
        let hidden = 8usize;
        let inter = 12usize;
        let x = sample_f32(hidden);
        let gate = sample_f32(inter * hidden);
        let up = sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect::<Vec<_>>();

        let x_q = quantize_bf16(&x);
        let gate_q = quantize_bf16(&gate);
        let up_q = quantize_bf16(&up);

        let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
        let input_bf16 = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
        let gate_f32 = make_tensor(&[inter, hidden], gate_q.clone(), DType::F32);
        let up_f32 = make_tensor(&[inter, hidden], up_q.clone(), DType::F32);
        let gate_bf16 = make_tensor(&[inter, hidden], gate.clone(), DType::BF16);
        let up_bf16 = make_tensor(&[inter, hidden], up.clone(), DType::BF16);

        let mut ref_out = vec![0.0f32; inter];
        let mut out = vec![0.0f32; inter];

        no_grad(|| {
            fused_gate_up_silu_infer_into(&input_f32, &gate_f32, &up_f32, &mut ref_out);
            fused_gate_up_silu_infer_into(&input_bf16, &gate_bf16, &up_bf16, &mut out);
        });

        assert_close(&ref_out, &out, 1e-3);
    }

    #[test]
    fn fused_gateup_f16_matches_quantized_reference() {
        let hidden = 8usize;
        let inter = 12usize;
        let x = sample_f32(hidden);
        let gate = sample_f32(inter * hidden);
        let up = sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect::<Vec<_>>();

        let x_q = quantize_f16(&x);
        let gate_q = quantize_f16(&gate);
        let up_q = quantize_f16(&up);

        let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
        let input_f16 = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
        let gate_f32 = make_tensor(&[inter, hidden], gate_q.clone(), DType::F32);
        let up_f32 = make_tensor(&[inter, hidden], up_q.clone(), DType::F32);
        let gate_f16 = make_tensor(&[inter, hidden], gate.clone(), DType::F16);
        let up_f16 = make_tensor(&[inter, hidden], up.clone(), DType::F16);

        let mut ref_out = vec![0.0f32; inter];
        let mut out = vec![0.0f32; inter];

        no_grad(|| {
            fused_gate_up_silu_infer_into(&input_f32, &gate_f32, &up_f32, &mut ref_out);
            fused_gate_up_silu_infer_into(&input_f16, &gate_f16, &up_f16, &mut out);
        });

        assert_close(&ref_out, &out, 1e-3);
    }

    #[test]
    fn fused_gateup_batch_accepts_non_contiguous_input() {
        let batch = 2usize;
        let hidden = 8usize;
        let inter = 12usize;
        let x = sample_f32(batch * hidden)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i / hidden) as f32 * 0.15)
            .collect::<Vec<_>>();
        let gate = sample_f32(inter * hidden);
        let up = sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect::<Vec<_>>();

        let input_contig = make_tensor(&[batch, 1, hidden], x.clone(), DType::F32);
        let input_non_contig = Tensor::from_array_no_grad(
            Array3::from_shape_vec((1, batch, hidden), x)
                .expect("input shape")
                .permuted_axes([1, 0, 2])
                .into_dyn(),
        );
        let gate_f32 = make_tensor(&[inter, hidden], gate, DType::F32);
        let up_f32 = make_tensor(&[inter, hidden], up, DType::F32);

        let ref_out = no_grad(|| fused_gate_up_silu_infer(&input_contig, &gate_f32, &up_f32));
        let out = no_grad(|| fused_gate_up_silu_infer(&input_non_contig, &gate_f32, &up_f32));

        let ref_vals = ref_out.data_ref().iter().copied().collect::<Vec<_>>();
        let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_close(&ref_vals, &out_vals, 1e-5);
    }

    #[test]
    fn fused_qkv_i8_matches_quantized_reference() {
        let hidden = 8usize;
        let x = sample_f32(hidden);
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let q_q = quantize_i8(&[hidden, hidden], &q);
        let k_q = quantize_i8(&[hidden, hidden], &k);
        let v_q = quantize_i8(&[hidden, hidden], &v);

        let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F32);
        let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
        let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
        let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
        let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
        let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
        let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

        let mut q_expected = vec![0.0f32; hidden];
        let mut k_expected = vec![0.0f32; hidden];
        let mut v_expected = vec![0.0f32; hidden];
        let mut q_out = vec![0.0f32; hidden];
        let mut k_out = vec![0.0f32; hidden];
        let mut v_out = vec![0.0f32; hidden];

        no_grad(|| {
            fused_qkv_decode_infer_into(
                &input,
                &q_ref,
                &k_ref,
                &v_ref,
                &mut q_expected,
                &mut k_expected,
                &mut v_expected,
            );
            fused_qkv_decode_infer_into(
                &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
            );
        });

        assert_close(&q_expected, &q_out, 1e-5);
        assert_close(&k_expected, &k_out, 1e-5);
        assert_close(&v_expected, &v_out, 1e-5);
    }

    #[test]
    fn fused_gateup_i8_matches_quantized_reference() {
        let hidden = 8usize;
        let inter = 12usize;
        let x = sample_f32(hidden);
        let gate = sample_f32(inter * hidden);
        let up = sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect::<Vec<_>>();

        let gate_q = quantize_i8(&[inter, hidden], &gate);
        let up_q = quantize_i8(&[inter, hidden], &up);

        let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F32);
        let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
        let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
        let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
        let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

        let mut ref_out = vec![0.0f32; inter];
        let mut out = vec![0.0f32; inter];

        no_grad(|| {
            fused_gate_up_silu_infer_into(&input, &gate_ref, &up_ref, &mut ref_out);
            fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
        });

        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn fused_qkv_bf16_input_i8_matches_quantized_reference() {
        let hidden = 8usize;
        let x = sample_f32(hidden);
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let q_q = quantize_i8(&[hidden, hidden], &q);
        let k_q = quantize_i8(&[hidden, hidden], &k);
        let v_q = quantize_i8(&[hidden, hidden], &v);

        let input = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
        let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
        let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
        let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
        let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
        let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
        let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

        let mut q_expected = vec![0.0f32; hidden];
        let mut k_expected = vec![0.0f32; hidden];
        let mut v_expected = vec![0.0f32; hidden];
        let mut q_out = vec![0.0f32; hidden];
        let mut k_out = vec![0.0f32; hidden];
        let mut v_out = vec![0.0f32; hidden];

        no_grad(|| {
            fused_qkv_decode_infer_into(
                &input,
                &q_ref,
                &k_ref,
                &v_ref,
                &mut q_expected,
                &mut k_expected,
                &mut v_expected,
            );
            fused_qkv_decode_infer_into(
                &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
            );
        });

        assert_close(&q_expected, &q_out, 1e-5);
        assert_close(&k_expected, &k_out, 1e-5);
        assert_close(&v_expected, &v_out, 1e-5);
    }

    #[test]
    fn fused_qkv_f16_input_i8_matches_quantized_reference() {
        let hidden = 8usize;
        let x = sample_f32(hidden);
        let q = sample_f32(hidden * hidden);
        let k = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let v = sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect::<Vec<_>>();

        let q_q = quantize_i8(&[hidden, hidden], &q);
        let k_q = quantize_i8(&[hidden, hidden], &k);
        let v_q = quantize_i8(&[hidden, hidden], &v);

        let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
        let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
        let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
        let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
        let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
        let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
        let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

        let mut q_expected = vec![0.0f32; hidden];
        let mut k_expected = vec![0.0f32; hidden];
        let mut v_expected = vec![0.0f32; hidden];
        let mut q_out = vec![0.0f32; hidden];
        let mut k_out = vec![0.0f32; hidden];
        let mut v_out = vec![0.0f32; hidden];

        no_grad(|| {
            fused_qkv_decode_infer_into(
                &input,
                &q_ref,
                &k_ref,
                &v_ref,
                &mut q_expected,
                &mut k_expected,
                &mut v_expected,
            );
            fused_qkv_decode_infer_into(
                &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
            );
        });

        assert_close(&q_expected, &q_out, 1e-5);
        assert_close(&k_expected, &k_out, 1e-5);
        assert_close(&v_expected, &v_out, 1e-5);
    }

    #[test]
    fn fused_gateup_bf16_input_i8_matches_quantized_reference() {
        let hidden = 8usize;
        let inter = 12usize;
        let x = sample_f32(hidden);
        let gate = sample_f32(inter * hidden);
        let up = sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect::<Vec<_>>();

        let gate_q = quantize_i8(&[inter, hidden], &gate);
        let up_q = quantize_i8(&[inter, hidden], &up);

        let input = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
        let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
        let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
        let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
        let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

        let mut ref_out = vec![0.0f32; inter];
        let mut out = vec![0.0f32; inter];

        no_grad(|| {
            fused_gate_up_silu_infer_into(&input, &gate_ref, &up_ref, &mut ref_out);
            fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
        });

        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn fused_gateup_f16_input_i8_matches_quantized_reference() {
        let hidden = 8usize;
        let inter = 12usize;
        let x = sample_f32(hidden);
        let gate = sample_f32(inter * hidden);
        let up = sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect::<Vec<_>>();

        let gate_q = quantize_i8(&[inter, hidden], &gate);
        let up_q = quantize_i8(&[inter, hidden], &up);

        let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
        let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
        let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
        let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
        let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

        let mut ref_out = vec![0.0f32; inter];
        let mut out = vec![0.0f32; inter];

        no_grad(|| {
            fused_gate_up_silu_infer_into(&input, &gate_ref, &up_ref, &mut ref_out);
            fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
        });

        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn fused_softmax_no_grad_preserves_bf16_dtype() {
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

        let ref_out = no_grad(|| fused_softmax(&input_f32, 1.0, false));
        let out = no_grad(|| fused_softmax(&input_bf16, 1.0, false));

        assert_eq!(input_bf16.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => {
                panic!("bf16 fused_softmax output should stay bf16 in no-grad")
            }
            TensorStorageView::F32(_) => {
                panic!("bf16 fused_softmax output should stay bf16 in no-grad")
            }
        });
    }

    #[test]
    fn fused_softmax_bf16_matches_quantized_reference() {
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

        let ref_out = no_grad(|| fused_softmax(&input_f32, 0.75, true));
        let out = no_grad(|| fused_softmax(&input_bf16, 0.75, true));

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => {
                panic!("bf16 fused_softmax output should stay bf16 in no-grad")
            }
            TensorStorageView::F32(_) => {
                panic!("bf16 fused_softmax output should stay bf16 in no-grad")
            }
        });
    }
}
