use crate::autograd::{ArcArray, IxDyn, Tensor, TensorData, TensorStorageOwned, is_no_grad};
use ndarray::{Axis, Slice};
use std::cell::RefCell;
use std::rc::Rc;

// 说明：Tensor 的 data 现在是 ArcArray（共享底层 + stride）。
// 因此 reshape/permute 在 stride 兼容时可以做到零拷贝（只改元数据）。

fn reshape_shared<T: Clone>(data: ArcArray<T, IxDyn>, new_shape: &[usize]) -> ArcArray<T, IxDyn> {
    match data.clone().into_shape(new_shape.to_vec()) {
        Ok(a) => a.into_dyn(),
        Err(_) => data
            .to_owned()
            .as_standard_layout()
            .into_owned()
            .into_shape(new_shape.to_vec())
            .expect("Reshape failed: Total element count mismatch")
            .into_dyn()
            .into_shared(),
    }
}

fn validate_permute_axes(ndim: usize, axes: &[usize]) {
    assert_eq!(
        axes.len(),
        ndim,
        "Permute axes length mismatch: got {}, expected {}",
        axes.len(),
        ndim
    );
    let mut seen = vec![false; ndim];
    for (i, &axis) in axes.iter().enumerate() {
        assert!(
            axis < ndim,
            "Permute axis {} out of bounds for ndim {}",
            axis,
            ndim
        );
        assert!(
            !seen[axis],
            "Permute axes must be unique, duplicate axis {} at position {}",
            axis, i
        );
        seen[axis] = true;
    }
}

fn validate_cat_inputs(tensors: &[Tensor], axis: usize) -> Vec<Vec<usize>> {
    let shapes = tensors.iter().map(Tensor::shape_vec).collect::<Vec<_>>();
    let ndim = shapes[0].len();
    assert!(
        axis < ndim,
        "Concat axis {} out of bounds for ndim {}",
        axis,
        ndim
    );
    for shape in shapes.iter().skip(1) {
        assert_eq!(
            shape.len(),
            ndim,
            "Concat ndim mismatch: expected {}, got {}",
            ndim,
            shape.len()
        );
    }
    for dim in 0..ndim {
        if dim == axis {
            continue;
        }
        let expected = shapes[0][dim];
        for shape in shapes.iter().skip(1) {
            assert_eq!(
                shape[dim], expected,
                "Concat shape mismatch on dim {}: expected {}, got {}",
                dim, expected, shape[dim]
            );
        }
    }
    shapes
}

pub fn reshape(input: &Tensor, shape: Vec<i32>) -> Tensor {
    assert!(!shape.is_empty(), "Reshape expects at least one dimension");

    let input_len = input.len();
    let mut infer_axis = None;
    let mut known_product = 1usize;
    let mut new_shape = Vec::with_capacity(shape.len());

    for (axis, &dim) in shape.iter().enumerate() {
        match dim {
            -1 => {
                assert!(
                    infer_axis.is_none(),
                    "Reshape only supports one inferred dimension (-1)"
                );
                infer_axis = Some(axis);
                new_shape.push(0);
            }
            d if d >= 0 => {
                let dim_usize = d as usize;
                known_product = known_product
                    .checked_mul(dim_usize)
                    .expect("Reshape dimension product overflow");
                new_shape.push(dim_usize);
            }
            _ => {
                panic!(
                    "Reshape dimension at axis {} must be >= -1, got {}",
                    axis, dim
                );
            }
        }
    }

    if let Some(axis) = infer_axis {
        assert!(
            known_product > 0,
            "Reshape cannot infer dimension when known product is zero"
        );
        assert!(
            input_len % known_product == 0,
            "Reshape inferred dimension mismatch: input elements {} not divisible by known product {}",
            input_len,
            known_product
        );
        new_shape[axis] = input_len / known_product;
    } else {
        let new_len = new_shape.iter().product::<usize>();
        assert_eq!(
            new_len, input_len,
            "Reshape failed: total element count mismatch (input {}, target {})",
            input_len, new_len
        );
    }

    if is_no_grad() || !input.requires_grad() {
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(data) => {
                Tensor::from_data_no_grad(reshape_shared(data, &new_shape))
            }
            TensorStorageOwned::F16(data) => {
                Tensor::from_f16_data_no_grad(reshape_shared(data, &new_shape))
            }
            TensorStorageOwned::BF16(data) => {
                Tensor::from_bf16_data_no_grad(reshape_shared(data, &new_shape))
            }
            TensorStorageOwned::I8(data, scale) => {
                Tensor::from_i8_data_no_grad(reshape_shared(data, &new_shape), scale)
            }
        };
    }

    // clone 仅增加 refcount
    let data: ArcArray<f32, IxDyn> = input.data_arc();
    let reshaped = reshape_shared(data, &new_shape);

    let input_clone = input.clone();
    Tensor(Rc::new(RefCell::new(TensorData {
        data: reshaped,
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
            let old_shape = input_clone.data_ref().shape().to_vec();
            let grad_contig = grad.as_standard_layout().into_owned();
            let grad_reshaped = grad_contig
                .into_shape(old_shape)
                .expect("Backward Reshape failed")
                .into_dyn();
            input_clone.add_grad(grad_reshaped);
        })),
        requires_grad: true,
    })))
}

pub fn permute(input: &Tensor, axes: Vec<usize>) -> Tensor {
    let ndim = input.ndim();
    validate_permute_axes(ndim, &axes);

    if is_no_grad() || !input.requires_grad() {
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(data) => {
                Tensor::from_data_no_grad(data.permuted_axes(axes.clone()).into_dyn())
            }
            TensorStorageOwned::F16(data) => {
                Tensor::from_f16_data_no_grad(data.permuted_axes(axes.clone()).into_dyn())
            }
            TensorStorageOwned::BF16(data) => {
                Tensor::from_bf16_data_no_grad(data.permuted_axes(axes.clone()).into_dyn())
            }
            TensorStorageOwned::I8(data, scale) => {
                Tensor::from_i8_data_no_grad(data.permuted_axes(axes.clone()).into_dyn(), scale)
            }
        };
    }

    let data: ArcArray<f32, IxDyn> = input.data_arc();
    let permuted: ArcArray<f32, IxDyn> = data.permuted_axes(axes.clone()).into_dyn();

    let input_clone = input.clone();
    let mut rev_axes = vec![0; axes.len()];
    for (i, &ax) in axes.iter().enumerate() {
        rev_axes[ax] = i;
    }

    Tensor(Rc::new(RefCell::new(TensorData {
        data: permuted,
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
            let grad_restored = grad.view().permuted_axes(rev_axes.clone()).to_owned();
            input_clone.add_grad(grad_restored);
        })),
        requires_grad: true,
    })))
}

pub fn cat(tensors: &[Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "Concat expects at least one tensor");
    let shapes = validate_cat_inputs(tensors, axis);

    if tensors.len() == 1 && (is_no_grad() || !tensors[0].requires_grad()) {
        return tensors[0].clone();
    }

    if is_no_grad() || tensors.iter().all(|t| !t.requires_grad()) {
        let storages = tensors
            .iter()
            .map(Tensor::native_storage_owned)
            .collect::<Vec<_>>();

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::F32(_)))
        {
            let arrays = storages
                .into_iter()
                .map(|storage| match storage {
                    TensorStorageOwned::F32(array) => array,
                    TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_)
                    | TensorStorageOwned::I8(_, _) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
            let result = ndarray::concatenate(Axis(axis), &views)
                .expect("Concat failed: shape mismatch or invalid axis")
                .into_dyn()
                .into_shared();
            return Tensor::from_data_no_grad(result);
        }

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::F16(_)))
        {
            let arrays = storages
                .into_iter()
                .map(|storage| match storage {
                    TensorStorageOwned::F16(array) => array,
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::BF16(_)
                    | TensorStorageOwned::I8(_, _) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
            let result = ndarray::concatenate(Axis(axis), &views)
                .expect("Concat failed: shape mismatch or invalid axis")
                .into_dyn()
                .into_shared();
            return Tensor::from_f16_data_no_grad(result);
        }

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::BF16(_)))
        {
            let arrays = storages
                .into_iter()
                .map(|storage| match storage {
                    TensorStorageOwned::BF16(array) => array,
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::I8(_, _) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
            let result = ndarray::concatenate(Axis(axis), &views)
                .expect("Concat failed: shape mismatch or invalid axis")
                .into_dyn()
                .into_shared();
            return Tensor::from_bf16_data_no_grad(result);
        }

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::I8(_, _)))
        {
            let scales = storages
                .iter()
                .map(|storage| match storage {
                    TensorStorageOwned::I8(_, scale) => *scale,
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            if scales.windows(2).all(|pair| pair[0] == pair[1]) {
                let scale = scales[0];
                let arrays = storages
                    .into_iter()
                    .map(|storage| match storage {
                        TensorStorageOwned::I8(array, _) => array,
                        TensorStorageOwned::F32(_)
                        | TensorStorageOwned::F16(_)
                        | TensorStorageOwned::BF16(_) => unreachable!("checked above"),
                    })
                    .collect::<Vec<_>>();
                let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
                let result = ndarray::concatenate(Axis(axis), &views)
                    .expect("Concat failed: shape mismatch or invalid axis")
                    .into_dyn()
                    .into_shared();
                return Tensor::from_i8_data_no_grad(result, scale);
            }
        }
    }

    // concatenate 本身会 materialize 结果；输入侧尽量保持零拷贝 view
    let arrays: Vec<_> = tensors.iter().map(|t| t.data_arc()).collect();
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let axis_obj = Axis(axis);
    let result = ndarray::concatenate(axis_obj, &views)
        .expect("Concat failed: shape mismatch or invalid axis")
        .into_dyn()
        .into_shared();

    let lengths: Vec<usize> = shapes.iter().map(|shape| shape[axis]).collect();
    let tensors_clone: Vec<Tensor> = tensors.to_vec();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result,
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        i8_scale: None,
        has_f32_data: true,
        storage_dtype: crate::precision::DType::F32,
        cache_dirty: false,
        is_parameter: false,
        grad: None,
        parents: tensors.to_vec(),
        backward_op: Some(std::rc::Rc::new(move |grad| {
            let mut start_idx = 0;
            for (i, &len) in lengths.iter().enumerate() {
                let slice_info = Slice::from(start_idx..start_idx + len);
                let sub_grad = grad.slice_axis(axis_obj, slice_info).to_owned().into_dyn();
                tensors_clone[i].add_grad(sub_grad);
                start_idx += len;
            }
        })),
        requires_grad: true,
    })))
}

pub fn slice_last_dim(input: &Tensor, start: usize, end: usize) -> Tensor {
    let input_shape = input.shape_vec();
    assert!(
        !input_shape.is_empty(),
        "slice_last_dim expects at least 1D input"
    );
    let last_dim = input_shape.len() - 1;
    let last_len = input_shape[last_dim];
    assert!(start <= end, "slice_last_dim expects start <= end");
    assert!(
        end <= last_len,
        "slice_last_dim end out of bounds: {} > {}",
        end,
        last_len
    );
    let axis = ndarray::Axis(last_dim);

    if (is_no_grad() || !input.requires_grad()) && start == 0 && end == last_len {
        return input.clone();
    }

    if is_no_grad() || !input.requires_grad() {
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(mut sliced) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_data_no_grad(sliced.into_dyn())
            }
            TensorStorageOwned::F16(mut sliced) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_f16_data_no_grad(sliced.into_dyn())
            }
            TensorStorageOwned::BF16(mut sliced) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_bf16_data_no_grad(sliced.into_dyn())
            }
            TensorStorageOwned::I8(mut sliced, scale) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_i8_data_no_grad(sliced.into_dyn(), scale)
            }
        };
    }

    let input_data = input.data_ref();
    let sliced = input_data
        .slice_axis(axis, ndarray::Slice::from(start..end))
        .to_owned()
        .into_dyn()
        .into_shared();

    let input_clone = input.clone();
    let full_shape = input_data.shape().to_vec();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: sliced,
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
            let mut full_grad = ndarray::Array::zeros(full_shape.clone());
            full_grad
                .slice_axis_mut(axis, ndarray::Slice::from(start..end))
                .assign(&grad);
            input_clone.add_grad(full_grad.into_dyn());
        })),
        requires_grad: true,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    use crate::precision::DType;
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
    fn reshape_supports_single_inferred_dimension() {
        let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
        let out = no_grad(|| reshape(&input, vec![2, -1]));
        assert_eq!(out.shape_vec(), vec![2, 12]);
    }

    #[test]
    #[should_panic(expected = "only supports one inferred dimension")]
    fn reshape_rejects_multiple_inferred_dimensions() {
        let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
        no_grad(|| {
            let _ = reshape(&input, vec![-1, -1]);
        });
    }

    #[test]
    #[should_panic(expected = "must be >= -1")]
    fn reshape_rejects_invalid_negative_dimension() {
        let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
        no_grad(|| {
            let _ = reshape(&input, vec![2, -2, 6]);
        });
    }

    #[test]
    #[should_panic(expected = "slice_last_dim end out of bounds")]
    fn slice_last_dim_rejects_out_of_bounds_end() {
        let input = make_tensor(&[2, 3], (0..6).map(|v| v as f32).collect(), DType::F32);
        no_grad(|| {
            let _ = slice_last_dim(&input, 0, 4);
        });
    }

    #[test]
    fn bf16_shape_ops_preserve_native_dtype_in_no_grad() {
        let input = make_tensor(
            &[2, 3, 4],
            (0..24).map(|v| v as f32 * 0.25).collect(),
            DType::BF16,
        );

        let reshaped = no_grad(|| reshape(&input, vec![2, -1]));
        let permuted = no_grad(|| permute(&input, vec![2, 0, 1]));
        let sliced = no_grad(|| slice_last_dim(&input, 1, 3));

        assert_eq!(reshaped.dtype(), DType::BF16);
        assert_eq!(permuted.dtype(), DType::BF16);
        assert_eq!(sliced.dtype(), DType::BF16);
    }

    #[test]
    fn bf16_cat_preserves_native_dtype_in_no_grad() {
        let lhs = make_tensor(&[1, 2], vec![0.0, 1.0], DType::BF16);
        let rhs = make_tensor(&[1, 2], vec![2.0, 3.0], DType::BF16);

        let out = no_grad(|| cat(&[lhs, rhs], 0));
        assert_eq!(out.shape_vec(), vec![2, 2]);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn i8_shape_ops_preserve_native_dtype_in_no_grad() {
        let input = make_tensor(
            &[2, 3, 4],
            (0..24).map(|v| v as f32 * 0.125 - 1.0).collect(),
            DType::I8,
        );

        let reshaped = no_grad(|| reshape(&input, vec![2, -1]));
        let permuted = no_grad(|| permute(&input, vec![2, 0, 1]));
        let sliced = no_grad(|| slice_last_dim(&input, 1, 3));
        let cat_out = no_grad(|| cat(&[input.clone(), input.clone()], 0));

        assert_eq!(reshaped.dtype(), DType::I8);
        assert_eq!(permuted.dtype(), DType::I8);
        assert_eq!(sliced.dtype(), DType::I8);
        assert_eq!(cat_out.dtype(), DType::I8);

        match reshaped.native_storage_owned() {
            TensorStorageOwned::I8(_, scale) => assert!(scale > 0.0),
            TensorStorageOwned::F32(_)
            | TensorStorageOwned::F16(_)
            | TensorStorageOwned::BF16(_) => {
                panic!("reshape should keep i8 storage")
            }
        }
    }

    #[test]
    #[should_panic(expected = "Permute axes must be unique")]
    fn permute_rejects_duplicate_axes() {
        let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
        no_grad(|| {
            let _ = permute(&input, vec![0, 1, 1]);
        });
    }
}
