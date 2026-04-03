use crate::arch;

#[inline]
fn dot_len_matches(x: &[f32], row: &[i8]) -> bool {
    x.len() == row.len()
}

#[inline]
fn dot2_len_matches(x: &[f32], row0: &[i8], row1: &[i8]) -> bool {
    x.len() == row0.len() && x.len() == row1.len()
}

#[inline]
fn dot3_len_matches(x: &[f32], row0: &[i8], row1: &[i8], row2: &[i8]) -> bool {
    x.len() == row0.len() && x.len() == row1.len() && x.len() == row2.len()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int8KernelBackend {
    Portable,
    Arm64Neon,
    X86Avx2,
}

#[inline]
pub fn active_int8_backend() -> Int8KernelBackend {
    if arch::arm64_i8_kernel_runtime_available() {
        Int8KernelBackend::Arm64Neon
    } else if arch::x86_i8_kernel_runtime_available() {
        Int8KernelBackend::X86Avx2
    } else {
        Int8KernelBackend::Portable
    }
}

#[inline]
pub fn active_int8_backend_name() -> &'static str {
    match active_int8_backend() {
        Int8KernelBackend::Portable => "portable",
        Int8KernelBackend::Arm64Neon => "arm64-neon",
        Int8KernelBackend::X86Avx2 => "x86-avx2",
    }
}

#[inline]
pub fn dot_f32_i8_arch(_x: &[f32], _row: &[i8], _scale: f32) -> Option<f32> {
    if !dot_len_matches(_x, _row) {
        return None;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        Int8KernelBackend::Arm64Neon => Some(unsafe { dot_f32_i8_arm64_neon(_x, _row, _scale) }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => Some(unsafe { dot_f32_i8_x86_avx2(_x, _row, _scale) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot2_f32_i8_arch(
    _x: &[f32],
    _row0: &[i8],
    _scale0: f32,
    _row1: &[i8],
    _scale1: f32,
) -> Option<(f32, f32)> {
    if !dot2_len_matches(_x, _row0, _row1) {
        return None;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        Int8KernelBackend::Arm64Neon => Some(unsafe {
            (
                dot_f32_i8_arm64_neon(_x, _row0, _scale0),
                dot_f32_i8_arm64_neon(_x, _row1, _scale1),
            )
        }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            Some(unsafe { dot2_f32_i8_x86_avx2(_x, _row0, _scale0, _row1, _scale1) })
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

pub fn dot3_f32_i8_arch(
    _x: &[f32],
    _row0: &[i8],
    _scale0: f32,
    _row1: &[i8],
    _scale1: f32,
    _row2: &[i8],
    _scale2: f32,
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches(_x, _row0, _row1, _row2) {
        return None;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        Int8KernelBackend::Arm64Neon => Some(unsafe {
            (
                dot_f32_i8_arm64_neon(_x, _row0, _scale0),
                dot_f32_i8_arm64_neon(_x, _row1, _scale1),
                dot_f32_i8_arm64_neon(_x, _row2, _scale2),
            )
        }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => Some(unsafe {
            dot3_f32_i8_x86_avx2(_x, _row0, _scale0, _row1, _scale1, _row2, _scale2)
        }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_i8_arm64_neon(x: &[f32], row: &[i8], scale: f32) -> f32 {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let row8 = unsafe { vld1_s8(row.as_ptr().add(kk)) };
        let row16 = vmovl_s8(row8);
        let row_lo_i32 = vmovl_s16(vget_low_s16(row16));
        let row_hi_i32 = vmovl_s16(vget_high_s16(row16));
        let row_lo = vcvtq_f32_s32(row_lo_i32);
        let row_hi = vcvtq_f32_s32(row_hi_i32);
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        acc0 = vfmaq_f32(acc0, row_lo, x_lo);
        acc1 = vfmaq_f32(acc1, row_hi, x_hi);
        kk += 8;
    }

    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    while kk < k_dim {
        sum += row[kk] as f32 * x[kk];
        kk += 1;
    }
    sum * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_x86_avx2(x: &[f32], row: &[i8], scale: f32) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let row_chunk = unsafe { _mm_loadu_si128(row.as_ptr().add(kk) as *const __m128i) };
        let row_lo_i32 = _mm256_cvtepi8_epi32(row_chunk);
        let row_hi_i32 = _mm256_cvtepi8_epi32(_mm_srli_si128(row_chunk, 8));
        let row_lo = _mm256_cvtepi32_ps(row_lo_i32);
        let row_hi = _mm256_cvtepi32_ps(row_hi_i32);
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let row_chunk = unsafe { _mm_loadl_epi64(row.as_ptr().add(kk) as *const __m128i) };
        let row_i32 = _mm256_cvtepi8_epi32(row_chunk);
        let row_f32 = _mm256_cvtepi32_ps(row_i32);
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_f32, x_chunk, acc0);
        kk += 8;
    }

    let mut buf0 = [0.0f32; 8];
    let mut buf1 = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
    }

    let mut sum: f32 = buf0.iter().sum::<f32>() + buf1.iter().sum::<f32>();
    while kk < k_dim {
        sum += row[kk] as f32 * x[kk];
        kk += 1;
    }
    sum * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2_f32_i8_x86_avx2(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_chunk = unsafe { _mm_loadu_si128(row0.as_ptr().add(kk) as *const __m128i) };
        let row1_chunk = unsafe { _mm_loadu_si128(row1.as_ptr().add(kk) as *const __m128i) };
        let row0_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row0_chunk));
        let row0_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(row0_chunk, 8)));
        let row1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row1_chunk));
        let row1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(row1_chunk, 8)));
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { _mm_loadl_epi64(row0.as_ptr().add(kk) as *const __m128i) };
        let row1_chunk = unsafe { _mm_loadl_epi64(row1.as_ptr().add(kk) as *const __m128i) };
        let row0_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row0_chunk));
        let row1_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row1_chunk));
        acc00 = _mm256_fmadd_ps(row0_f32, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_f32, x_chunk, acc10);
        kk += 8;
    }

    let mut buf00 = [0.0f32; 8];
    let mut buf01 = [0.0f32; 8];
    let mut buf10 = [0.0f32; 8];
    let mut buf11 = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf00.as_mut_ptr(), acc00);
        _mm256_storeu_ps(buf01.as_mut_ptr(), acc01);
        _mm256_storeu_ps(buf10.as_mut_ptr(), acc10);
        _mm256_storeu_ps(buf11.as_mut_ptr(), acc11);
    }

    let mut sum0 = buf00.iter().sum::<f32>() + buf01.iter().sum::<f32>();
    let mut sum1 = buf10.iter().sum::<f32>() + buf11.iter().sum::<f32>();
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot3_f32_i8_x86_avx2(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_chunk = unsafe { _mm_loadu_si128(row0.as_ptr().add(kk) as *const __m128i) };
        let row1_chunk = unsafe { _mm_loadu_si128(row1.as_ptr().add(kk) as *const __m128i) };
        let row2_chunk = unsafe { _mm_loadu_si128(row2.as_ptr().add(kk) as *const __m128i) };
        let row0_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row0_chunk));
        let row0_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(row0_chunk, 8)));
        let row1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row1_chunk));
        let row1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(row1_chunk, 8)));
        let row2_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row2_chunk));
        let row2_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(row2_chunk, 8)));
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm256_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm256_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { _mm_loadl_epi64(row0.as_ptr().add(kk) as *const __m128i) };
        let row1_chunk = unsafe { _mm_loadl_epi64(row1.as_ptr().add(kk) as *const __m128i) };
        let row2_chunk = unsafe { _mm_loadl_epi64(row2.as_ptr().add(kk) as *const __m128i) };
        let row0_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row0_chunk));
        let row1_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row1_chunk));
        let row2_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(row2_chunk));
        acc00 = _mm256_fmadd_ps(row0_f32, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_f32, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_f32, x_chunk, acc20);
        kk += 8;
    }

    let mut buf00 = [0.0f32; 8];
    let mut buf01 = [0.0f32; 8];
    let mut buf10 = [0.0f32; 8];
    let mut buf11 = [0.0f32; 8];
    let mut buf20 = [0.0f32; 8];
    let mut buf21 = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf00.as_mut_ptr(), acc00);
        _mm256_storeu_ps(buf01.as_mut_ptr(), acc01);
        _mm256_storeu_ps(buf10.as_mut_ptr(), acc10);
        _mm256_storeu_ps(buf11.as_mut_ptr(), acc11);
        _mm256_storeu_ps(buf20.as_mut_ptr(), acc20);
        _mm256_storeu_ps(buf21.as_mut_ptr(), acc21);
    }

    let mut sum0 = buf00.iter().sum::<f32>() + buf01.iter().sum::<f32>();
    let mut sum1 = buf10.iter().sum::<f32>() + buf11.iter().sum::<f32>();
    let mut sum2 = buf20.iter().sum::<f32>() + buf21.iter().sum::<f32>();
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        sum2 += row2[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1, sum2 * scale2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_matches_backend_enum() {
        let name = active_int8_backend_name();
        match active_int8_backend() {
            Int8KernelBackend::Portable => assert_eq!(name, "portable"),
            Int8KernelBackend::Arm64Neon => assert_eq!(name, "arm64-neon"),
            Int8KernelBackend::X86Avx2 => assert_eq!(name, "x86-avx2"),
        }
    }

    #[test]
    fn architecture_dispatch_consistency() {
        let x = [0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 3.0];
        let row0 = [3i8, -2, 5, 1, -4, 6, -1, 2];
        let row1 = [-1i8, 4, -3, 2, 5, -2, 7, -6];
        let row2 = [2i8, 1, -2, 3, -5, 4, 1, -3];
        let single = dot_f32_i8_arch(&x, &row0, 0.125);
        let triple = dot3_f32_i8_arch(&x, &row0, 0.125, &row1, 0.25, &row2, 0.5);

        if active_int8_backend() == Int8KernelBackend::Portable {
            assert!(single.is_none());
            assert!(triple.is_none());
        } else {
            assert!(single.is_some());
            assert!(triple.is_some());
        }
    }

    #[test]
    fn length_mismatch_disables_arch_fast_path() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let short = [1i8, 2];

        assert!(dot_f32_i8_arch(&x, &short, 0.5).is_none());
        assert!(dot2_f32_i8_arch(&x, &short, 0.5, &short, 0.25).is_none());
        assert!(dot3_f32_i8_arch(&x, &short, 0.5, &short, 0.25, &short, 0.125).is_none());
    }
}
