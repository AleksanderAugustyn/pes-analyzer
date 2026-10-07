//! Steepest-descent path: from a cell, repeatedly move to the strictly
//! lower stencil neighbour with the largest slope (ΔE / Δs) until no
//! neighbour is lower. Pure Rust; the PyO3 wrapper is in `topology/mod.rs`.

use ndarray::ArrayViewD;

use crate::common::metric::StepMetric;
use crate::common::nd::{Stencil, compute_strides, linear_to_coords};
use crate::common::scalar::Scalar;

/// Path of linear indices from `start_lin` (which must be a valid, non-NaN
/// cell) to the first cell with no strictly lower stencil neighbour.
///
/// Each step goes to the candidate with the largest `(E_u − E_v) / Δs`;
/// equal slopes go to the smaller linear index. Energies decrease strictly,
/// so the walk terminates in at most V steps and never repeats a cell.
pub fn steepest_descent<T: Scalar>(
    energies: ArrayViewD<'_, T>,
    start_lin: usize,
    stencil: Stencil,
    metric: &StepMetric,
) -> Vec<usize> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let ndim = shape.len();
    let strides = compute_strides(&shape);
    let flat = energies.as_slice().expect("energies must be C-contiguous (checked by the wrapper)");
    let mut nbrs: Vec<(usize, u16)> = Vec::new();
    let mut path = vec![start_lin];
    let mut cur = start_lin;
    loop {
        let e_cur = flat[cur].to_f64();
        let idx = linear_to_coords(cur, &strides);
        stencil.neighbors_with_codes(cur, &shape, &strides, &mut nbrs);
        let mut best: Option<(f64, usize)> = None; // (slope, lin)
        for &(v, code) in &nbrs {
            let e_v = flat[v].to_f64();
            if !(e_v < e_cur) {
                continue; // not lower, or NaN
            }
            let slope = (e_cur - e_v) / metric.step_length(&idx[..ndim], code);
            let better = match best {
                None => true,
                Some((s, l)) => slope > s || (slope == s && v < l),
            };
            if better {
                best = Some((slope, v));
            }
        }
        match best {
            None => return path,
            Some((_, v)) => {
                path.push(v);
                cur = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD, IxDyn};

    fn grid<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    fn run(e: &ArrayD<f64>, start: usize, stencil: Stencil) -> Vec<usize> {
        steepest_descent(e.view(), start, stencil, &StepMetric::unit(e.ndim()))
    }

    #[test]
    fn chain_descends_to_the_end() {
        let e = grid([1, 5], vec![4.0, 3.0, 2.0, 1.0, 0.0]);
        assert_eq!(run(&e, 0, Stencil::Moore), vec![0, 1, 2, 3, 4]);
        assert_eq!(run(&e, 4, Stencil::Moore), vec![4]);
    }

    #[test]
    fn equal_slopes_go_to_the_smallest_linear_index() {
        let e = grid([3, 3], vec![5.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 0.0, 5.0]);
        assert_eq!(run(&e, 4, Stencil::VonNeumann), vec![4, 1]);
    }

    #[test]
    fn unequal_steps_change_the_choice() {
        let mut data = vec![20.0; 9];
        data[4] = 10.0;
        data[5] = 8.0;
        data[7] = 9.0;
        let e = grid([3, 3], data);
        assert_eq!(run(&e, 4, Stencil::VonNeumann), vec![4, 5]);
        let stretched = StepMetric::from_axes(&[vec![0.0, 1.0, 2.0], vec![0.0, 4.0, 8.0]]);
        assert_eq!(steepest_descent(e.view(), 4, Stencil::VonNeumann, &stretched), vec![4, 7]);
    }

    #[test]
    fn plateau_and_nan_neighbours_end_the_path() {
        let flat = grid([1, 3], vec![1.0, 1.0, 0.0]);
        assert_eq!(run(&flat, 0, Stencil::Moore), vec![0]);
        let n = f64::NAN;
        let walled = grid([3, 3], vec![n, n, n, n, 3.0, n, n, n, n]);
        assert_eq!(run(&walled, 4, Stencil::Moore), vec![4]);
    }

    #[test]
    fn moore_takes_the_steeper_diagonal() {
        let e = grid([2, 2], vec![10.0, 9.0, 20.0, 8.5]);
        assert_eq!(run(&e, 0, Stencil::Moore), vec![0, 3]);
        assert_eq!(run(&e, 0, Stencil::VonNeumann), vec![0, 1, 3]);
    }

    #[test]
    fn f32_matches_f64() {
        let data = vec![4.0, 3.0, 2.5, 2.0, 1.0, 0.0];
        let e64 = grid([2, 3], data.clone());
        let e32 = Array::from_shape_vec(IxDyn(&[2, 3]), data.iter().map(|&v| v as f32).collect()).unwrap();
        assert_eq!(
            steepest_descent(e32.view(), 0, Stencil::Moore, &StepMetric::unit(2)),
            steepest_descent(e64.view(), 0, Stencil::Moore, &StepMetric::unit(2))
        );
    }
}
