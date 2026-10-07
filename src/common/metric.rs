//! Step lengths between stencil neighbours, for the kernels whose result
//! depends on physical distances (`topology::steepest`, `topology::dijkstra`).
//! No PyO3.

use crate::common::nd::{MAX_NDIM, code_offsets, code_space};

/// Euclidean step lengths between stencil neighbours. `unit` is index
/// coordinates; `from_axes` uses the per-axis coordinate arrays, so steps
/// may differ between axes and along an axis.
pub struct StepMetric {
    ndim: usize,
    /// `forward[a][i] = x_a[i + 1] − x_a[i]`; `None` means unit steps on every axis.
    forward: Option<Vec<Vec<f64>>>,
    /// Offsets per direction code, decoded once (3ᴺ entries).
    offsets: Vec<[i8; MAX_NDIM]>,
}

impl StepMetric {
    pub fn unit(ndim: usize) -> Self {
        Self { ndim, forward: None, offsets: decode_all(ndim) }
    }

    /// `axes[a]` holds the strictly increasing coordinates of axis `a`
    /// (validated by the caller through `validate::check_axes`).
    pub fn from_axes(axes: &[Vec<f64>]) -> Self {
        let forward = axes.iter().map(|x| x.windows(2).map(|w| w[1] - w[0]).collect()).collect();
        Self { ndim: axes.len(), forward: Some(forward), offsets: decode_all(axes.len()) }
    }

    /// Length of the step from the cell with N-D index `idx` in direction
    /// `code`. The caller guarantees the step stays in bounds.
    #[inline]
    pub fn step_length(&self, idx: &[usize], code: u16) -> f64 {
        let off = &self.offsets[code as usize];
        let mut sum = 0.0;
        for axis in 0..self.ndim {
            let d = match (off[axis], &self.forward) {
                (0, _) => continue,
                (_, None) => 1.0,
                (1, Some(f)) => f[axis][idx[axis]],
                (_, Some(f)) => f[axis][idx[axis] - 1],
            };
            sum += d * d;
        }
        sum.sqrt()
    }
}

fn decode_all(ndim: usize) -> Vec<[i8; MAX_NDIM]> {
    (0..code_space(ndim)).map(|code| code_offsets(code, ndim)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::nd::{Stencil, compute_strides, index_to_linear, linear_to_coords};

    fn lengths(metric: &StepMetric, shape: &[usize], centre: &[usize], stencil: Stencil) -> Vec<f64> {
        let strides = compute_strides(shape);
        let lin = index_to_linear(centre, &strides);
        let idx = linear_to_coords(lin, &strides);
        let mut nbrs = Vec::new();
        stencil.neighbors_with_codes(lin, shape, &strides, &mut nbrs);
        nbrs.iter().map(|&(_, code)| metric.step_length(&idx[..shape.len()], code)).collect()
    }

    #[test]
    fn unit_axis_steps_are_one_and_diagonals_root_k() {
        let m = StepMetric::unit(3);
        assert!(lengths(&m, &[3, 3, 3], &[1, 1, 1], Stencil::VonNeumann).iter().all(|&l| l == 1.0));
        let moore = lengths(&m, &[3, 3, 3], &[1, 1, 1], Stencil::Moore);
        assert_eq!(moore.len(), 26);
        let count = |v: f64| moore.iter().filter(|&&l| (l - v).abs() < 1e-15).count();
        assert_eq!((count(1.0), count(2f64.sqrt()), count(3f64.sqrt())), (6, 12, 8));
    }

    #[test]
    fn variable_steps_use_the_interval_being_crossed() {
        // axis 0: 0, 1, 4 (steps 1 then 3); axis 1: 0, 2, 2.5 (steps 2 then 0.5)
        let m = StepMetric::from_axes(&[vec![0.0, 1.0, 4.0], vec![0.0, 2.0, 2.5]]);
        // von Neumann order at (1, 1): axis 0 −, axis 0 +, axis 1 −, axis 1 +
        assert_eq!(lengths(&m, &[3, 3], &[1, 1], Stencil::VonNeumann), vec![1.0, 3.0, 2.0, 0.5]);
        // Moore order: odometer over {−1,0,1}², last axis fastest, centre skipped
        let got = lengths(&m, &[3, 3], &[1, 1], Stencil::Moore);
        let want = [(1.0f64, 2.0f64), (1.0, 0.0), (1.0, 0.5), (0.0, 2.0), (0.0, 0.5), (3.0, 2.0), (3.0, 0.0), (3.0, 0.5)];
        assert_eq!(got.len(), want.len());
        for (g, (a, b)) in got.iter().zip(want) {
            assert!((g - (a * a + b * b).sqrt()).abs() < 1e-15, "{g} vs {a},{b}");
        }
    }

    #[test]
    fn length_one_axis_never_steps() {
        let m = StepMetric::from_axes(&[vec![7.0], vec![0.0, 1.0, 3.0]]);
        assert_eq!(lengths(&m, &[1, 3], &[0, 1], Stencil::Moore), vec![1.0, 2.0]);
    }
}
