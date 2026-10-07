//! Shared search kernel for `find_least_action_path` and
//! `find_minimum_ascent_path`: Dijkstra over the stencil graph with keys
//! (sum of step weights, path length) compared lexicographically, so among
//! equal-weight paths the shortest wins. Sequential; 18 bytes per cell.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::ArrayViewD;
use rayon::prelude::*;

use crate::common::metric::StepMetric;
use crate::common::nd::{PARENT_NONE, Stencil, apply_code, compute_strides, linear_to_coords, reverse_code};
use crate::common::scalar::Scalar;

/// How the weight of a step u → v is computed from the input array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeRule {
    /// `½ (cost_u + cost_v) · Δs`: the trapezoid rule for ∫ cost ds.
    CostIntegral,
    /// `max(E_v − E_u, 0)`: descents are free, climbs cost what they gain.
    Ascent,
}

/// Where the search stops: one cell, or any `true` cell of a mask in C order.
pub enum Target<'a> {
    Cell(usize),
    Mask(&'a [bool]),
}

impl Target<'_> {
    #[inline]
    fn contains(&self, lin: usize) -> bool {
        match self {
            Target::Cell(c) => *c == lin,
            Target::Mask(m) => m[lin],
        }
    }
}

pub struct SearchResult {
    /// Linear indices from the start to the target reached.
    pub path: Vec<usize>,
    /// Sum of step weights at each path cell; `[0] == 0.0`, last = total.
    pub cumulative: Vec<f64>,
}

/// Heap entry ordered so that `BinaryHeap` (a max-heap) pops the smallest
/// `(action, length, lin)` first, which makes the pop order fully determined.
struct Entry {
    action: f64,
    length: f64,
    lin: u32,
}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.action
            .total_cmp(&self.action)
            .then_with(|| o.length.total_cmp(&self.length))
            .then_with(|| o.lin.cmp(&self.lin))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}

/// First violation of the input contract, checked in a fixed order so the
/// message never depends on the thread count: an infinite value anywhere,
/// then (cost rule only) a negative value. `NaN` is allowed: it is a wall.
pub fn field_violation<T: Scalar>(flat: &[T], rule: EdgeRule) -> Option<&'static str> {
    // `|&v|` binds `T` by value: on `&T`, `.to_f64()` would resolve to
    // `num_traits::ToPrimitive` (an `Option<f64>`), not to `Scalar::to_f64`.
    if flat.par_iter().any(|&v| Scalar::to_f64(v).is_infinite()) {
        return Some("an infinite");
    }
    if rule == EdgeRule::CostIntegral && flat.par_iter().any(|&v| Scalar::to_f64(v) < 0.0) {
        return Some("a negative");
    }
    None
}

/// Dijkstra from `start_lin` (a valid, non-NaN cell) until the first target
/// is popped. Keys are `(Σ weight, Σ Δs)` compared lexicographically; a cell
/// is updated only on a strictly smaller key, so among bit-equal keys the
/// first path found wins. Returns `None` when no target is reachable.
///
/// Memory: three grid-sized arrays (8 + 8 + 2 bytes per cell) plus the heap.
pub fn search<T: Scalar>(
    field: ArrayViewD<'_, T>,
    start_lin: usize,
    target: Target<'_>,
    rule: EdgeRule,
    stencil: Stencil,
    metric: &StepMetric,
) -> Option<SearchResult> {
    let shape: Vec<usize> = field.shape().to_vec();
    let ndim = shape.len();
    let strides = compute_strides(&shape);
    let flat = field.as_slice().expect("field must be C-contiguous (checked by the wrapper)");
    let n = flat.len();
    let mut action = vec![f64::INFINITY; n];
    let mut length = vec![f64::INFINITY; n];
    let mut back = vec![PARENT_NONE; n];
    let mut heap = BinaryHeap::new();
    let mut nbrs: Vec<(usize, u16)> = Vec::new();
    action[start_lin] = 0.0;
    length[start_lin] = 0.0;
    heap.push(Entry { action: 0.0, length: 0.0, lin: start_lin as u32 });
    while let Some(Entry { action: a_u, length: l_u, lin }) = heap.pop() {
        let u = lin as usize;
        if a_u > action[u] || (a_u == action[u] && l_u > length[u]) {
            continue; // stale entry: a better key was stored after this push
        }
        if target.contains(u) {
            return Some(rebuild(u, start_lin, &back, &action, &strides));
        }
        let f_u = flat[u].to_f64();
        let idx = linear_to_coords(u, &strides);
        stencil.neighbors_with_codes(u, &shape, &strides, &mut nbrs);
        for &(v, code) in &nbrs {
            let f_v = flat[v].to_f64();
            if f_v.is_nan() {
                continue; // wall
            }
            let ds = metric.step_length(&idx[..ndim], code);
            let w = match rule {
                EdgeRule::CostIntegral => 0.5 * (f_u + f_v) * ds,
                EdgeRule::Ascent => (f_v - f_u).max(0.0),
            };
            let a_v = a_u + w;
            let l_v = l_u + ds;
            if a_v < action[v] || (a_v == action[v] && l_v < length[v]) {
                action[v] = a_v;
                length[v] = l_v;
                back[v] = reverse_code(code, ndim);
                heap.push(Entry { action: a_v, length: l_v, lin: v as u32 });
            }
        }
    }
    None
}

fn rebuild(end: usize, start: usize, back: &[u16], action: &[f64], strides: &[usize]) -> SearchResult {
    let mut path = vec![end];
    let mut cur = end;
    while cur != start {
        cur = apply_code(cur, back[cur], strides);
        path.push(cur);
    }
    path.reverse();
    let cumulative = path.iter().map(|&l| action[l]).collect();
    SearchResult { path, cumulative }
}

#[cfg(test)]
mod tests {
    use super::*;
    use EdgeRule::{Ascent, CostIntegral};
    use Stencil::{Moore, VonNeumann};
    use ndarray::{Array, ArrayD, IxDyn};

    fn grid<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    fn run(f: &ArrayD<f64>, start: usize, target: Target<'_>, rule: EdgeRule, stencil: Stencil) -> Option<SearchResult> {
        search(f.view(), start, target, rule, stencil, &StepMetric::unit(f.ndim()))
    }

    #[test]
    fn cost_path_takes_the_cheap_detour() {
        let c = grid([3, 3], vec![1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 2.0, 2.0, 2.0]);
        let r = run(&c, 3, Target::Cell(5), CostIntegral, VonNeumann).unwrap();
        assert_eq!(r.path, vec![3, 0, 1, 2, 5]);
        assert_eq!(r.cumulative, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unit_cost_measures_length() {
        let c = grid([3, 4], vec![1.0; 12]);
        let vn = run(&c, 0, Target::Cell(11), CostIntegral, VonNeumann).unwrap();
        assert_eq!((*vn.cumulative.last().unwrap(), vn.path.len()), (5.0, 6));
        let moore = run(&c, 0, Target::Cell(11), CostIntegral, Moore).unwrap();
        assert!((moore.cumulative.last().unwrap() - (2.0 * 2f64.sqrt() + 1.0)).abs() < 1e-12);
        assert_eq!(moore.path.len(), 4);
    }

    #[test]
    fn axes_make_the_length_physical() {
        let c = grid([3, 4], vec![1.0; 12]);
        let m = StepMetric::from_axes(&[vec![0.0, 1.0, 3.0], vec![0.0, 2.0, 3.0, 7.0]]);
        let r = search(c.view(), 0, Target::Cell(11), CostIntegral, VonNeumann, &m).unwrap();
        assert!((r.cumulative.last().unwrap() - 10.0).abs() < 1e-12); // every staircase spans 3 + 7
    }

    #[test]
    fn ascent_counts_climbs_only() {
        let e = grid([1, 5], vec![0.0, 3.0, 1.0, 2.0, 0.0]);
        let r = run(&e, 0, Target::Cell(4), Ascent, VonNeumann).unwrap();
        assert_eq!(r.cumulative, vec![0.0, 3.0, 3.0, 4.0, 4.0]);
        let d = grid([2, 3], vec![0.0, 5.0, 0.0, 0.0, 1.0, 0.0]);
        let r = run(&d, 0, Target::Cell(2), Ascent, VonNeumann).unwrap();
        assert_eq!(r.path, vec![0, 3, 4, 5, 2]);
        assert_eq!(r.cumulative, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn zero_weight_region_returns_the_shortest_path() {
        let flat = grid([5, 5], vec![0.0; 25]);
        let r = run(&flat, 10, Target::Cell(14), Ascent, VonNeumann).unwrap();
        assert_eq!(r.path, vec![10, 11, 12, 13, 14]);
        let r = run(&grid([3, 3], vec![0.0; 9]), 0, Target::Cell(8), Ascent, Moore).unwrap();
        assert_eq!(r.path, vec![0, 4, 8]);
    }

    #[test]
    fn mask_ends_at_the_nearest_target_and_start_in_mask_is_trivial() {
        let c = grid([1, 6], vec![1.0; 6]);
        let mut mask = vec![false; 6];
        mask[0] = true;
        mask[5] = true;
        let r = run(&c, 2, Target::Mask(&mask), CostIntegral, VonNeumann).unwrap();
        assert_eq!((r.path, r.cumulative), (vec![2, 1, 0], vec![0.0, 1.0, 2.0]));
        let mut only = vec![false; 6];
        only[2] = true;
        let r = run(&c, 2, Target::Mask(&only), CostIntegral, VonNeumann).unwrap();
        assert_eq!((r.path, r.cumulative), (vec![2], vec![0.0]));
    }

    #[test]
    fn walls_make_the_target_unreachable() {
        let n = f64::NAN;
        let c = grid([2, 3], vec![1.0, n, 1.0, 1.0, n, 1.0]);
        assert!(run(&c, 0, Target::Cell(2), CostIntegral, Moore).is_none());
        let mut mask = vec![false; 6];
        mask[1] = true; // the only target is a wall
        assert!(run(&c, 0, Target::Mask(&mask), CostIntegral, Moore).is_none());
    }

    #[test]
    fn rounding_can_hide_a_shorter_equal_total() {
        // Spec 6.4 / review finding 3: the tie-break is exact only where partial sums are.
        let e = grid([2, 4], vec![0.0, 0.0, 0.0, f64::NAN, 0.0, 2f64.powi(-54), 0.0, 1.0]);
        let r = run(&e, 4, Target::Cell(7), Ascent, VonNeumann).unwrap();
        assert_eq!(r.path.len(), 6);
        assert_eq!(*r.cumulative.last().unwrap(), 1.0);
    }

    #[test]
    fn field_violation_reports_infinite_before_negative() {
        assert_eq!(field_violation(&[1.0, f64::INFINITY, -1.0], CostIntegral), Some("an infinite"));
        assert_eq!(field_violation(&[1.0, f64::NAN, -1.0], CostIntegral), Some("a negative"));
        assert_eq!(field_violation(&[1.0, f64::NAN, -1.0], Ascent), None);
        assert_eq!(field_violation(&[1.0f32, 2.0], CostIntegral), None);
    }

    #[test]
    fn f32_matches_f64() {
        let data = vec![0.0, 5.0, 0.0, 0.0, 1.0, 0.0];
        let e64 = grid([2, 3], data.clone());
        let e32 = Array::from_shape_vec(IxDyn(&[2, 3]), data.iter().map(|&v| v as f32).collect()).unwrap();
        let a = search(e32.view(), 0, Target::Cell(2), Ascent, VonNeumann, &StepMetric::unit(2)).unwrap();
        let b = search(e64.view(), 0, Target::Cell(2), Ascent, VonNeumann, &StepMetric::unit(2)).unwrap();
        assert_eq!((a.path, a.cumulative), (b.path, b.cumulative));
    }
}
