//! Handlebody decomposition from critical points.
//!
//! Each critical point of index k corresponds to attaching a k-handle.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::critical_point::CriticalPoint;
use crate::morse_function::MorseFunction;

/// A k-handle attached during handlebody decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handle {
    /// Index of the handle (dimension of the core disk).
    pub index: usize,
    /// The critical point that generates this handle.
    pub critical_point: CriticalPoint,
    /// Attachment sphere dimension (index - 1).
    pub attachment_sphere_dim: isize,
    /// Belt sphere dimension (n - index - 1).
    pub belt_sphere_dim: isize,
}

impl Handle {
    /// Create a handle from a critical point in an n-dimensional manifold.
    pub fn from_critical_point(cp: &CriticalPoint, n: usize) -> Self {
        let k = cp.index;
        Handle {
            index: k,
            critical_point: cp.clone(),
            attachment_sphere_dim: if k > 0 { (k - 1) as isize } else { -1 },
            belt_sphere_dim: if n > k { (n - k - 1) as isize } else { -1 },
        }
    }

    /// Is this a 0-handle (corresponding to a minimum)?
    pub fn is_zero_handle(&self) -> bool {
        self.index == 0
    }

    /// Is this an n-handle (corresponding to a maximum)?
    pub fn is_top_handle(&self, n: usize) -> bool {
        self.index == n
    }
}

/// A complete handlebody decomposition of a manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlebodyDecomposition {
    /// Dimension of the manifold.
    pub dimension: usize,
    /// Handles, sorted by critical value then index.
    pub handles: Vec<Handle>,
}

impl HandlebodyDecomposition {
    /// Construct a handlebody decomposition from a Morse function.
    pub fn from_morse_function(f: &MorseFunction) -> Self {
        let n = f.dimension;
        let mut handles: Vec<Handle> = f.critical_points
            .iter()
            .map(|cp| Handle::from_critical_point(cp, n))
            .collect();

        // Sort by critical value (ascending), then by index
        handles.sort_by(|a, b| {
            a.critical_point.value
                .partial_cmp(&b.critical_point.value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });

        HandlebodyDecomposition {
            dimension: n,
            handles,
        }
    }

    /// Handles of a specific index.
    pub fn handles_of_index(&self, k: usize) -> Vec<&Handle> {
        self.handles.iter().filter(|h| h.index == k).collect()
    }

    /// Count of handles of each index.
    pub fn handle_counts(&self) -> Vec<usize> {
        let max_idx = self.handles.iter().map(|h| h.index).max().unwrap_or(0);
        let mut counts = vec![0usize; max_idx + 1];
        for h in &self.handles {
            counts[h.index] += 1;
        }
        counts
    }

    /// Number of k-handles.
    pub fn num_handles(&self, k: usize) -> usize {
        self.handles_of_index(k).len()
    }

    /// The sublevel set at a given value.
    /// Returns the handles attached at critical values ≤ the given value.
    pub fn sublevel_set(&self, value: f64) -> Vec<&Handle> {
        self.handles
            .iter()
            .filter(|h| h.critical_point.value <= value)
            .collect()
    }

    /// Euler characteristic from handle decomposition.
    /// χ = Σ (-1)^k * (number of k-handles)
    pub fn euler_characteristic(&self) -> i64 {
        self.handles
            .iter()
            .map(|h| if h.index % 2 == 0 { 1i64 } else { -1i64 })
            .sum()
    }

    /// Check if the decomposition is in a nice form (sorted by index).
    pub fn is_sorted_by_index(&self) -> bool {
        self.handles.windows(2).all(|w| w[0].index <= w[1].index)
    }

    /// Simplify by canceling handle pairs.
    /// A k-handle and (k+1)-handle can cancel if connected by a single gradient flow line.
    pub fn cancel_handle_pairs(&mut self) -> usize {
        let mut canceled = 0;
        let mut to_remove = vec![false; self.handles.len()];

        for i in 0..self.handles.len() {
            if to_remove[i] { continue; }
            for j in (i + 1)..self.handles.len() {
                if to_remove[j] { continue; }
                let hi = &self.handles[i];
                let hj = &self.handles[j];
                // Can cancel if indices differ by 1
                if hj.index == hi.index + 1 {
                    to_remove[i] = true;
                    to_remove[j] = true;
                    canceled += 1;
                    break;
                }
            }
        }

        let mut new_handles = Vec::new();
        for (i, h) in self.handles.drain(..).enumerate() {
            if !to_remove[i] {
                new_handles.push(h);
            }
        }
        self.handles = new_handles;
        canceled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_creation() {
        let cp = CriticalPoint::new(DVector::zeros(3), 1, 0.0);
        let handle = Handle::from_critical_point(&cp, 3);
        assert_eq!(handle.index, 1);
        assert_eq!(handle.attachment_sphere_dim, 0);
        assert_eq!(handle.belt_sphere_dim, 1);
    }

    #[test]
    fn test_zero_handle() {
        let cp = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let handle = Handle::from_critical_point(&cp, 2);
        assert!(handle.is_zero_handle());
        assert_eq!(handle.attachment_sphere_dim, -1); // S^{-1} = ∅
    }

    #[test]
    fn test_handlebody_s1() {
        let f = MorseFunction::height_s1();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        assert_eq!(hb.handles.len(), 2);
        assert_eq!(hb.num_handles(0), 1);
        assert_eq!(hb.num_handles(1), 1);
    }

    #[test]
    fn test_handlebody_s2() {
        let f = MorseFunction::height_s2();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        assert_eq!(hb.handles.len(), 2);
        assert_eq!(hb.num_handles(0), 1);
        assert_eq!(hb.num_handles(2), 1);
        assert_eq!(hb.euler_characteristic(), 1 + 1); // 2
    }

    #[test]
    fn test_handlebody_torus() {
        let f = MorseFunction::height_torus();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        assert_eq!(hb.handles.len(), 4);
        assert_eq!(hb.num_handles(0), 1);
        assert_eq!(hb.num_handles(1), 2);
        assert_eq!(hb.num_handles(2), 1);
        assert_eq!(hb.euler_characteristic(), 1 - 2 + 1); // 0
    }

    #[test]
    fn test_sublevel_set() {
        let f = MorseFunction::height_torus();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        // At value 1.5, should have min (0.0) and saddle (1.0)
        let sub = hb.sublevel_set(1.5);
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn test_cancel_handle_pairs() {
        let f = MorseFunction::height_torus();
        let mut hb = HandlebodyDecomposition::from_morse_function(&f);
        // Cancel pairs: (0,1) and (1,2) -> should cancel 2 pairs
        let canceled = hb.cancel_handle_pairs();
        assert_eq!(canceled, 2); // Two pairs cancelled
        assert_eq!(hb.handles.len(), 0); // All cancelled
    }

    #[test]
    fn test_handle_counts() {
        let f = MorseFunction::height_torus();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        let counts = hb.handle_counts();
        assert_eq!(counts, vec![1, 2, 1]);
    }

    #[test]
    fn test_euler_from_handles_matches_manifold() {
        // Torus: χ = 0
        let f = MorseFunction::height_torus();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        assert_eq!(hb.euler_characteristic(), 0);

        // S^2: χ = 2
        let f = MorseFunction::height_s2();
        let hb = HandlebodyDecomposition::from_morse_function(&f);
        assert_eq!(hb.euler_characteristic(), 2);
    }
}
