//! Critical points of Morse functions.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A critical point of a Morse function.
///
/// At a critical point, the gradient vanishes. The index (Morse index) is
/// the number of negative eigenvalues of the Hessian at that point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPoint {
    /// Position of the critical point in R^n.
    pub position: DVector<f64>,
    /// Morse index (number of negative eigenvalues of the Hessian).
    pub index: usize,
    /// Function value at the critical point.
    pub value: f64,
    /// Optional Hessian at the critical point.
    pub hessian: Option<DMatrix<f64>>,
}

impl CriticalPoint {
    /// Create a new critical point.
    pub fn new(position: DVector<f64>, index: usize, value: f64) -> Self {
        Self {
            position,
            index,
            value,
            hessian: None,
        }
    }

    /// Create a critical point with an explicit Hessian.
    pub fn with_hessian(position: DVector<f64>, index: usize, value: f64, hessian: DMatrix<f64>) -> Self {
        Self {
            position,
            index,
            value,
            hessian: Some(hessian),
        }
    }

    /// Dimension of the ambient space.
    pub fn ambient_dimension(&self) -> usize {
        self.position.len()
    }

    /// Check if the critical point is non-degenerate (Hessian has no zero eigenvalues).
    pub fn is_non_degenerate(&self) -> bool {
        match &self.hessian {
            Some(h) => {
                let eigenvalues = h.symmetric_eigenvalues();
                eigenvalues.iter().all(|&e| e.abs() > 1e-10)
            }
            None => true, // Assume non-degenerate if Hessian not provided
        }
    }

    /// Compute the index from the Hessian (count negative eigenvalues).
    pub fn compute_index_from_hessian(hessian: &DMatrix<f64>) -> usize {
        let eigenvalues = hessian.symmetric_eigenvalues();
        eigenvalues.iter().filter(|&&e| e < -1e-10).count()
    }

    /// Nullity of the critical point (dimension of kernel of Hessian).
    pub fn nullity(&self) -> usize {
        match &self.hessian {
            Some(h) => {
                let eigenvalues = h.symmetric_eigenvalues();
                eigenvalues.iter().filter(|&&e| e.abs() <= 1e-10).count()
            }
            None => 0,
        }
    }

    /// Check if this is a local minimum (index 0).
    pub fn is_minimum(&self) -> bool {
        self.index == 0
    }

    /// Check if this is a local maximum.
    pub fn is_maximum(&self) -> bool {
        self.index == self.ambient_dimension()
    }

    /// Check if this is a saddle point.
    pub fn is_saddle(&self) -> bool {
        self.index > 0 && self.index < self.ambient_dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_point_creation() {
        let cp = CriticalPoint::new(DVector::from_vec(vec![1.0, 2.0]), 1, 3.0);
        assert_eq!(cp.index, 1);
        assert_eq!(cp.value, 3.0);
        assert_eq!(cp.ambient_dimension(), 2);
    }

    #[test]
    fn test_minimum_detection() {
        let cp = CriticalPoint::new(DVector::zeros(3), 0, 0.0);
        assert!(cp.is_minimum());
        assert!(!cp.is_maximum());
        assert!(!cp.is_saddle());
    }

    #[test]
    fn test_maximum_detection() {
        let cp = CriticalPoint::new(DVector::zeros(3), 3, 1.0);
        assert!(cp.is_maximum());
        assert!(!cp.is_minimum());
    }

    #[test]
    fn test_saddle_detection() {
        let cp = CriticalPoint::new(DVector::zeros(3), 1, 0.5);
        assert!(cp.is_saddle());
    }

    #[test]
    fn test_non_degenerate_with_positive_hessian() {
        // Positive definite 2x2 -> index 0, non-degenerate
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let cp = CriticalPoint::with_hessian(DVector::zeros(2), 0, 0.0, h);
        assert!(cp.is_non_degenerate());
        assert_eq!(CriticalPoint::compute_index_from_hessian(&DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0])), 0);
    }

    #[test]
    fn test_index_from_hessian_mixed() {
        // 2 negative, 1 positive -> index 2
        let h = DMatrix::from_row_slice(3, 3, &[-2.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(CriticalPoint::compute_index_from_hessian(&h), 2);
    }

    #[test]
    fn test_index_from_hessian_saddle() {
        // One positive, one negative -> index 1 (saddle)
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
        assert_eq!(CriticalPoint::compute_index_from_hessian(&h), 1);
    }

    #[test]
    fn test_degenerate_hessian() {
        // Zero eigenvalue -> degenerate
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let cp = CriticalPoint::with_hessian(DVector::zeros(2), 0, 0.0, h);
        assert!(!cp.is_non_degenerate());
        assert_eq!(cp.nullity(), 1);
    }
}
