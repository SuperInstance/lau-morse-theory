//! Morse lemma: near a non-degenerate critical point, coordinates exist
//! such that f = f(p) - x1^2 - ... - xλ^2 + x_{λ+1}^2 + ... + x_n^2.

use nalgebra::{DVector, DMatrix};
use crate::critical_point::CriticalPoint;

/// Result of applying the Morse lemma at a critical point.
#[derive(Debug, Clone)]
pub struct MorseLemmaResult {
    /// The critical point.
    pub critical_point: CriticalPoint,
    /// The Morse index (number of negative terms).
    pub index: usize,
    /// The function value at the critical point.
    pub critical_value: f64,
    /// The transformation matrix to Morse coordinates.
    pub coordinate_transform: DMatrix<f64>,
}

impl MorseLemmaResult {
    /// Apply the Morse lemma to a non-degenerate critical point.
    ///
    /// Returns the coordinate transformation and canonical form:
    /// f(y) = f(p) - y1^2 - ... - y_λ^2 + y_{λ+1}^2 + ... + y_n^2
    pub fn apply(cp: &CriticalPoint) -> Result<Self, String> {
        let n = cp.ambient_dimension();
        let lambda = cp.index;

        match &cp.hessian {
            Some(hessian) => {
                // Verify non-degeneracy
                let eigenvalues = hessian.symmetric_eigenvalues();
                if eigenvalues.iter().any(|&e| e.abs() <= 1e-10) {
                    return Err("Critical point is degenerate; Morse lemma does not apply".to_string());
                }

                // Compute the eigendecomposition to build coordinate transform
                let eigendecomp = hessian.clone().symmetric_eigen();

                // The coordinate transform diagonalizes the Hessian
                let q = eigendecomp.eigenvectors.clone();
                let d = eigendecomp.eigenvalues.clone();

                // Build the transformation: Q^T * H * Q = D
                // Normalize eigenvalues to ±1 via scaling
                let mut scale = DMatrix::zeros(n, n);
                for i in 0..n {
                    let abs_d = d[i].abs();
                    if abs_d > 1e-10 {
                        scale[(i, i)] = 1.0 / abs_d.sqrt();
                    } else {
                        scale[(i, i)] = 1.0;
                    }
                }

                // Full transform: y = scale * Q^T * (x - p)
                let transform = &scale * &q.transpose();

                Ok(Self {
                    critical_point: cp.clone(),
                    index: lambda,
                    critical_value: cp.value,
                    coordinate_transform: transform,
                })
            }
            None => {
                // Without Hessian, use identity transform
                Ok(Self {
                    critical_point: cp.clone(),
                    index: lambda,
                    critical_value: cp.value,
                    coordinate_transform: DMatrix::identity(n, n),
                })
            }
        }
    }

    /// Evaluate the canonical Morse form at coordinates y.
    pub fn canonical_form(&self, y: &DVector<f64>) -> f64 {
        let mut result = self.critical_value;
        for i in 0..y.len() {
            if i < self.index {
                result -= y[i] * y[i];
            } else {
                result += y[i] * y[i];
            }
        }
        result
    }

    /// Transform world coordinates to Morse coordinates.
    pub fn to_morse_coordinates(&self, x: &DVector<f64>) -> DVector<f64> {
        let dx = x - &self.critical_point.position;
        &self.coordinate_transform * &dx
    }
}

/// Compute the local Morse data at a critical point.
#[derive(Debug, Clone)]
pub struct LocalMorseData {
    /// The ascending disk dimension (n - index).
    pub ascending_dimension: usize,
    /// The descending disk dimension (index).
    pub descending_dimension: usize,
    /// The Morse index.
    pub index: usize,
    /// The ambient dimension.
    pub ambient_dimension: usize,
}

impl LocalMorseData {
    pub fn from_critical_point(cp: &CriticalPoint) -> Self {
        let n = cp.ambient_dimension();
        Self {
            ascending_dimension: n - cp.index,
            descending_dimension: cp.index,
            index: cp.index,
            ambient_dimension: n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morse_lemma_minimum() {
        // Minimum: Hessian positive definite, index 0
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let cp = CriticalPoint::with_hessian(DVector::zeros(2), 0, 5.0, h);
        let result = MorseLemmaResult::apply(&cp).unwrap();
        assert_eq!(result.index, 0);
        assert_eq!(result.critical_value, 5.0);
    }

    #[test]
    fn test_morse_lemma_saddle() {
        // Saddle: index 1 in R^2
        let h = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        let cp = CriticalPoint::with_hessian(DVector::zeros(2), 1, 0.0, h);
        let result = MorseLemmaResult::apply(&cp).unwrap();
        assert_eq!(result.index, 1);
    }

    #[test]
    fn test_morse_lemma_maximum() {
        // Maximum: Hessian negative definite, index 2 in R^2
        let h = DMatrix::from_row_slice(2, 2, &[-2.0, 0.0, 0.0, -3.0]);
        let cp = CriticalPoint::with_hessian(DVector::zeros(2), 2, 10.0, h);
        let result = MorseLemmaResult::apply(&cp).unwrap();
        assert_eq!(result.index, 2);
    }

    #[test]
    fn test_canonical_form_minimum() {
        let cp = CriticalPoint::new(DVector::zeros(2), 0, 5.0);
        let result = MorseLemmaResult::apply(&cp).unwrap();
        // f(y) = 5 + y1^2 + y2^2
        let y = DVector::from_vec(vec![1.0, 2.0]);
        assert_eq!(result.canonical_form(&y), 5.0 + 1.0 + 4.0);
    }

    #[test]
    fn test_canonical_form_saddle() {
        let cp = CriticalPoint::new(DVector::zeros(2), 1, 0.0);
        let result = MorseLemmaResult::apply(&cp).unwrap();
        // f(y) = 0 - y1^2 + y2^2
        let y = DVector::from_vec(vec![3.0, 4.0]);
        assert_eq!(result.canonical_form(&y), 0.0 - 9.0 + 16.0);
    }

    #[test]
    fn test_degenerate_rejected() {
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let cp = CriticalPoint::with_hessian(DVector::zeros(2), 0, 0.0, h);
        assert!(MorseLemmaResult::apply(&cp).is_err());
    }

    #[test]
    fn test_local_morse_data() {
        let cp = CriticalPoint::new(DVector::zeros(3), 1, 0.0);
        let data = LocalMorseData::from_critical_point(&cp);
        assert_eq!(data.ascending_dimension, 2);
        assert_eq!(data.descending_dimension, 1);
        assert_eq!(data.index, 1);
    }
}
