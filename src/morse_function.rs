//! Morse functions and their properties.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::critical_point::CriticalPoint;

/// A smooth function f: M -> R on a manifold of dimension n.
/// Represented by its evaluation, gradient, and Hessian closures.
#[derive(Clone, Serialize, Deserialize)]
pub struct MorseFunction {
    /// Name of the function.
    pub name: String,
    /// Dimension of the domain manifold.
    pub dimension: usize,
    /// Evaluation: f(x).
    #[serde(skip)]
    pub eval: Option<fn(&DVector<f64>) -> f64>,
    /// Gradient: ∇f(x).
    #[serde(skip)]
    pub grad: Option<fn(&DVector<f64>) -> DVector<f64>>,
    /// Hessian: Hf(x).
    #[serde(skip)]
    pub hessian: Option<fn(&DVector<f64>) -> DMatrix<f64>>,
    /// Known critical points (precomputed).
    pub critical_points: Vec<CriticalPoint>,
}

impl std::fmt::Debug for MorseFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorseFunction")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .field("critical_points", &self.critical_points)
            .finish()
    }
}

impl MorseFunction {
    /// Create a Morse function with explicit closures.
    pub fn new(
        name: &str,
        dimension: usize,
        eval: fn(&DVector<f64>) -> f64,
        grad: fn(&DVector<f64>) -> DVector<f64>,
        hessian: fn(&DVector<f64>) -> DMatrix<f64>,
    ) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            eval: Some(eval),
            grad: Some(grad),
            hessian: Some(hessian),
            critical_points: Vec::new(),
        }
    }

    /// Create from known critical points only (no closures).
    pub fn from_critical_points(name: &str, dimension: usize, cps: Vec<CriticalPoint>) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            eval: None,
            grad: None,
            hessian: None,
            critical_points: cps,
        }
    }

    /// Evaluate the function at a point.
    pub fn evaluate(&self, x: &DVector<f64>) -> Option<f64> {
        self.eval.map(|f| f(x))
    }

    /// Compute gradient at a point.
    pub fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.grad.map(|g| g(x))
    }

    /// Compute Hessian at a point.
    pub fn hessian_at(&self, x: &DVector<f64>) -> Option<DMatrix<f64>> {
        self.hessian.map(|h| h(x))
    }

    /// Count critical points by index.
    pub fn count_by_index(&self) -> Vec<usize> {
        if self.critical_points.is_empty() {
            return Vec::new();
        }
        let max_index = self.critical_points.iter().map(|cp| cp.index).max().unwrap();
        let mut counts = vec![0usize; max_index + 1];
        for cp in &self.critical_points {
            counts[cp.index] += 1;
        }
        counts
    }

    /// Number of critical points of index k.
    pub fn mu_k(&self, k: usize) -> usize {
        self.count_by_index().get(k).copied().unwrap_or(0)
    }

    /// Total number of critical points.
    pub fn total_critical_points(&self) -> usize {
        self.critical_points.len()
    }

    /// Check if this is a valid Morse function (all critical points non-degenerate).
    pub fn is_morse(&self) -> bool {
        self.critical_points.iter().all(|cp| cp.is_non_degenerate())
    }

    /// The standard Morse function on R^n: f(x) = sum of x_i^2 with a critical point at origin.
    pub fn standard(dim: usize) -> Self {
        let cp = CriticalPoint::new(DVector::zeros(dim), 0, 0.0);
        Self::from_critical_points("standard_quadratic", dim, vec![cp])
    }

    /// Height function on S^1: f(θ) = sin(θ) with critical points at π/2 (index 0) and 3π/2 (index 1).
    pub fn height_s1() -> Self {
        Self::from_critical_points(
            "height_S1",
            1,
            vec![
                CriticalPoint::new(DVector::from_vec(vec![std::f64::consts::FRAC_PI_2]), 0, 1.0),
                CriticalPoint::new(DVector::from_vec(vec![3.0 * std::f64::consts::FRAC_PI_2]), 1, -1.0),
            ],
        )
    }

    /// Height function on S^2: f(φ,θ) = cos(φ), with one min (index 0) and one max (index 2).
    pub fn height_s2() -> Self {
        Self::from_critical_points(
            "height_S2",
            2,
            vec![
                CriticalPoint::new(DVector::from_vec(vec![0.0, 0.0]), 0, -1.0), // south pole
                CriticalPoint::new(DVector::from_vec(vec![std::f64::consts::PI, 0.0]), 2, 1.0), // north pole
            ],
        )
    }

    /// Morse function on T^2 (torus) with 4 critical points.
    pub fn height_torus() -> Self {
        Self::from_critical_points(
            "height_T2",
            2,
            vec![
                CriticalPoint::new(DVector::from_vec(vec![0.0, 0.0]), 0, 0.0),     // min
                CriticalPoint::new(DVector::from_vec(vec![std::f64::consts::PI, 0.0]), 1, 2.0),  // saddle
                CriticalPoint::new(DVector::from_vec(vec![0.0, std::f64::consts::PI]), 1, 1.0),  // saddle
                CriticalPoint::new(DVector::from_vec(vec![std::f64::consts::PI, std::f64::consts::PI]), 2, 3.0), // max
            ],
        )
    }
}

/// A Morse function builder for constructing functions on R^n.
pub struct MorseFunctionBuilder {
    name: String,
    dimension: usize,
    critical_points: Vec<CriticalPoint>,
}

impl MorseFunctionBuilder {
    pub fn new(name: &str, dimension: usize) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            critical_points: Vec::new(),
        }
    }

    pub fn critical_point(mut self, position: DVector<f64>, index: usize, value: f64) -> Self {
        self.critical_points.push(CriticalPoint::new(position, index, value));
        self
    }

    pub fn build(self) -> MorseFunction {
        MorseFunction::from_critical_points(&self.name, self.dimension, self.critical_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_morse_function() {
        let f = MorseFunction::standard(3);
        assert_eq!(f.total_critical_points(), 1);
        assert_eq!(f.mu_k(0), 1);
        assert!(f.is_morse());
    }

    #[test]
    fn test_height_s1() {
        let f = MorseFunction::height_s1();
        assert_eq!(f.total_critical_points(), 2);
        assert_eq!(f.mu_k(0), 1);
        assert_eq!(f.mu_k(1), 1);
        assert!(f.is_morse());
    }

    #[test]
    fn test_height_s2() {
        let f = MorseFunction::height_s2();
        assert_eq!(f.total_critical_points(), 2);
        assert_eq!(f.mu_k(0), 1);
        assert_eq!(f.mu_k(2), 1);
    }

    #[test]
    fn test_height_torus() {
        let f = MorseFunction::height_torus();
        assert_eq!(f.total_critical_points(), 4);
        assert_eq!(f.mu_k(0), 1);
        assert_eq!(f.mu_k(1), 2);
        assert_eq!(f.mu_k(2), 1);
    }

    #[test]
    fn test_count_by_index() {
        let f = MorseFunction::height_torus();
        let counts = f.count_by_index();
        assert_eq!(counts, vec![1, 2, 1]);
    }

    #[test]
    fn test_builder() {
        let f = MorseFunctionBuilder::new("custom", 2)
            .critical_point(DVector::zeros(2), 0, 0.0)
            .critical_point(DVector::from_vec(vec![1.0, 0.0]), 1, 1.0)
            .critical_point(DVector::from_vec(vec![0.0, 1.0]), 1, 1.5)
            .critical_point(DVector::from_vec(vec![1.0, 1.0]), 2, 3.0)
            .build();
        assert_eq!(f.total_critical_points(), 4);
        assert_eq!(f.mu_k(0), 1);
        assert_eq!(f.mu_k(1), 2);
        assert_eq!(f.mu_k(2), 1);
    }
}
