//! Manifold representations for Morse theory.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// A differentiable manifold represented as a subset of R^n with charts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifold {
    /// Embedding dimension.
    pub dimension: usize,
    /// Optional name for identification.
    pub name: String,
}

impl Manifold {
    /// Create a new manifold of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: String::new(),
        }
    }

    /// Create a named manifold.
    pub fn named(dimension: usize, name: &str) -> Self {
        Self {
            dimension,
            name: name.to_string(),
        }
    }

    /// Tangent space at a point (in R^n, this is just R^n itself).
    pub fn tangent_space(&self, _point: &DVector<f64>) -> TangentSpace {
        TangentSpace {
            dimension: self.dimension,
            point: DVector::zeros(self.dimension),
        }
    }

    /// Euler characteristic from Betti numbers.
    pub fn euler_characteristic(betti: &[usize]) -> i64 {
        betti.iter()
            .enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum()
    }

    /// The n-sphere.
    pub fn sphere(n: usize) -> Self {
        Self::named(n, &format!("S^{}", n))
    }

    /// The n-torus.
    pub fn torus(n: usize) -> Self {
        Self::named(n, &format!("T^{}", n))
    }

    /// Rn as a manifold.
    pub fn euclidean(n: usize) -> Self {
        Self::named(n, &format!("R^{}", n))
    }
}

/// Tangent space at a point on a manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TangentSpace {
    pub dimension: usize,
    pub point: DVector<f64>,
}

impl TangentSpace {
    /// Inner product of two tangent vectors.
    pub fn inner_product(&self, v: &DVector<f64>, w: &DVector<f64>) -> f64 {
        v.dot(w)
    }

    /// Project a vector onto this tangent space.
    pub fn project(&self, v: &DVector<f64>) -> DVector<f64> {
        v.clone()
    }
}

/// Betti numbers for a topological space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiNumbers {
    pub numbers: Vec<usize>,
}

impl BettiNumbers {
    pub fn new(numbers: Vec<usize>) -> Self {
        Self { numbers }
    }

    pub fn euler_characteristic(&self) -> i64 {
        Manifold::euler_characteristic(&self.numbers)
    }

    pub fn get(&self, k: usize) -> usize {
        self.numbers.get(k).copied().unwrap_or(0)
    }

    /// Total rank (sum of all Betti numbers).
    pub fn total_rank(&self) -> usize {
        self.numbers.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifold_creation() {
        let m = Manifold::new(3);
        assert_eq!(m.dimension, 3);
    }

    #[test]
    fn test_sphere_creation() {
        let s = Manifold::sphere(2);
        assert_eq!(s.dimension, 2);
        assert_eq!(s.name, "S^2");
    }

    #[test]
    fn test_euler_characteristic() {
        // S^2: betti = [1, 0, 1] -> chi = 1 - 0 + 1 = 2
        assert_eq!(Manifold::euler_characteristic(&[1, 0, 1]), 2);
        // Torus T^2: betti = [1, 2, 1] -> chi = 1 - 2 + 1 = 0
        assert_eq!(Manifold::euler_characteristic(&[1, 2, 1]), 0);
        // Klein bottle: betti = [1, 1, 0] -> chi = 1 - 1 + 0 = 0
        assert_eq!(Manifold::euler_characteristic(&[1, 1, 0]), 0);
    }

    #[test]
    fn test_tangent_space() {
        let m = Manifold::new(3);
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let ts = m.tangent_space(&point);
        assert_eq!(ts.dimension, 3);
    }

    #[test]
    fn test_betti_numbers() {
        let b = BettiNumbers::new(vec![1, 0, 1]);
        assert_eq!(b.get(0), 1);
        assert_eq!(b.get(1), 0);
        assert_eq!(b.get(2), 1);
        assert_eq!(b.get(3), 0); // out of range
        assert_eq!(b.total_rank(), 2);
    }
}
