//! Gradient flow lines and stable/unstable manifolds.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::critical_point::CriticalPoint;

/// A gradient flow line connecting two critical points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowLine {
    /// Starting critical point (higher value).
    pub source: CriticalPoint,
    /// Ending critical point (lower value).
    pub target: CriticalPoint,
    /// Sampled points along the flow line.
    pub trajectory: Vec<DVector<f64>>,
    /// Duration parameter.
    pub duration: f64,
}

impl FlowLine {
    /// Create a flow line with sampled trajectory.
    pub fn new(source: CriticalPoint, target: CriticalPoint, trajectory: Vec<DVector<f64>>) -> Self {
        Self {
            source,
            target,
            trajectory,
            duration: 1.0,
        }
    }

    /// Length of the flow line (sum of segment lengths).
    pub fn length(&self) -> f64 {
        self.trajectory
            .windows(2)
            .map(|w| (&w[1] - &w[0]).norm())
            .sum()
    }

    /// Number of trajectory samples.
    pub fn num_samples(&self) -> usize {
        self.trajectory.len()
    }

    /// Start point of the flow.
    pub fn start(&self) -> &DVector<f64> {
        &self.trajectory.first().unwrap_or(&self.source.position)
    }

    /// End point of the flow.
    pub fn end(&self) -> &DVector<f64> {
        &self.trajectory.last().unwrap_or(&self.target.position)
    }

    /// Interpolate the flow line at parameter t ∈ [0, 1].
    pub fn interpolate(&self, t: f64) -> DVector<f64> {
        assert!((0.0..=1.0).contains(&t), "t must be in [0, 1]");
        if self.trajectory.is_empty() {
            return self.source.position.clone();
        }
        let idx = t * (self.trajectory.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(self.trajectory.len() - 1);
        let frac = idx - lo as f64;
        &self.trajectory[lo] * (1.0 - frac) + &self.trajectory[hi] * frac
    }
}

/// Stable manifold (ascending manifold) of a critical point.
/// The set of points that flow to the critical point under negative gradient flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableManifold {
    /// The critical point.
    pub critical_point: CriticalPoint,
    /// Dimension of the stable manifold = n - index.
    pub dimension: usize,
    /// Sample boundary points.
    pub boundary_points: Vec<DVector<f64>>,
}

impl StableManifold {
    pub fn new(cp: &CriticalPoint, n: usize) -> Self {
        Self {
            dimension: n - cp.index,
            critical_point: cp.clone(),
            boundary_points: Vec::new(),
        }
    }

    /// Is the stable manifold open (dimension > 0)?
    pub fn is_open(&self) -> bool {
        self.dimension > 0
    }
}

/// Unstable manifold (descending manifold) of a critical point.
/// The set of points that flow from the critical point under negative gradient flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnstableManifold {
    /// The critical point.
    pub critical_point: CriticalPoint,
    /// Dimension of the unstable manifold = index.
    pub dimension: usize,
    /// Sample boundary points.
    pub boundary_points: Vec<DVector<f64>>,
}

impl UnstableManifold {
    pub fn new(cp: &CriticalPoint) -> Self {
        Self {
            dimension: cp.index,
            critical_point: cp.clone(),
            boundary_points: Vec::new(),
        }
    }

    /// Is the unstable manifold open (dimension > 0)?
    pub fn is_open(&self) -> bool {
        self.dimension > 0
    }
}

/// Numerical gradient flow integration using forward Euler.
pub struct GradientFlowIntegrator {
    /// Step size.
    pub step_size: f64,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Convergence threshold.
    pub tolerance: f64,
}

impl GradientFlowIntegrator {
    pub fn new(step_size: f64, max_steps: usize, tolerance: f64) -> Self {
        Self { step_size, max_steps, tolerance }
    }

    /// Integrate the negative gradient flow from a starting point.
    /// Returns the trajectory as a sequence of points.
    pub fn integrate(
        &self,
        start: &DVector<f64>,
        gradient: fn(&DVector<f64>) -> DVector<f64>,
    ) -> Vec<DVector<f64>> {
        let mut trajectory = vec![start.clone()];
        let mut x = start.clone();

        for _ in 0..self.max_steps {
            let grad = gradient(&x);
            let norm = grad.norm();
            if norm < self.tolerance {
                break;
            }
            // Negative gradient flow: x_{n+1} = x_n - ε ∇f(x_n)
            x = &x - self.step_size * &grad;
            trajectory.push(x.clone());
        }

        trajectory
    }

    /// Integrate and find which critical point the flow converges to.
    pub fn flow_to_critical(
        &self,
        start: &DVector<f64>,
        gradient: fn(&DVector<f64>) -> DVector<f64>,
        critical_points: &[CriticalPoint],
    ) -> Option<usize> {
        let trajectory = self.integrate(start, gradient);
        let final_point = trajectory.last()?;

        // Find nearest critical point
        critical_points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.position.clone() - final_point).norm()
                    .partial_cmp(&(b.position.clone() - final_point).norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

/// Generate a flow line between two critical points by linear interpolation.
pub fn interpolate_flow_line(source: &CriticalPoint, target: &CriticalPoint, steps: usize) -> FlowLine {
    let trajectory: Vec<DVector<f64>> = (0..=steps)
        .map(|i| {
            let t = i as f64 / steps as f64;
            &source.position * (1.0 - t) + &target.position * t
        })
        .collect();
    FlowLine::new(source.clone(), target.clone(), trajectory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_line_creation() {
        let src = CriticalPoint::new(DVector::from_vec(vec![1.0, 1.0]), 1, 1.0);
        let tgt = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let fl = interpolate_flow_line(&src, &tgt, 10);
        assert_eq!(fl.num_samples(), 11);
    }

    #[test]
    fn test_flow_line_length() {
        let src = CriticalPoint::new(DVector::from_vec(vec![1.0, 0.0]), 1, 1.0);
        let tgt = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let fl = interpolate_flow_line(&src, &tgt, 100);
        let len = fl.length();
        assert!((len - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_flow_line_interpolation() {
        let src = CriticalPoint::new(DVector::from_vec(vec![2.0, 0.0]), 1, 1.0);
        let tgt = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let fl = interpolate_flow_line(&src, &tgt, 100);
        let mid = fl.interpolate(0.5);
        assert!((mid[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_stable_manifold() {
        let cp = CriticalPoint::new(DVector::zeros(3), 1, 0.0);
        let sm = StableManifold::new(&cp, 3);
        assert_eq!(sm.dimension, 2); // n - index = 3 - 1 = 2
    }

    #[test]
    fn test_unstable_manifold() {
        let cp = CriticalPoint::new(DVector::zeros(3), 1, 0.0);
        let um = UnstableManifold::new(&cp);
        assert_eq!(um.dimension, 1); // index = 1
    }

    #[test]
    fn test_gradient_flow_integration() {
        // f(x) = x1^2 + x2^2, ∇f = (2x1, 2x2)
        let integrator = GradientFlowIntegrator::new(0.1, 1000, 1e-6);
        let start = DVector::from_vec(vec![1.0, 1.0]);
        let trajectory = integrator.integrate(&start, |x| 2.0 * x.clone());
        let final_point = trajectory.last().unwrap();
        assert!(final_point.norm() < 0.1);
    }

    #[test]
    fn test_flow_to_critical() {
        let integrator = GradientFlowIntegrator::new(0.1, 1000, 1e-6);
        let start = DVector::from_vec(vec![0.9, 0.9]);
        let cps = vec![
            CriticalPoint::new(DVector::zeros(2), 0, 0.0),
            CriticalPoint::new(DVector::from_vec(vec![5.0, 5.0]), 1, 50.0),
        ];
        let idx = integrator.flow_to_critical(&start, |x| 2.0 * x.clone(), &cps);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_stable_manifold_minimum() {
        let cp = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let sm = StableManifold::new(&cp, 2);
        assert_eq!(sm.dimension, 2); // Whole manifold flows to minimum
    }

    #[test]
    fn test_unstable_manifold_minimum() {
        let cp = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let um = UnstableManifold::new(&cp);
        assert_eq!(um.dimension, 0); // Just the point itself
    }
}
