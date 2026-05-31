//! Agent state spaces as manifolds with Morse theory.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::critical_point::CriticalPoint;
use crate::morse_function::MorseFunction;
use crate::gradient_flow::FlowLine;
use crate::manifold::Manifold;

/// An agent state as a point on a manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// State vector in R^n.
    pub state: DVector<f64>,
    /// Timestamp or step.
    pub step: f64,
    /// Optional label.
    pub label: String,
}

impl AgentState {
    pub fn new(state: DVector<f64>) -> Self {
        Self {
            state,
            step: 0.0,
            label: String::new(),
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    pub fn at_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Dimension of the state space.
    pub fn dimension(&self) -> usize {
        self.state.len()
    }

    /// Distance to another state.
    pub fn distance_to(&self, other: &AgentState) -> f64 {
        (&self.state - &other.state).norm()
    }
}

/// An agent state space: a manifold with a Morse function describing fitness/energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateSpace {
    /// The underlying manifold.
    pub manifold: Manifold,
    /// The Morse function (objective/energy landscape).
    pub morse_function: MorseFunction,
    /// Current state.
    pub current_state: Option<AgentState>,
    /// History of states.
    pub trajectory: Vec<AgentState>,
}

impl AgentStateSpace {
    /// Create a new agent state space.
    pub fn new(manifold: Manifold, morse_function: MorseFunction) -> Self {
        Self {
            manifold,
            morse_function,
            current_state: None,
            trajectory: Vec::new(),
        }
    }

    /// Set the current state.
    pub fn set_state(&mut self, state: AgentState) {
        self.trajectory.push(state.clone());
        self.current_state = Some(state);
    }

    /// Get the nearest critical point to the current state.
    pub fn nearest_critical_point(&self) -> Option<&CriticalPoint> {
        self.current_state.as_ref()?;
        let current = self.current_state.as_ref().unwrap();
        self.morse_function.critical_points
            .iter()
            .min_by(|a, b| {
                (&a.position - &current.state).norm()
                    .partial_cmp(&(&b.position - &current.state).norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get the nearest minimum (stable state) to the current state.
    pub fn nearest_minimum(&self) -> Option<&CriticalPoint> {
        self.current_state.as_ref()?;
        let current = self.current_state.as_ref().unwrap();
        self.morse_function.critical_points
            .iter()
            .filter(|cp| cp.is_minimum())
            .min_by(|a, b| {
                (&a.position - &current.state).norm()
                    .partial_cmp(&(&b.position - &current.state).norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Transition: move toward a critical point along a gradient flow.
    pub fn transition_toward(&mut self, target: &CriticalPoint, step_size: f64) -> Option<AgentState> {
        let current = self.current_state.as_ref()?;
        let direction = &target.position - &current.state;
        let dist = direction.norm();
        if dist < 1e-10 {
            return None;
        }
        let new_state = if dist < step_size {
            AgentState::new(target.position.clone()).at_step(current.step + 1.0)
        } else {
            let step = direction.normalize() * step_size;
            AgentState::new(&current.state + &step).at_step(current.step + 1.0)
        };
        self.set_state(new_state.clone());
        Some(new_state)
    }

    /// Classify current state region (which stable manifold it belongs to).
    pub fn classify_region(&self) -> Option<usize> {
        let cp = self.nearest_critical_point()?;
        Some(cp.index)
    }

    /// Get all stable states (local minima).
    pub fn stable_states(&self) -> Vec<&CriticalPoint> {
        self.morse_function.critical_points
            .iter()
            .filter(|cp| cp.is_minimum())
            .collect()
    }

    /// Get all unstable states (local maxima).
    pub fn unstable_states(&self) -> Vec<&CriticalPoint> {
        self.morse_function.critical_points
            .iter()
            .filter(|cp| cp.is_maximum())
            .collect()
    }

    /// Get all saddle states.
    pub fn saddle_states(&self) -> Vec<&CriticalPoint> {
        self.morse_function.critical_points
            .iter()
            .filter(|cp| cp.is_saddle())
            .collect()
    }

    /// Generate a trajectory following gradient descent.
    pub fn gradient_descent_trajectory(
        &self,
        start: &AgentState,
        gradient: fn(&DVector<f64>) -> DVector<f64>,
        step_size: f64,
        max_steps: usize,
    ) -> Vec<AgentState> {
        let mut trajectory = vec![start.clone()];
        let mut x = start.state.clone();

        for i in 0..max_steps {
            let grad = gradient(&x);
            if grad.norm() < 1e-8 {
                break;
            }
            x = &x - step_size * &grad;
            trajectory.push(AgentState::new(x.clone()).at_step(i as f64 + 1.0));
        }

        trajectory
    }

    /// Count transitions between regions (barriers).
    pub fn count_barrier_crossings(&self) -> usize {
        let mut crossings = 0;
        for window in self.trajectory.windows(2) {
            // Find nearest CP for each state
            let cp0 = self.morse_function.critical_points.iter()
                .min_by(|a, b| {
                    (&a.position - &window[0].state).norm()
                        .partial_cmp(&(&b.position - &window[0].state).norm())
                        .unwrap()
                });
            let cp1 = self.morse_function.critical_points.iter()
                .min_by(|a, b| {
                    (&a.position - &window[1].state).norm()
                        .partial_cmp(&(&b.position - &window[1].state).norm())
                        .unwrap()
                });
            if let (Some(c0), Some(c1)) = (cp0, cp1) {
                if c0.value != c1.value {
                    crossings += 1;
                }
            }
        }
        crossings
    }
}

/// An agent transition viewed as a gradient flow line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTransition {
    /// Source state.
    pub from: AgentState,
    /// Target state.
    pub to: AgentState,
    /// The flow line.
    pub flow_line: FlowLine,
    /// Energy barrier crossed.
    pub energy_barrier: f64,
}

impl AgentTransition {
    /// Create a transition from one state to another.
    pub fn new(from: AgentState, to: AgentState, flow_line: FlowLine, barrier: f64) -> Self {
        Self { from, to, flow_line, energy_barrier: barrier }
    }

    /// Duration of the transition.
    pub fn duration(&self) -> f64 {
        self.to.step - self.from.step
    }

    /// Distance traveled.
    pub fn distance(&self) -> f64 {
        self.from.distance_to(&self.to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(state.dimension(), 3);
    }

    #[test]
    fn test_agent_state_distance() {
        let s1 = AgentState::new(DVector::from_vec(vec![0.0, 0.0]));
        let s2 = AgentState::new(DVector::from_vec(vec![3.0, 4.0]));
        assert!((s1.distance_to(&s2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_state_space() {
        let manifold = Manifold::sphere(2);
        let morse = MorseFunction::height_s2();
        let space = AgentStateSpace::new(manifold, morse);
        assert!(space.stable_states().is_empty() == false);
        assert_eq!(space.stable_states().len(), 1);
    }

    #[test]
    fn test_set_and_get_state() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::standard(2);
        let mut space = AgentStateSpace::new(manifold, morse);
        space.set_state(AgentState::new(DVector::from_vec(vec![1.0, 1.0])));
        assert!(space.current_state.is_some());
        assert_eq!(space.trajectory.len(), 1);
    }

    #[test]
    fn test_nearest_critical_point() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::height_torus();
        let mut space = AgentStateSpace::new(manifold, morse);
        space.set_state(AgentState::new(DVector::from_vec(vec![0.1, 0.1])));
        let nearest = space.nearest_critical_point().unwrap();
        assert_eq!(nearest.index, 0); // Nearest to the minimum
    }

    #[test]
    fn test_transition_toward() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::standard(2);
        let mut space = AgentStateSpace::new(manifold, morse);
        space.set_state(AgentState::new(DVector::from_vec(vec![1.0, 0.0])));
        let target = CriticalPoint::new(DVector::zeros(2), 0, 0.0);
        let new_state = space.transition_toward(&target, 0.5).unwrap();
        assert!((new_state.state[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_classify_region() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::height_torus();
        let mut space = AgentStateSpace::new(manifold, morse);
        space.set_state(AgentState::new(DVector::from_vec(vec![0.1, 0.1])));
        let region = space.classify_region().unwrap();
        assert_eq!(region, 0); // Near the minimum
    }

    #[test]
    fn test_gradient_descent() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::standard(2);
        let space = AgentStateSpace::new(manifold, morse);
        let start = AgentState::new(DVector::from_vec(vec![2.0, 2.0]));
        let traj = space.gradient_descent_trajectory(
            &start,
            |x| 2.0 * x.clone(), // gradient of x^2 + y^2
            0.1,
            100,
        );
        assert!(traj.len() > 1);
        let final_state = traj.last().unwrap();
        assert!(final_state.state.norm() < 0.5);
    }

    #[test]
    fn test_agent_transition() {
        let from = AgentState::new(DVector::from_vec(vec![1.0, 0.0])).at_step(0.0);
        let to = AgentState::new(DVector::zeros(2)).at_step(1.0);
        let fl = crate::gradient_flow::interpolate_flow_line(
            &CriticalPoint::new(DVector::from_vec(vec![1.0, 0.0]), 0, 0.0),
            &CriticalPoint::new(DVector::zeros(2), 0, 0.0),
            10,
        );
        let transition = AgentTransition::new(from, to, fl, 0.5);
        assert_eq!(transition.duration(), 1.0);
        assert!((transition.distance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_saddle_states() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::height_torus();
        let space = AgentStateSpace::new(manifold, morse);
        assert_eq!(space.saddle_states().len(), 2);
    }

    #[test]
    fn test_stable_states_count() {
        let manifold = Manifold::euclidean(2);
        let morse = MorseFunction::height_torus();
        let space = AgentStateSpace::new(manifold, morse);
        assert_eq!(space.stable_states().len(), 1); // One minimum
        assert_eq!(space.unstable_states().len(), 1); // One maximum
    }
}
