//! # lau-morse-theory
//!
//! Morse theory for agent state spaces — critical points, gradient flows,
//! and handlebody decomposition.
//!
//! This crate provides tools to analyze manifolds (including agent state spaces)
//! via Morse functions, studying critical points, their indices, gradient flows,
//! stable/unstable manifolds, and the resulting topological invariants.

pub mod manifold;
pub mod morse_function;
pub mod critical_point;
pub mod morse_lemma;
pub mod morse_inequalities;
pub mod handlebody;
pub mod gradient_flow;
pub mod morse_smale;
pub mod persistence;
pub mod agent_state;

pub use manifold::*;
pub use morse_function::*;
pub use critical_point::*;
pub use morse_lemma::*;
pub use morse_inequalities::*;
pub use handlebody::*;
pub use gradient_flow::*;
pub use morse_smale::*;
pub use persistence::*;
pub use agent_state::*;
