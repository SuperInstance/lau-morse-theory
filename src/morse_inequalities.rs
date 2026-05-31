//! Morse inequalities: weak and strong.
//!
//! Weak Morse inequalities: μ_k ≥ β_k for all k.
//! Strong Morse inequality: μ_k - μ_{k-1} + ... ± μ_0 ≥ β_k - β_{k-1} + ... ± β_0
//! Equality for k ≥ n: χ(M) = Σ(-1)^k μ_k = Σ(-1)^k β_k

#[allow(unused_imports)]
use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::morse_function::MorseFunction;
use crate::manifold::BettiNumbers;

/// Result of verifying Morse inequalities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseInequalityResult {
    /// Whether all weak Morse inequalities are satisfied.
    pub weak_satisfied: bool,
    /// Whether all strong Morse inequalities are satisfied.
    pub strong_satisfied: bool,
    /// Individual weak inequality results: (k, μ_k, β_k, satisfied).
    pub weak_results: Vec<(usize, usize, usize, bool)>,
    /// Individual strong inequality results: (k, lhs, rhs, satisfied).
    pub strong_results: Vec<(usize, i64, i64, bool)>,
    /// Euler characteristic from critical points.
    pub euler_from_critical: i64,
    /// Euler characteristic from Betti numbers.
    pub euler_from_betti: i64,
    /// Whether the Euler characteristic equality holds.
    pub euler_equality: bool,
}

/// Verify all Morse inequalities for a given Morse function and Betti numbers.
pub fn verify_morse_inequalities(f: &MorseFunction, betti: &BettiNumbers) -> MorseInequalityResult {
    let mu = f.count_by_index();
    let n = mu.len().max(betti.numbers.len());

    // Extend both to same length with zeros
    let mut mu_ext = mu.clone();
    mu_ext.resize(n, 0);
    let mut beta_ext = betti.numbers.clone();
    beta_ext.resize(n, 0);

    // Weak Morse inequalities: μ_k ≥ β_k
    let weak_results: Vec<(usize, usize, usize, bool)> = (0..n)
        .map(|k| {
            let m = mu_ext[k];
            let b = beta_ext[k];
            (k, m, b, m >= b)
        })
        .collect();
    let weak_satisfied = weak_results.iter().all(|&(_, _, _, s)| s);

    // Strong Morse inequalities:
    // μ_k - μ_{k-1} + ... ± μ_0 ≥ β_k - β_{k-1} + ... ± β_0
    let strong_results: Vec<(usize, i64, i64, bool)> = (0..n)
        .map(|k| {
            let lhs: i64 = (0..=k)
                .map(|j| {
                    let sign: i64 = if (k - j) % 2 == 0 { 1 } else { -1 };
                    sign * mu_ext[j] as i64
                })
                .sum();
            let rhs: i64 = (0..=k)
                .map(|j| {
                    let sign: i64 = if (k - j) % 2 == 0 { 1 } else { -1 };
                    sign * beta_ext[j] as i64
                })
                .sum();
            (k, lhs, rhs, lhs >= rhs)
        })
        .collect();
    let strong_satisfied = strong_results.iter().all(|&(_, _, _, s)| s);

    // Euler characteristic
    let euler_from_critical: i64 = mu_ext
        .iter()
        .enumerate()
        .map(|(k, &m)| if k % 2 == 0 { m as i64 } else { -(m as i64) })
        .sum();
    let euler_from_betti = betti.euler_characteristic();
    let euler_equality = euler_from_critical == euler_from_betti;

    MorseInequalityResult {
        weak_satisfied,
        strong_satisfied,
        weak_results,
        strong_results,
        euler_from_critical,
        euler_from_betti,
        euler_equality,
    }
}

/// Compute the Morse polynomial: M(t) = Σ μ_k * t^k.
pub fn morse_polynomial(mu: &[usize]) -> Vec<(usize, usize)> {
    mu.iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(k, &c)| (k, c))
        .collect()
}

/// Compute the Poincaré polynomial: P(t) = Σ β_k * t^k.
pub fn poincare_polynomial(betti: &[usize]) -> Vec<(usize, usize)> {
    betti.iter()
        .enumerate()
        .filter(|&(_, &b)| b > 0)
        .map(|(k, &b)| (k, b))
        .collect()
}

/// Strong Morse inequality: alternating sum μ_0 - μ_1 + ... ± μ_k ≥ β_0 - β_1 + ... ± β_k
pub fn strong_morse_inequality(mu: &[usize], betti: &[usize], k: usize) -> (i64, i64, bool) {
    let lhs: i64 = (0..=k)
        .map(|j| {
            let sign: i64 = if j % 2 == 0 { 1 } else { -1 };
            sign * mu.get(j).copied().unwrap_or(0) as i64
        })
        .sum();
    let rhs: i64 = (0..=k)
        .map(|j| {
            let sign: i64 = if j % 2 == 0 { 1 } else { -1 };
            sign * betti.get(j).copied().unwrap_or(0) as i64
        })
        .sum();
    (lhs, rhs, lhs >= rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weak_inequalities_s2() {
        // S^2: μ = [1, 0, 1], β = [1, 0, 1]
        let f = MorseFunction::height_s2();
        let betti = BettiNumbers::new(vec![1, 0, 1]);
        let result = verify_morse_inequalities(&f, &betti);
        assert!(result.weak_satisfied);
        assert!(result.strong_satisfied);
        assert!(result.euler_equality);
    }

    #[test]
    fn test_weak_inequalities_torus() {
        // T^2: μ = [1, 2, 1], β = [1, 2, 1]
        let f = MorseFunction::height_torus();
        let betti = BettiNumbers::new(vec![1, 2, 1]);
        let result = verify_morse_inequalities(&f, &betti);
        assert!(result.weak_satisfied);
        assert!(result.strong_satisfied);
        assert!(result.euler_equality);
    }

    #[test]
    fn test_weak_inequalities_s1() {
        // S^1: μ = [1, 1], β = [1, 1]
        let f = MorseFunction::height_s1();
        let betti = BettiNumbers::new(vec![1, 1]);
        let result = verify_morse_inequalities(&f, &betti);
        assert!(result.weak_satisfied);
        assert!(result.euler_equality);
    }

    #[test]
    fn test_euler_characteristic_s2() {
        let f = MorseFunction::height_s2();
        let mu = f.count_by_index();
        let euler: i64 = mu.iter().enumerate()
            .map(|(k, &m)| if k % 2 == 0 { m as i64 } else { -(m as i64) })
            .sum();
        assert_eq!(euler, 2); // χ(S^2) = 2
    }

    #[test]
    fn test_euler_characteristic_torus() {
        let f = MorseFunction::height_torus();
        let mu = f.count_by_index();
        let euler: i64 = mu.iter().enumerate()
            .map(|(k, &m)| if k % 2 == 0 { m as i64 } else { -(m as i64) })
            .sum();
        assert_eq!(euler, 0); // χ(T^2) = 0
    }

    #[test]
    fn test_strong_inequality_torus() {
        // T^2: μ = [1, 2, 1], β = [1, 2, 1]
        let mu = vec![1, 2, 1];
        let betti = vec![1, 2, 1];
        // k=0: 1 ≥ 1
        let (lhs, rhs, sat) = strong_morse_inequality(&mu, &betti, 0);
        assert!(sat);
        assert_eq!(lhs, 1);
        assert_eq!(rhs, 1);
        // k=1: 1 - 2 = -1 ≥ 1 - 2 = -1
        let (lhs, rhs, sat) = strong_morse_inequality(&mu, &betti, 1);
        assert!(sat);
        // k=2: 1 - 2 + 1 = 0 ≥ 1 - 2 + 1 = 0
        let (lhs, rhs, sat) = strong_morse_inequality(&mu, &betti, 2);
        assert!(sat);
        assert_eq!(lhs, 0);
    }

    #[test]
    fn test_morse_polynomial() {
        let mp = morse_polynomial(&[1, 2, 1]);
        assert_eq!(mp, vec![(0, 1), (1, 2), (2, 1)]);
    }

    #[test]
    fn test_poincare_polynomial() {
        let pp = poincare_polynomial(&[1, 2, 1]);
        assert_eq!(pp, vec![(0, 1), (1, 2), (2, 1)]);
    }

    #[test]
    fn test_complex_projective_plane() {
        // CP^2: betti = [1, 0, 1, 0, 1]
        // Morse function with μ = [1, 0, 2, 0, 1]
        let f = MorseFunction::from_critical_points("CP2", 4, vec![
            crate::critical_point::CriticalPoint::new(DVector::zeros(4), 0, 0.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(4), 2, 1.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(4), 2, 2.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(4), 4, 3.0),
        ]);
        let betti = BettiNumbers::new(vec![1, 0, 1, 0, 1]);
        let result = verify_morse_inequalities(&f, &betti);
        assert!(result.weak_satisfied);
        // μ_2 = 2 ≥ β_2 = 1 (strict inequality)
        assert!(result.weak_results[2].1 > result.weak_results[2].2);
        assert_eq!(result.euler_from_critical, 1 - 0 + 2 - 0 + 1); // = 4... 
        // Actually chi(CP^2) = 1 - 0 + 1 - 0 + 1 = 3
        // and from critical points: 1 - 0 + 2 - 0 + 1 = 4 ≠ 3
        // This would fail Euler equality, which is correct since our mu doesn't match
    }

    #[test]
    fn test_weak_with_extra_critical_points() {
        // A function with more critical points than the minimum (still satisfies weak inequalities)
        let f = MorseFunction::from_critical_points("extra", 2, vec![
            crate::critical_point::CriticalPoint::new(DVector::zeros(2), 0, 0.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(2), 0, 1.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(2), 1, 2.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(2), 1, 3.0),
            crate::critical_point::CriticalPoint::new(DVector::zeros(2), 2, 4.0),
        ]);
        let betti = BettiNumbers::new(vec![1, 0, 1]); // S^2 betti
        let result = verify_morse_inequalities(&f, &betti);
        assert!(result.weak_satisfied); // 2 ≥ 1, 2 ≥ 0, 1 ≥ 1
    }
}
