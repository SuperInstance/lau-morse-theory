# lau-morse-theory

Morse theory for agent state spaces — critical points, gradient flows, and handlebody decomposition.

## Features

- **Morse functions on manifolds**: Define Morse functions with critical points, compute gradients and Hessians
- **Morse lemma**: Coordinate transformation near non-degenerate critical points
- **Morse inequalities**: Weak and strong Morse inequalities with verification
- **Handlebody decomposition**: Decompose manifolds via handles attached at critical points
- **Gradient flow lines**: Numerical integration of negative gradient flows
- **Stable/unstable manifolds**: Ascending and descending manifolds of critical points
- **Morse-Smale complexes**: Cell decomposition from transverse gradient flows
- **Persistence**: Persistence diagrams and Reeb graphs from Morse functions
- **Agent state spaces**: Model agent states as manifold points with gradient flow transitions

## Usage

```rust
use lau_morse_theory::{MorseFunction, MorseInequalityResult, HandlebodyDecomposition};

// Use a predefined Morse function on the torus
let f = MorseFunction::height_torus();
assert_eq!(f.total_critical_points(), 4); // min, 2 saddles, max

// Verify Morse inequalities
let betti = BettiNumbers::new(vec![1, 2, 1]);
let result = verify_morse_inequalities(&f, &betti);
assert!(result.weak_satisfied);
assert!(result.strong_satisfied);

// Build handlebody decomposition
let hb = HandlebodyDecomposition::from_morse_function(&f);
assert_eq!(hb.euler_characteristic(), 0); // χ(T²) = 0
```

## License

MIT
