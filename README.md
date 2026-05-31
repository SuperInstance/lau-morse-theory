# lau-morse-theory

Critical points, gradient flows, and handlebody decomposition for manifolds — including agent state spaces.

A Morse function tells you the shape of a space by counting where it's flat. Every peak, valley, and saddle point is a critical point, and the *index* of each one tells you what dimension of "handle" to glue on. Morse theory is how you reconstruct a whole manifold from a single well-chosen function.

Like a hermit crab surveying the ocean floor — every bump and depression tells it something about the terrain ahead.

## The math in 60 seconds

A **Morse function** f: M → ℝ is a smooth function whose critical points are all non-degenerate (the Hessian is full-rank). At each critical point p, the **index** λ(p) is the number of negative eigenvalues of the Hessian. Morse theory tells us:

- **Weak Morse inequalities:** cₖ ≥ bₖ (number of index-k critical points ≥ k-th Betti number)
- **Strong Morse inequalities:** c₀ - c₁ + c₂ - ... gives the Euler characteristic
- **Handle decomposition:** M is built by attaching one handle of dimension λ(p) per critical point
- **Morse-Smale complex:** gradient flow lines connecting critical points of adjacent index

References: Milnor, *Morse Theory* (1963); Matsumoto, *An Introduction to Morse Theory* (2002)

## Quick start

```rust
use lau_morse_theory::{Manifold, MorseFunction, MorseInequalities};

// Create a 2-sphere S²
let sphere = Manifold::sphere(2);

// Define a height function (standard Morse function on S²)
let morse = MorseFunction::height_function(&sphere);

// Find critical points
let critical = morse.critical_points();
// S² with height function has: 1 minimum (index 0), 1 maximum (index 2)

// Verify Morse inequalities against Betti numbers
let betti = sphere.betti_numbers(); // [1, 0, 1]
let verified = MorseInequalities::verify(&critical, &betti);
assert!(verified.weak);   // cₖ ≥ bₖ for all k
assert!(verified.strong); // alternating sums match
```

## Key types

| Type | What it is |
|------|-----------|
| `Manifold` | A differentiable manifold (sphere, torus, CPⁿ, or custom) |
| `MorseFunction` | A smooth function with non-degenerate critical points |
| `CriticalPoint` | A point where ∇f = 0, with index, position, and Hessian |
| `HandleBody` | Decomposition of M into handles attached at critical points |
| `GradientFlow` | Flow lines of ∇f connecting critical points |
| `MorseSmaleComplex` | Cells formed by stable/unstable manifold intersections |
| `PersistenceDiagram` | Birth-death pairs from sublevel set filtration |
| `AgentStateSpace` | Agent states as a manifold with energy landscape as Morse function |

## Contributing

Found a bug? Have a cool application of Morse theory to agent systems? [Open an issue](https://github.com/SuperInstance/lau-morse-theory/issues) or send a PR. We're especially interested in:

- Numerical stability improvements for Hessian computation
- New manifold types (projective spaces, Grassmannians)
- Visualization tools for gradient flows
- Applications to optimization landscapes in ML
