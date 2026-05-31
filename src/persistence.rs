//! Persistence from Morse functions via Reeb graphs and sublevel set persistence.

use serde::{Serialize, Deserialize};
use crate::critical_point::CriticalPoint;
use crate::morse_function::MorseFunction;

/// A persistence bar: interval [birth, death) for a topological feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceBar {
    /// Dimension of the feature (homology degree).
    pub dimension: usize,
    /// Birth value (function value where feature appears).
    pub birth: f64,
    /// Death value (function value where feature disappears). None = infinite persistence.
    pub death: Option<f64>,
    /// Birth critical point index.
    pub birth_index: usize,
    /// Death critical point index.
    pub death_index: Option<usize>,
}

impl PersistenceBar {
    /// Persistence length (death - birth). Infinity for essential features.
    pub fn persistence(&self) -> Option<f64> {
        self.death.map(|d| d - self.birth)
    }

    /// Is this an essential feature (infinite persistence)?
    pub fn is_essential(&self) -> bool {
        self.death.is_none()
    }

    /// Is this a non-essential feature with finite persistence?
    pub fn is_finite(&self) -> bool {
        self.death.is_some()
    }
}

/// A persistence diagram: collection of persistence bars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    /// Name of the function.
    pub function_name: String,
    /// Persistence bars.
    pub bars: Vec<PersistenceBar>,
}

impl PersistenceDiagram {
    /// Compute persistence from a Morse function's critical points.
    ///
    /// Uses the standard Morse persistence algorithm:
    /// - A critical point of index k births a k-dimensional feature
    /// - A critical point of index (k+1) can kill a k-dimensional feature
    pub fn from_morse_function(f: &MorseFunction) -> Self {
        let mut sorted_cps: Vec<&CriticalPoint> = f.critical_points.iter().collect();
        sorted_cps.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));

        let mut bars = Vec::new();
        let mut active: Vec<(usize, &CriticalPoint)> = Vec::new(); // (index, cp) pairs, sorted by value

        for cp in &sorted_cps {
            if cp.index % 2 == 0 {
                // Even index: birth
                active.push((cp.index, cp));
            } else {
                // Odd index: try to kill a feature
                if let Some(pos) = active.iter().rposition(|&(idx, _)| idx + 1 == cp.index) {
                    let (_, birth_cp) = active.remove(pos);
                    bars.push(PersistenceBar {
                        dimension: birth_cp.index,
                        birth: birth_cp.value,
                        death: Some(cp.value),
                        birth_index: birth_cp.index,
                        death_index: Some(cp.index),
                    });
                }
            }
        }

        // Remaining active features are essential (infinite persistence)
        for (_, cp) in active {
            bars.push(PersistenceBar {
                dimension: cp.index,
                birth: cp.value,
                death: None,
                birth_index: cp.index,
                death_index: None,
            });
        }

        PersistenceDiagram {
            function_name: f.name.clone(),
            bars,
        }
    }

    /// Bars of a given dimension.
    pub fn bars_of_dimension(&self, dim: usize) -> Vec<&PersistenceBar> {
        self.bars.iter().filter(|b| b.dimension == dim).collect()
    }

    /// Number of essential features.
    pub fn num_essential(&self) -> usize {
        self.bars.iter().filter(|b| b.is_essential()).count()
    }

    /// Maximum finite persistence value.
    pub fn max_finite_persistence(&self) -> Option<f64> {
        self.bars.iter().filter_map(|b| b.persistence()).max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Betti numbers from the persistence diagram (counting essential features).
    pub fn betti_numbers(&self) -> Vec<usize> {
        let max_dim = self.bars.iter().map(|b| b.dimension).max().unwrap_or(0);
        (0..=max_dim)
            .map(|k| self.bars.iter().filter(|b| b.dimension == k && b.is_essential()).count())
            .collect()
    }
}

/// A Reeb graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReebNode {
    /// Function value.
    pub value: f64,
    /// Associated critical point index.
    pub critical_index: usize,
    /// Topological type (number of connected components in sublevel set).
    pub components: usize,
}

/// A Reeb graph edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReebEdge {
    /// Source node.
    pub source: usize,
    /// Target node.
    pub target: usize,
}

/// A Reeb graph of a Morse function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReebGraph {
    /// Nodes.
    pub nodes: Vec<ReebNode>,
    /// Edges.
    pub edges: Vec<ReebEdge>,
}

impl ReebGraph {
    /// Construct a Reeb graph from a Morse function.
    pub fn from_morse_function(f: &MorseFunction) -> Self {
        let mut sorted_cps: Vec<&CriticalPoint> = f.critical_points.iter().collect();
        sorted_cps.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut components = 0;

        for (i, cp) in sorted_cps.iter().enumerate() {
            match cp.index {
                0 => {
                    // Minimum: new component
                    components += 1;
                    nodes.push(ReebNode {
                        value: cp.value,
                        critical_index: 0,
                        components,
                    });
                    if i > 0 {
                        edges.push(ReebEdge { source: i - 1, target: i });
                    }
                }
                _ if cp.index == f.dimension => {
                    // Maximum: merge components
                    components = components.saturating_sub(1).max(1);
                    nodes.push(ReebNode {
                        value: cp.value,
                        critical_index: f.dimension,
                        components,
                    });
                    edges.push(ReebEdge { source: i - 1, target: i });
                }
                _ => {
                    // Saddle
                    if cp.index % 2 == 1 {
                        components = components.saturating_sub(1).max(1);
                    } else {
                        components += 1;
                    }
                    nodes.push(ReebNode {
                        value: cp.value,
                        critical_index: cp.index,
                        components,
                    });
                    edges.push(ReebEdge { source: i - 1, target: i });
                }
            }
        }

        ReebGraph { nodes, edges }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistence_s1() {
        let f = MorseFunction::height_s1();
        let pd = PersistenceDiagram::from_morse_function(&f);
        // S^1 has 1 essential H_0 feature and 1 essential H_1 feature
        assert!(pd.num_essential() >= 1);
    }

    #[test]
    fn test_persistence_s2() {
        let f = MorseFunction::height_s2();
        let pd = PersistenceDiagram::from_morse_function(&f);
        // S^2: min (index 0) and max (index 2), both essential
        assert_eq!(pd.num_essential(), 2);
    }

    #[test]
    fn test_persistence_bars() {
        let bar = PersistenceBar {
            dimension: 0,
            birth: 0.0,
            death: Some(1.0),
            birth_index: 0,
            death_index: Some(1),
        };
        assert!(!bar.is_essential());
        assert!(bar.is_finite());
        assert_eq!(bar.persistence(), Some(1.0));
    }

    #[test]
    fn test_essential_bar() {
        let bar = PersistenceBar {
            dimension: 0,
            birth: 0.0,
            death: None,
            birth_index: 0,
            death_index: None,
        };
        assert!(bar.is_essential());
        assert!(!bar.is_finite());
        assert_eq!(bar.persistence(), None);
    }

    #[test]
    fn test_reeb_graph_s2() {
        let f = MorseFunction::height_s2();
        let rg = ReebGraph::from_morse_function(&f);
        assert_eq!(rg.num_nodes(), 2);
        assert_eq!(rg.num_edges(), 1);
    }

    #[test]
    fn test_reeb_graph_torus() {
        let f = MorseFunction::height_torus();
        let rg = ReebGraph::from_morse_function(&f);
        assert_eq!(rg.num_nodes(), 4);
    }

    #[test]
    fn test_betti_from_persistence_s2() {
        let f = MorseFunction::height_s2();
        let pd = PersistenceDiagram::from_morse_function(&f);
        let betti = pd.betti_numbers();
        assert!(betti.len() >= 1);
        assert_eq!(betti[0], 1);
    }

    #[test]
    fn test_persistence_diagram_dimension_filter() {
        let f = MorseFunction::height_torus();
        let pd = PersistenceDiagram::from_morse_function(&f);
        let dim0 = pd.bars_of_dimension(0);
        assert!(!dim0.is_empty());
    }
}
