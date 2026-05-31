//! Morse-Smale complexes from transverse gradient flows.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::critical_point::CriticalPoint;
use crate::gradient_flow::FlowLine;

/// A Morse-Smale complex cell, dual to the stable/unstable manifold decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseSmaleCell {
    /// The critical point this cell is associated with.
    pub critical_point: CriticalPoint,
    /// Dimension of the cell.
    pub dimension: usize,
    /// Vertices (0-dimensional critical points connected to this cell).
    pub vertices: Vec<DVector<f64>>,
}

/// A Morse-Smale complex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseSmaleComplex {
    /// Dimension of the manifold.
    pub dimension: usize,
    /// Cells of the complex, one per critical point.
    pub cells: Vec<MorseSmaleCell>,
    /// Flow lines connecting critical points of adjacent index.
    pub flow_lines: Vec<FlowLine>,
}

impl MorseSmaleComplex {
    /// Build a Morse-Smale complex from a sorted list of critical points.
    pub fn from_critical_points(dimension: usize, critical_points: &[CriticalPoint]) -> Self {
        let cells: Vec<MorseSmaleCell> = critical_points
            .iter()
            .map(|cp| MorseSmaleCell {
                dimension: cp.index,
                critical_point: cp.clone(),
                vertices: Vec::new(),
            })
            .collect();

        // Generate flow lines between critical points of adjacent index
        let mut flow_lines = Vec::new();
        for (i, cp_high) in critical_points.iter().enumerate() {
            for (j, cp_low) in critical_points.iter().enumerate() {
                if i != j && cp_high.index == cp_low.index + 1 && cp_high.value > cp_low.value {
                    flow_lines.push(FlowLine::new(
                        cp_high.clone(),
                        cp_low.clone(),
                        vec![cp_high.position.clone(), cp_low.position.clone()],
                    ));
                }
            }
        }

        MorseSmaleComplex {
            dimension,
            cells,
            flow_lines,
        }
    }

    /// Cells of a given dimension.
    pub fn cells_of_dimension(&self, dim: usize) -> Vec<&MorseSmaleCell> {
        self.cells.iter().filter(|c| c.dimension == dim).collect()
    }

    /// Count of cells per dimension.
    pub fn cell_counts(&self) -> Vec<usize> {
        let max_dim = self.cells.iter().map(|c| c.dimension).max().unwrap_or(0);
        (0..=max_dim)
            .map(|d| self.cells_of_dimension(d).len())
            .collect()
    }

    /// Euler characteristic of the complex.
    pub fn euler_characteristic(&self) -> i64 {
        self.cells
            .iter()
            .map(|c| if c.dimension % 2 == 0 { 1i64 } else { -1i64 })
            .sum()
    }

    /// Number of flow lines.
    pub fn num_flow_lines(&self) -> usize {
        self.flow_lines.len()
    }

    /// Check if the Smale transversality condition holds.
    /// (Simplified: check that flow lines connect critical points of adjacent index.)
    pub fn is_transverse(&self) -> bool {
        self.flow_lines.iter().all(|fl| {
            fl.source.index == fl.target.index + 1
        })
    }

    /// Boundary operator: for each k-cell, list the (k-1)-cells in its boundary.
    pub fn boundary(&self, k: usize) -> Vec<Vec<usize>> {
        let k_cells: Vec<usize> = self.cells.iter()
            .enumerate()
            .filter(|(_, c)| c.dimension == k)
            .map(|(i, _)| i)
            .collect();

        let km1_cells: Vec<usize> = self.cells.iter()
            .enumerate()
            .filter(|(_, c)| c.dimension == k - 1)
            .map(|(i, _)| i)
            .collect();

        k_cells.iter().map(|&kc| {
            self.flow_lines.iter()
                .enumerate()
                .filter(|(_, fl)| fl.source.index == k && fl.target.index == k - 1)
                .filter(|(_, fl)| {
                    (fl.source.position.clone() - self.cells[kc].critical_point.position.clone()).norm() < 1e-6
                })
                .filter_map(|(_, fl)| {
                    km1_cells.iter().find(|&&idx| {
                        (fl.target.position.clone() - self.cells[idx].critical_point.position.clone()).norm() < 1e-6
                    }).copied()
                })
                .collect()
        }).collect()
    }
}

/// Boundary matrix for the Morse-Smale complex.
/// Used for computing homology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryMatrix {
    /// The matrix entries: boundary_matrix[i][j] = 1 if cell j is in boundary of cell i.
    pub matrix: Vec<Vec<usize>>,
    /// Dimensions of each cell (indexed by column).
    pub dimensions: Vec<usize>,
}

impl BoundaryMatrix {
    /// Construct the boundary matrix from a Morse-Smale complex.
    pub fn from_complex(complex: &MorseSmaleComplex) -> Self {
        let n = complex.cells.len();
        let mut matrix = vec![vec![0usize; n]; n];
        let dimensions: Vec<usize> = complex.cells.iter().map(|c| c.dimension).collect();

        for fl in &complex.flow_lines {
            let src_idx = complex.cells.iter().position(|c| {
                (c.critical_point.position.clone() - fl.source.position.clone()).norm() < 1e-6
                    && c.dimension == fl.source.index
            });
            let tgt_idx = complex.cells.iter().position(|c| {
                (c.critical_point.position.clone() - fl.target.position.clone()).norm() < 1e-6
                    && c.dimension == fl.target.index
            });
            if let (Some(si), Some(ti)) = (src_idx, tgt_idx) {
                matrix[si][ti] += 1;
            }
        }

        BoundaryMatrix { matrix, dimensions }
    }

    /// Compute Betti numbers via Smith normal form (simplified).
    /// Uses Z/2Z homology for simplicity.
    pub fn betti_numbers_z2(&self) -> Vec<usize> {
        let n = self.matrix.len();
        if n == 0 {
            return vec![1]; // Empty complex = point
        }

        let max_dim = *self.dimensions.iter().max().unwrap_or(&0);

        // Convert to Z/2Z
        let mut mat: Vec<Vec<u8>> = self.matrix.iter()
            .map(|row| row.iter().map(|&v| (v % 2) as u8).collect())
            .collect();

        // Gaussian elimination (Z/2Z)
        let mut rank = 0;
        let mut pivot_col = vec![None; n];
        
        for col in 0..n {
            // Find a row with 1 in this column
            let mut found = None;
            for row in rank..n {
                if mat[row][col] == 1 {
                    found = Some(row);
                    break;
                }
            }
            if let Some(row) = found {
                // Swap
                mat.swap(rank, row);
                pivot_col[col] = Some(rank);
                // Eliminate
                for r in 0..n {
                    if r != rank && mat[r][col] == 1 {
                        for c in 0..n {
                            mat[r][c] ^= mat[rank][c];
                        }
                    }
                }
                rank += 1;
            }
        }

        // Count cycles per dimension
        let mut betti = vec![0usize; max_dim + 1];
        let mut boundary_rank = vec![0usize; max_dim + 2];
        let mut cycle_rank = vec![0usize; max_dim + 1];

        for col in 0..n {
            let dim = self.dimensions[col];
            if pivot_col[col].is_some() {
                boundary_rank[dim] += 1;
            }
        }

        for col in 0..n {
            let dim = self.dimensions[col];
            cycle_rank[dim] += 1;
            if pivot_col[col].is_some() {
                cycle_rank[dim] -= 1;
            }
        }

        for k in 0..=max_dim {
            betti[k] = cycle_rank.get(k).copied().unwrap_or(0)
                .saturating_sub(boundary_rank.get(k + 1).copied().unwrap_or(0));
        }

        betti
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morse_smale_s1() {
        let cps = vec![
            CriticalPoint::new(DVector::from_vec(vec![0.0]), 0, -1.0),
            CriticalPoint::new(DVector::from_vec(vec![1.0]), 1, 1.0),
        ];
        let ms = MorseSmaleComplex::from_critical_points(1, &cps);
        assert_eq!(ms.cells.len(), 2);
        assert_eq!(ms.cell_counts(), vec![1, 1]);
        assert_eq!(ms.euler_characteristic(), 0);
    }

    #[test]
    fn test_morse_smale_s2() {
        let cps = vec![
            CriticalPoint::new(DVector::zeros(2), 0, -1.0),
            CriticalPoint::new(DVector::from_vec(vec![1.0, 1.0]), 2, 1.0),
        ];
        let ms = MorseSmaleComplex::from_critical_points(2, &cps);
        assert_eq!(ms.cells.len(), 2);
        assert_eq!(ms.euler_characteristic(), 1 + 1); // 2
    }

    #[test]
    fn test_morse_smale_torus() {
        let cps = vec![
            CriticalPoint::new(DVector::from_vec(vec![0.0, 0.0]), 0, 0.0),
            CriticalPoint::new(DVector::from_vec(vec![1.0, 0.0]), 1, 1.0),
            CriticalPoint::new(DVector::from_vec(vec![0.0, 1.0]), 1, 1.5),
            CriticalPoint::new(DVector::from_vec(vec![1.0, 1.0]), 2, 2.0),
        ];
        let ms = MorseSmaleComplex::from_critical_points(2, &cps);
        assert_eq!(ms.cells.len(), 4);
        assert_eq!(ms.cell_counts(), vec![1, 2, 1]);
        assert_eq!(ms.euler_characteristic(), 0);
    }

    #[test]
    fn test_transversality() {
        let cps = vec![
            CriticalPoint::new(DVector::zeros(2), 0, 0.0),
            CriticalPoint::new(DVector::from_vec(vec![1.0, 0.0]), 1, 1.0),
            CriticalPoint::new(DVector::from_vec(vec![2.0, 2.0]), 2, 3.0),
        ];
        let ms = MorseSmaleComplex::from_critical_points(2, &cps);
        assert!(ms.is_transverse());
    }

    #[test]
    fn test_flow_lines_adjacent_index() {
        let cps = vec![
            CriticalPoint::new(DVector::zeros(2), 0, 0.0),
            CriticalPoint::new(DVector::from_vec(vec![1.0, 0.0]), 1, 1.0),
            CriticalPoint::new(DVector::from_vec(vec![2.0, 2.0]), 2, 3.0),
        ];
        let ms = MorseSmaleComplex::from_critical_points(2, &cps);
        // 1→0 and 2→1 flows
        assert_eq!(ms.num_flow_lines(), 2);
    }

    #[test]
    fn test_boundary_matrix() {
        let cps = vec![
            CriticalPoint::new(DVector::from_vec(vec![0.0]), 0, 0.0),
            CriticalPoint::new(DVector::from_vec(vec![1.0]), 1, 1.0),
        ];
        let ms = MorseSmaleComplex::from_critical_points(1, &cps);
        let bm = BoundaryMatrix::from_complex(&ms);
        assert_eq!(bm.dimensions, vec![0, 1]);
    }

    #[test]
    fn test_betti_z2_simple() {
        // A single point
        let cps = vec![CriticalPoint::new(DVector::zeros(1), 0, 0.0)];
        let ms = MorseSmaleComplex::from_critical_points(1, &cps);
        let bm = BoundaryMatrix::from_complex(&ms);
        let betti = bm.betti_numbers_z2();
        assert_eq!(betti[0], 1);
    }
}
