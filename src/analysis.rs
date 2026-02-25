use crate::maze::Maze;
use crate::solver::{solve_bfs, Solution};

/// Structural analysis of a maze.
pub struct MazeAnalysis {
    pub size: usize,
    pub total_cells: usize,
    /// Number of dead ends (cells with exactly one open passage).
    pub dead_ends: usize,
    /// Number of corridor cells (exactly two open passages).
    pub corridors: usize,
    /// Number of junction cells (three or more open passages).
    pub junctions: usize,
    /// Solution from top-left to bottom-right.
    pub solution: Option<Solution>,
    /// Distribution of passage counts per cell: index = passage count, value = cell count.
    pub passage_distribution: Vec<usize>,
}

impl MazeAnalysis {
    /// Dead end ratio: fraction of cells that are dead ends.
    pub fn dead_end_ratio(&self) -> f64 {
        self.dead_ends as f64 / self.total_cells as f64
    }

    /// Average branching factor: mean passages per cell.
    pub fn avg_branching(&self) -> f64 {
        let total_passages: usize = self.passage_distribution
            .iter()
            .enumerate()
            .map(|(count, &cells)| count * cells)
            .sum();
        total_passages as f64 / self.total_cells as f64
    }

    /// Solution length (number of cells in shortest path), or None if unsolved.
    pub fn solution_length(&self) -> Option<usize> {
        self.solution.as_ref().map(|s| s.length())
    }

    /// Tortuosity of solution path.
    pub fn tortuosity(&self) -> Option<f64> {
        self.solution.as_ref().map(|s| s.tortuosity())
    }

    /// Turn count along solution path.
    pub fn turn_count(&self) -> Option<usize> {
        self.solution.as_ref().map(|s| s.turn_count())
    }

    /// Difficulty score: composite metric combining tortuosity, dead end ratio,
    /// and junction density. Higher = harder to solve by a human.
    /// Range roughly 0-100.
    pub fn difficulty_score(&self) -> f64 {
        let tortuosity = self.tortuosity().unwrap_or(1.0);
        let junction_ratio = self.junctions as f64 / self.total_cells as f64;
        let turn_density = self.turn_count().unwrap_or(0) as f64
            / self.solution_length().unwrap_or(1) as f64;

        // Tortuosity contribution (40%): higher tortuosity = longer path = harder
        let tort_score = ((tortuosity - 1.0) * 20.0).min(40.0);

        // Junction contribution (30%): more junctions = more wrong choices
        let junction_score = (junction_ratio * 100.0).min(30.0);

        // Turn density (30%): more turns per step = harder to follow
        let turn_score = (turn_density * 40.0).min(30.0);

        tort_score + junction_score + turn_score
    }

    /// Print a human-readable analysis report.
    pub fn print_report(&self) {
        println!("=== Maze Analysis ({0}x{0}) ===", self.size);
        println!("  Cells:       {}", self.total_cells);
        println!("  Dead ends:   {} ({:.1}%)", self.dead_ends, self.dead_end_ratio() * 100.0);
        println!("  Corridors:   {}", self.corridors);
        println!("  Junctions:   {}", self.junctions);
        println!("  Avg branch:  {:.2}", self.avg_branching());
        println!("  Passages:    {:?}", self.passage_distribution);
        if let Some(ref sol) = self.solution {
            println!("  Solution:    {} steps", sol.length());
            println!("  Tortuosity:  {:.2}", sol.tortuosity());
            println!("  Turns:       {}", sol.turn_count());
            println!("  Difficulty:  {:.1}/100", self.difficulty_score());
        } else {
            println!("  Solution:    none found");
        }
    }
}

/// Count open passages for a cell at (x, y).
fn passage_count(maze: &Maze, x: usize, y: usize) -> usize {
    maze.get_open_adj(x, y).len()
}

/// Analyze a maze and return structural metrics.
pub fn analyze(maze: &Maze) -> MazeAnalysis {
    let size = maze.get_size();
    let total_cells = size * size;

    // Count passages per cell
    let mut passage_distribution = vec![0usize; 5]; // 0-4 passages possible
    let mut dead_ends = 0;
    let mut corridors = 0;
    let mut junctions = 0;

    for y in 0..size {
        for x in 0..size {
            let p = passage_count(maze, x, y);
            passage_distribution[p] += 1;
            match p {
                0 | 1 => dead_ends += 1,
                2 => corridors += 1,
                _ => junctions += 1,
            }
        }
    }

    // Solve from top-left to bottom-right
    let solution = if size > 0 {
        solve_bfs(maze, (0, 0), (size - 1, size - 1))
    } else {
        None
    };

    MazeAnalysis {
        size,
        total_cells,
        dead_ends,
        corridors,
        junctions,
        solution,
        passage_distribution,
    }
}

/// Compare all algorithms on the same size maze and print a summary table.
pub fn compare_algorithms(size: usize, trials: usize) {
    use crate::algos::{get_algorithm, ALGORITHMS};

    println!("=== Algorithm Comparison ({0}x{0}, {1} trials each) ===", size, trials);
    println!("{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Algorithm", "DeadEnd%", "AvgBrnch", "SolLen", "Tortuous", "Diffi.");
    println!("{}", "-".repeat(60));

    for algo_name in &ALGORITHMS {
        let mut total_dead = 0.0;
        let mut total_branch = 0.0;
        let mut total_sol = 0.0;
        let mut total_tort = 0.0;
        let mut total_diff = 0.0;

        for _ in 0..trials {
            let mut m = Maze::new(size);
            let mut a = get_algorithm(&algo_name.to_string(), false).unwrap();
            m.apply_algorithm(&mut a).unwrap();
            let analysis = analyze(&m);
            total_dead += analysis.dead_end_ratio();
            total_branch += analysis.avg_branching();
            total_sol += analysis.solution_length().unwrap_or(0) as f64;
            total_tort += analysis.tortuosity().unwrap_or(1.0);
            total_diff += analysis.difficulty_score();
        }

        let n = trials as f64;
        println!("{:<12} {:>7.1}% {:>8.2} {:>8.1} {:>8.2} {:>8.1}",
            algo_name,
            total_dead / n * 100.0,
            total_branch / n,
            total_sol / n,
            total_tort / n,
            total_diff / n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::get_algorithm;

    fn make_maze(algo: &str, size: usize) -> Maze {
        let mut m = Maze::new(size);
        let mut a = get_algorithm(&algo.to_string(), false).unwrap();
        m.apply_algorithm(&mut a).unwrap();
        m
    }

    #[test]
    fn test_analysis_cell_counts() {
        let m = make_maze("dfs", 20);
        let a = analyze(&m);
        assert_eq!(a.total_cells, 400);
        assert_eq!(a.dead_ends + a.corridors + a.junctions, 400);
        assert_eq!(
            a.passage_distribution.iter().sum::<usize>(),
            400,
            "passage distribution doesn't sum to total cells"
        );
    }

    #[test]
    fn test_perfect_maze_no_isolated_cells() {
        // In a perfect maze, every cell has at least 1 passage
        let m = make_maze("kruskals", 15);
        let a = analyze(&m);
        assert_eq!(a.passage_distribution[0], 0, "isolated cells found");
    }

    #[test]
    fn test_analysis_has_solution() {
        for algo in &["dfs", "wilsons", "prims"] {
            let m = make_maze(algo, 15);
            let a = analyze(&m);
            assert!(a.solution.is_some(), "no solution for {}", algo);
        }
    }

    #[test]
    fn test_difficulty_nonnegative() {
        let m = make_maze("dfs", 10);
        let a = analyze(&m);
        assert!(a.difficulty_score() >= 0.0);
    }

    #[test]
    fn test_algorithms_produce_different_structures() {
        // Different algorithms produce mazes with measurably different properties.
        // DFS creates long winding corridors (low dead-end ratio).
        // Kruskal's creates short branchy passages (high dead-end ratio).
        let trials = 20;
        let size = 30;
        let mut dfs_dead = 0.0;
        let mut kruskal_dead = 0.0;
        for _ in 0..trials {
            dfs_dead += analyze(&make_maze("dfs", size)).dead_end_ratio();
            kruskal_dead += analyze(&make_maze("kruskals", size)).dead_end_ratio();
        }
        let dfs_avg = dfs_dead / trials as f64;
        let kruskal_avg = kruskal_dead / trials as f64;
        let diff = (dfs_avg - kruskal_avg).abs();
        assert!(
            diff > 0.05,
            "expected different dead-end ratios: DFS={:.3} Kruskal's={:.3} (diff={:.3})",
            dfs_avg, kruskal_avg, diff,
        );
    }
}
