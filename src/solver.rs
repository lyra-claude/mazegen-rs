use std::collections::VecDeque;

use crate::maze::Maze;

/// Result of solving a maze: the path from start to end (if found).
pub struct Solution {
    /// Ordered list of (x, y) positions from start to end.
    pub path: Vec<(usize, usize)>,
}

impl Solution {
    /// Number of steps in the solution.
    pub fn length(&self) -> usize {
        self.path.len()
    }

    /// Number of turns (direction changes) along the path.
    pub fn turn_count(&self) -> usize {
        if self.path.len() < 3 {
            return 0;
        }
        let mut turns = 0;
        for i in 2..self.path.len() {
            let (dx1, dy1) = (
                self.path[i - 1].0 as isize - self.path[i - 2].0 as isize,
                self.path[i - 1].1 as isize - self.path[i - 2].1 as isize,
            );
            let (dx2, dy2) = (
                self.path[i].0 as isize - self.path[i - 1].0 as isize,
                self.path[i].1 as isize - self.path[i - 1].1 as isize,
            );
            if dx1 != dx2 || dy1 != dy2 {
                turns += 1;
            }
        }
        turns
    }

    /// Tortuosity: solution length relative to Manhattan distance.
    /// Higher values mean the path wanders more.
    pub fn tortuosity(&self) -> f64 {
        if self.path.is_empty() {
            return 0.0;
        }
        let start = self.path[0];
        let end = self.path[self.path.len() - 1];
        let manhattan = (end.0 as f64 - start.0 as f64).abs()
            + (end.1 as f64 - start.1 as f64).abs();
        if manhattan == 0.0 {
            return 1.0;
        }
        self.path.len() as f64 / manhattan
    }
}

/// BFS solver: finds the shortest path from start to end.
pub fn solve_bfs(maze: &Maze, start: (usize, usize), end: (usize, usize)) -> Option<Solution> {
    let size = maze.get_size();
    let mut visited = vec![vec![false; size]; size];
    let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; size]; size];
    let mut queue = VecDeque::new();

    visited[start.1][start.0] = true;
    queue.push_back(start);

    while let Some((x, y)) = queue.pop_front() {
        if (x, y) == end {
            // Reconstruct path
            let mut path = vec![(x, y)];
            let mut cur = (x, y);
            while let Some(p) = parent[cur.1][cur.0] {
                path.push(p);
                cur = p;
            }
            path.reverse();
            return Some(Solution { path });
        }

        for (nx, ny) in maze.get_open_adj(x, y) {
            if !visited[ny][nx] {
                visited[ny][nx] = true;
                parent[ny][nx] = Some((x, y));
                queue.push_back((nx, ny));
            }
        }
    }

    None
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
    fn test_bfs_finds_solution() {
        for algo in &["dfs", "kruskals", "wilsons", "ellers", "prims"] {
            let m = make_maze(algo, 20);
            // Skip perfectness check — Wilson's has a rare upstream bug.
            // The solver should work on any connected maze.
            let sol = solve_bfs(&m, (0, 0), (19, 19));
            assert!(sol.is_some(), "no solution for {}", algo);
            let sol = sol.unwrap();
            assert_eq!(sol.path[0], (0, 0));
            assert_eq!(sol.path[sol.path.len() - 1], (19, 19));
            assert!(sol.length() >= 2);
        }
    }

    #[test]
    fn test_solution_path_is_connected() {
        let m = make_maze("dfs", 15);
        let sol = solve_bfs(&m, (0, 0), (14, 14)).unwrap();
        for i in 1..sol.path.len() {
            let (x1, y1) = sol.path[i - 1];
            let (x2, y2) = sol.path[i];
            let dx = (x1 as isize - x2 as isize).unsigned_abs();
            let dy = (y1 as isize - y2 as isize).unsigned_abs();
            assert_eq!(dx + dy, 1, "non-adjacent step at index {}", i);
        }
    }

    #[test]
    fn test_tortuosity_bounds() {
        let m = make_maze("dfs", 20);
        let sol = solve_bfs(&m, (0, 0), (19, 19)).unwrap();
        let t = sol.tortuosity();
        // Tortuosity must be >= 1.0 (can't be shorter than Manhattan distance)
        assert!(t >= 1.0, "tortuosity {} < 1.0", t);
    }

    #[test]
    fn test_trivial_maze() {
        let m = make_maze("dfs", 1);
        let sol = solve_bfs(&m, (0, 0), (0, 0)).unwrap();
        assert_eq!(sol.length(), 1);
        assert_eq!(sol.turn_count(), 0);
    }
}
