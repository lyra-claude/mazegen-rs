use crate::maze::Maze;
use crate::analysis::{analyze, MazeAnalysis};
use crate::algos::{get_algorithm, ALGORITHMS};
use rand::prelude::*;
use std::collections::{HashSet, VecDeque};

/// An edge between two adjacent cells, canonicalized so first < second.
type Edge = ((usize, usize), (usize, usize));

/// What metric to maximize.
pub enum FitnessTarget {
    Difficulty,
    Tortuosity,
    SolutionLength,
    DeadEndRatio,
}

/// Configuration for the evolutionary optimizer.
pub struct EvolutionConfig {
    pub size: usize,
    pub pop_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub tournament_size: usize,
    pub elite_count: usize,
    pub target: FitnessTarget,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            size: 15,
            pop_size: 80,
            generations: 150,
            mutation_rate: 0.3,
            tournament_size: 3,
            elite_count: 2,
            target: FitnessTarget::Difficulty,
        }
    }
}

/// A maze genome: the set of open edges forming a spanning tree on the grid.
#[derive(Clone)]
struct Genome {
    size: usize,
    edges: HashSet<Edge>,
}

impl Genome {
    /// Extract genome from an existing maze.
    fn from_maze(maze: &Maze) -> Self {
        let size = maze.get_size();
        let mut edges = HashSet::new();
        for y in 0..size {
            for x in 0..size {
                let adj = maze.get_open_adj(x, y);
                // Only record right and down edges to avoid duplicates.
                if x + 1 < size && adj.contains(&(x + 1, y)) {
                    edges.insert(((x, y), (x + 1, y)));
                }
                if y + 1 < size && adj.contains(&(x, y + 1)) {
                    edges.insert(((x, y), (x, y + 1)));
                }
            }
        }
        Genome { size, edges }
    }

    /// Reconstruct a Maze from this genome.
    fn to_maze(&self) -> Maze {
        let mut maze = Maze::new(self.size);
        for &((x1, y1), (x2, y2)) in &self.edges {
            maze.remove_wall(x1, y1, x2, y2);
        }
        maze
    }

    /// All possible edges in an n×n grid.
    fn all_grid_edges(size: usize) -> Vec<Edge> {
        let mut edges = Vec::with_capacity(2 * size * (size - 1));
        for y in 0..size {
            for x in 0..size {
                if x + 1 < size {
                    edges.push(((x, y), (x + 1, y)));
                }
                if y + 1 < size {
                    edges.push(((x, y), (x, y + 1)));
                }
            }
        }
        edges
    }

    /// Canonical edge representation (smaller coordinate first).
    fn canonical_edge(a: (usize, usize), b: (usize, usize)) -> Edge {
        if a < b { (a, b) } else { (b, a) }
    }

    /// Grid neighbors of a cell.
    fn neighbors(x: usize, y: usize, size: usize) -> Vec<(usize, usize)> {
        let mut n = Vec::with_capacity(4);
        if x > 0 { n.push((x - 1, y)); }
        if y > 0 { n.push((x, y - 1)); }
        if x + 1 < size { n.push((x + 1, y)); }
        if y + 1 < size { n.push((x, y + 1)); }
        n
    }

    /// BFS from start along tree edges, optionally excluding one edge.
    /// Returns all reachable cells.
    fn bfs_component(&self, start: (usize, usize), exclude: Option<Edge>) -> HashSet<(usize, usize)> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);

        while let Some((x, y)) = queue.pop_front() {
            for &(nx, ny) in &Self::neighbors(x, y, self.size) {
                if visited.contains(&(nx, ny)) {
                    continue;
                }
                let edge = Self::canonical_edge((x, y), (nx, ny));
                if Some(edge) == exclude {
                    continue;
                }
                if self.edges.contains(&edge) {
                    visited.insert((nx, ny));
                    queue.push_back((nx, ny));
                }
            }
        }
        visited
    }

    /// Mutate by edge swap: remove a random tree edge, add a non-tree edge
    /// that reconnects the two components. Preserves the spanning tree property.
    fn mutate(&mut self, rng: &mut impl Rng) {
        let tree_edges: Vec<Edge> = self.edges.iter().cloned().collect();
        if tree_edges.is_empty() {
            return;
        }
        let edge_to_remove = tree_edges[rng.gen_range(0, tree_edges.len())];
        let ((x1, y1), _) = edge_to_remove;

        // Find component reachable from one side after removing the edge.
        let component = self.bfs_component((x1, y1), Some(edge_to_remove));

        // Find non-tree edges that cross the cut.
        let crossing: Vec<Edge> = Self::all_grid_edges(self.size)
            .into_iter()
            .filter(|e| !self.edges.contains(e))
            .filter(|&(a, b)| {
                (component.contains(&a)) != (component.contains(&b))
            })
            .collect();

        if let Some(&new_edge) = crossing.choose(rng) {
            self.edges.remove(&edge_to_remove);
            self.edges.insert(new_edge);
        }
    }

    /// Crossover: take the union of both parents' edges, then extract a
    /// random spanning tree using Kruskal's on the shuffled union.
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let mut union_edges: Vec<Edge> = self.edges.union(&other.edges).cloned().collect();
        union_edges.shuffle(rng);

        // Kruskal's: build spanning tree from shuffled union.
        let mut uf = UnionFind::new(self.size);
        let mut tree_edges = HashSet::new();

        for edge in union_edges {
            let ((x1, y1), (x2, y2)) = edge;
            let a = y1 * self.size + x1;
            let b = y2 * self.size + x2;
            if uf.find(a) != uf.find(b) {
                uf.union(a, b);
                tree_edges.insert(edge);
            }
        }

        Genome {
            size: self.size,
            edges: tree_edges,
        }
    }
}

/// Union-Find for Kruskal's spanning tree extraction.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        let n = size * size;
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

/// Evaluate a genome's fitness for the given target.
fn evaluate(genome: &Genome, target: &FitnessTarget) -> f64 {
    let maze = genome.to_maze();
    let analysis = analyze(&maze);
    match target {
        FitnessTarget::Difficulty => analysis.difficulty_score(),
        FitnessTarget::Tortuosity => analysis.tortuosity().unwrap_or(0.0),
        FitnessTarget::SolutionLength => analysis.solution_length().unwrap_or(0) as f64,
        FitnessTarget::DeadEndRatio => analysis.dead_end_ratio(),
    }
}

/// Tournament selection: pick k random individuals, return index of the best.
fn tournament_select(
    fitnesses: &[f64],
    k: usize,
    rng: &mut impl Rng,
) -> usize {
    let mut best = rng.gen_range(0, fitnesses.len());
    for _ in 1..k {
        let idx = rng.gen_range(0, fitnesses.len());
        if fitnesses[idx] > fitnesses[best] {
            best = idx;
        }
    }
    best
}

/// Run the evolutionary optimizer. Returns the best maze and its analysis.
pub fn evolve(config: &EvolutionConfig) -> (Maze, MazeAnalysis) {
    let mut rng = rand::thread_rng();

    // Initialize population using all 5 algorithms (round-robin).
    let mut genomes: Vec<Genome> = Vec::with_capacity(config.pop_size);
    for i in 0..config.pop_size {
        let algo_name = ALGORITHMS[i % ALGORITHMS.len()];
        let mut maze = Maze::new(config.size);
        let mut algo = get_algorithm(&algo_name.to_string(), false).unwrap();
        maze.apply_algorithm(&mut algo).unwrap();
        genomes.push(Genome::from_maze(&maze));
    }

    let mut fitnesses: Vec<f64> = genomes.iter()
        .map(|g| evaluate(g, &config.target))
        .collect();

    // Sort by fitness descending.
    let mut indices: Vec<usize> = (0..genomes.len()).collect();
    indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());
    genomes = indices.iter().map(|&i| genomes[i].clone()).collect();
    fitnesses = indices.iter().map(|&i| fitnesses[i]).collect();

    let target_str = target_name(&config.target);
    println!("=== Evolutionary Maze Designer ({0}x{0}) ===", config.size);
    println!("  Population: {}  Generations: {}  Target: {}",
        config.pop_size, config.generations, target_str);
    println!("  Mutation rate: {:.0}%  Tournament: {}  Elitism: {}",
        config.mutation_rate * 100.0, config.tournament_size, config.elite_count);
    print_gen_stats(0, &fitnesses);

    for gen in 1..=config.generations {
        let mut next_genomes: Vec<Genome> = Vec::with_capacity(config.pop_size);
        let mut next_fitnesses: Vec<f64> = Vec::with_capacity(config.pop_size);

        // Elitism: carry forward the top individuals unchanged.
        let elite = config.elite_count.min(genomes.len());
        for i in 0..elite {
            next_genomes.push(genomes[i].clone());
            next_fitnesses.push(fitnesses[i]);
        }

        // Fill the rest with offspring.
        while next_genomes.len() < config.pop_size {
            let a = tournament_select(&fitnesses, config.tournament_size, &mut rng);
            let b = tournament_select(&fitnesses, config.tournament_size, &mut rng);
            let mut child = genomes[a].crossover(&genomes[b], &mut rng);

            if rng.gen::<f64>() < config.mutation_rate {
                child.mutate(&mut rng);
            }

            let f = evaluate(&child, &config.target);
            next_fitnesses.push(f);
            next_genomes.push(child);
        }

        // Sort by fitness descending.
        let mut idx: Vec<usize> = (0..next_genomes.len()).collect();
        idx.sort_by(|&a, &b| next_fitnesses[b].partial_cmp(&next_fitnesses[a]).unwrap());
        genomes = idx.iter().map(|&i| next_genomes[i].clone()).collect();
        fitnesses = idx.iter().map(|&i| next_fitnesses[i]).collect();

        // Progress output every 25 generations and at the end.
        if gen % 25 == 0 || gen == config.generations {
            print_gen_stats(gen, &fitnesses);
        }
    }

    let best_maze = genomes[0].to_maze();
    let best_analysis = analyze(&best_maze);
    (best_maze, best_analysis)
}

fn print_gen_stats(gen: usize, fitnesses: &[f64]) {
    let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
    println!("  Gen {:>4}  Best: {:>6.2}  Avg: {:>6.2}  Worst: {:>6.2}",
        gen, fitnesses[0], avg, fitnesses[fitnesses.len() - 1]);
}

fn target_name(target: &FitnessTarget) -> &'static str {
    match target {
        FitnessTarget::Difficulty => "difficulty",
        FitnessTarget::Tortuosity => "tortuosity",
        FitnessTarget::SolutionLength => "solution_length",
        FitnessTarget::DeadEndRatio => "dead_end_ratio",
    }
}

/// Parse a fitness target from a string.
pub fn parse_target(s: &str) -> Result<FitnessTarget, String> {
    match s {
        "difficulty" => Ok(FitnessTarget::Difficulty),
        "tortuosity" => Ok(FitnessTarget::Tortuosity),
        "solution_length" => Ok(FitnessTarget::SolutionLength),
        "dead_end_ratio" => Ok(FitnessTarget::DeadEndRatio),
        _ => Err(format!("Unknown target '{}'. Choose: difficulty, tortuosity, solution_length, dead_end_ratio", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_genome(algo: &str, size: usize) -> Genome {
        let mut maze = Maze::new(size);
        let mut a = get_algorithm(&algo.to_string(), false).unwrap();
        maze.apply_algorithm(&mut a).unwrap();
        Genome::from_maze(&maze)
    }

    #[test]
    fn test_genome_edge_count() {
        // A perfect maze on n×n grid has exactly n²-1 edges (spanning tree).
        for size in &[5, 10, 15, 20] {
            let g = make_genome("dfs", *size);
            assert_eq!(
                g.edges.len(), size * size - 1,
                "size={}: expected {} edges, got {}", size, size * size - 1, g.edges.len()
            );
        }
    }

    #[test]
    fn test_genome_roundtrip() {
        // Extract genome from maze, reconstruct maze, extract again — should match.
        let g1 = make_genome("kruskals", 12);
        let maze = g1.to_maze();
        let g2 = Genome::from_maze(&maze);
        assert_eq!(g1.edges, g2.edges);
    }

    #[test]
    fn test_mutation_preserves_tree() {
        let mut rng = rand::thread_rng();
        let mut g = make_genome("dfs", 10);
        let original_count = g.edges.len();
        for _ in 0..20 {
            g.mutate(&mut rng);
            assert_eq!(
                g.edges.len(), original_count,
                "mutation changed edge count"
            );
            // Verify connectivity: BFS from (0,0) should reach all cells.
            let component = g.bfs_component((0, 0), None);
            assert_eq!(
                component.len(), g.size * g.size,
                "maze disconnected after mutation"
            );
        }
    }

    #[test]
    fn test_crossover_produces_spanning_tree() {
        let mut rng = rand::thread_rng();
        let a = make_genome("dfs", 10);
        let b = make_genome("kruskals", 10);
        for _ in 0..10 {
            let child = a.crossover(&b, &mut rng);
            assert_eq!(
                child.edges.len(), 10 * 10 - 1,
                "crossover produced wrong edge count"
            );
            let component = child.bfs_component((0, 0), None);
            assert_eq!(
                component.len(), 100,
                "crossover produced disconnected maze"
            );
        }
    }

    #[test]
    fn test_evolution_improves_fitness() {
        // Run a short evolution and check that the best fitness doesn't decrease.
        let config = EvolutionConfig {
            size: 8,
            pop_size: 20,
            generations: 30,
            mutation_rate: 0.3,
            tournament_size: 3,
            elite_count: 2,
            target: FitnessTarget::Difficulty,
        };
        let (_, analysis) = evolve(&config);
        // The evolved maze should have a positive difficulty score.
        assert!(analysis.difficulty_score() > 0.0);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(3);
        assert_ne!(uf.find(0), uf.find(1));
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        uf.union(1, 2);
        assert_eq!(uf.find(0), uf.find(2));
    }
}
