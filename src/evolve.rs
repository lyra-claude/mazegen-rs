use crate::maze::Maze;
use crate::analysis::{analyze, MazeAnalysis};
use crate::algos::{get_algorithm, ALGORITHMS};
use rand::prelude::*;
use std::collections::{HashSet, VecDeque};

/// An edge between two adjacent cells, canonicalized so first < second.
type Edge = ((usize, usize), (usize, usize));

/// What metric to maximize.
#[derive(Clone, Copy)]
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

    /// Hamming distance: number of edges in symmetric difference.
    fn hamming_distance(&self, other: &Self) -> usize {
        let shared = self.edges.intersection(&other.edges).count();
        // Each has n²-1 edges, symmetric difference = |A| + |B| - 2|A∩B|
        (self.edges.len() + other.edges.len()) - 2 * shared
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

/// Estimate population diversity: average normalized Hamming distance between sampled pairs.
/// Returns a value in [0, 1]: 0 = all identical, 1 = no shared edges.
fn population_diversity(genomes: &[Genome], rng: &mut impl Rng) -> f64 {
    if genomes.len() < 2 {
        return 0.0;
    }
    let max_distance = 2 * (genomes[0].size * genomes[0].size - 1);
    if max_distance == 0 {
        return 0.0;
    }
    // Sample up to 50 random pairs.
    let samples = 50.min(genomes.len() * (genomes.len() - 1) / 2);
    let mut total = 0.0;
    for _ in 0..samples {
        let a = rng.gen_range(0, genomes.len());
        let mut b = rng.gen_range(0, genomes.len() - 1);
        if b >= a { b += 1; }
        total += genomes[a].hamming_distance(&genomes[b]) as f64 / max_distance as f64;
    }
    total / samples as f64
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
    let div0 = population_diversity(&genomes, &mut rng);
    print_gen_stats(0, &fitnesses, Some(div0));

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
            let div = population_diversity(&genomes, &mut rng);
            print_gen_stats(gen, &fitnesses, Some(div));
        }
    }

    let best_maze = genomes[0].to_maze();
    let best_analysis = analyze(&best_maze);
    (best_maze, best_analysis)
}

fn print_gen_stats(gen: usize, fitnesses: &[f64], diversity: Option<f64>) {
    let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
    match diversity {
        Some(d) => println!("  Gen {:>4}  Best: {:>6.2}  Avg: {:>6.2}  Diversity: {:.3}",
            gen, fitnesses[0], avg, d),
        None => println!("  Gen {:>4}  Best: {:>6.2}  Avg: {:>6.2}",
            gen, fitnesses[0], avg),
    }
}

pub fn target_name_pub(target: &FitnessTarget) -> &'static str {
    target_name(target)
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

/// Parse a comma-separated list of objectives.
pub fn parse_objectives(s: &str) -> Result<Vec<FitnessTarget>, String> {
    s.split(',')
        .map(|t| parse_target(t.trim()))
        .collect()
}

// ── Multi-objective (NSGA-II) Pareto evolution ──────────────────────────

/// Configuration for Pareto (multi-objective) evolution.
pub struct ParetoConfig {
    pub size: usize,
    pub pop_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub objectives: Vec<FitnessTarget>,
}

impl Default for ParetoConfig {
    fn default() -> Self {
        Self {
            size: 15,
            pop_size: 80,
            generations: 150,
            mutation_rate: 0.3,
            objectives: vec![FitnessTarget::Difficulty, FitnessTarget::Tortuosity],
        }
    }
}

/// Evaluate a genome on multiple objectives.
fn evaluate_multi(genome: &Genome, objectives: &[FitnessTarget]) -> Vec<f64> {
    let maze = genome.to_maze();
    let analysis = analyze(&maze);
    objectives.iter().map(|t| match t {
        FitnessTarget::Difficulty => analysis.difficulty_score(),
        FitnessTarget::Tortuosity => analysis.tortuosity().unwrap_or(0.0),
        FitnessTarget::SolutionLength => analysis.solution_length().unwrap_or(0) as f64,
        FitnessTarget::DeadEndRatio => analysis.dead_end_ratio(),
    }).collect()
}

/// Returns true if a dominates b (>= on all objectives, strictly > on at least one).
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai < bi {
            return false;
        }
        if ai > bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Non-dominated sorting: assign each individual to a front (0 = best).
fn non_dominated_sort(fitnesses: &[Vec<f64>]) -> Vec<usize> {
    let n = fitnesses.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut ranks = vec![0usize; n];

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            if dominates(&fitnesses[i], &fitnesses[j]) {
                dominated_set[i].push(j);
            } else if dominates(&fitnesses[j], &fitnesses[i]) {
                domination_count[i] += 1;
            }
        }
    }

    // Build fronts iteratively.
    let mut current_front: Vec<usize> = (0..n)
        .filter(|&i| domination_count[i] == 0)
        .collect();
    let mut front_rank = 0;
    for &i in &current_front {
        ranks[i] = front_rank;
    }

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    ranks[j] = front_rank + 1;
                    next_front.push(j);
                }
            }
        }
        front_rank += 1;
        current_front = next_front;
    }

    ranks
}

/// Crowding distance for individuals within the same front.
fn crowding_distance(fitnesses: &[Vec<f64>], front_indices: &[usize]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let num_obj = fitnesses[front_indices[0]].len();
    let mut distances = vec![0.0f64; n];

    for obj in 0..num_obj {
        // Sort front members by this objective.
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            fitnesses[front_indices[a]][obj]
                .partial_cmp(&fitnesses[front_indices[b]][obj])
                .unwrap()
        });

        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[n - 1]] = f64::INFINITY;

        let f_max = fitnesses[front_indices[sorted[n - 1]]][obj];
        let f_min = fitnesses[front_indices[sorted[0]]][obj];
        let range = f_max - f_min;
        if range == 0.0 {
            continue;
        }

        for i in 1..(n - 1) {
            distances[sorted[i]] +=
                (fitnesses[front_indices[sorted[i + 1]]][obj]
                    - fitnesses[front_indices[sorted[i - 1]]][obj])
                    / range;
        }
    }

    distances
}

/// NSGA-II tournament selection: prefer lower rank, then higher crowding distance.
fn nsga2_tournament(
    ranks: &[usize],
    crowd_dist: &[f64],
    rng: &mut impl Rng,
) -> usize {
    let a = rng.gen_range(0, ranks.len());
    let b = rng.gen_range(0, ranks.len());
    if ranks[a] < ranks[b] {
        a
    } else if ranks[b] < ranks[a] {
        b
    } else if crowd_dist[a] > crowd_dist[b] {
        a
    } else {
        b
    }
}

/// A solution on the Pareto front.
pub struct ParetoSolution {
    pub maze: Maze,
    pub analysis: MazeAnalysis,
    pub objectives: Vec<f64>,
}

/// Run NSGA-II multi-objective evolution. Returns the Pareto front.
pub fn evolve_pareto(config: &ParetoConfig) -> Vec<ParetoSolution> {
    let mut rng = rand::thread_rng();

    // Initialize population.
    let mut genomes: Vec<Genome> = Vec::with_capacity(config.pop_size);
    for i in 0..config.pop_size {
        let algo_name = ALGORITHMS[i % ALGORITHMS.len()];
        let mut maze = Maze::new(config.size);
        let mut algo = get_algorithm(&algo_name.to_string(), false).unwrap();
        maze.apply_algorithm(&mut algo).unwrap();
        genomes.push(Genome::from_maze(&maze));
    }

    let mut fitnesses: Vec<Vec<f64>> = genomes.iter()
        .map(|g| evaluate_multi(g, &config.objectives))
        .collect();

    let obj_names: Vec<&str> = config.objectives.iter().map(|t| target_name(t)).collect();
    println!("=== Pareto Evolution ({0}x{0}) ===", config.size);
    println!("  Population: {}  Generations: {}  Objectives: {}",
        config.pop_size, config.generations, obj_names.join(", "));

    let ranks = non_dominated_sort(&fitnesses);
    let front0_count = ranks.iter().filter(|&&r| r == 0).count();
    let div0 = population_diversity(&genomes, &mut rng);
    println!("  Gen    0  Front: {:>3}  Diversity: {:.3}", front0_count, div0);

    for gen in 1..=config.generations {
        let ranks = non_dominated_sort(&fitnesses);

        // Compute crowding distance for the full population.
        // Group by front, compute per-front, then scatter back.
        let max_rank = *ranks.iter().max().unwrap_or(&0);
        let mut crowd_dist = vec![0.0f64; genomes.len()];
        for r in 0..=max_rank {
            let front: Vec<usize> = (0..genomes.len())
                .filter(|&i| ranks[i] == r)
                .collect();
            if front.is_empty() { continue; }
            let cd = crowding_distance(&fitnesses, &front);
            for (fi, &idx) in front.iter().enumerate() {
                crowd_dist[idx] = cd[fi];
            }
        }

        // Generate offspring.
        let mut offspring_genomes: Vec<Genome> = Vec::with_capacity(config.pop_size);
        let mut offspring_fitnesses: Vec<Vec<f64>> = Vec::with_capacity(config.pop_size);

        while offspring_genomes.len() < config.pop_size {
            let a = nsga2_tournament(&ranks, &crowd_dist, &mut rng);
            let b = nsga2_tournament(&ranks, &crowd_dist, &mut rng);
            let mut child = genomes[a].crossover(&genomes[b], &mut rng);
            if rng.gen::<f64>() < config.mutation_rate {
                child.mutate(&mut rng);
            }
            let f = evaluate_multi(&child, &config.objectives);
            offspring_fitnesses.push(f);
            offspring_genomes.push(child);
        }

        // Combine parents + offspring (2N pool).
        let combined_genomes: Vec<Genome> = genomes.into_iter()
            .chain(offspring_genomes.into_iter())
            .collect();
        let combined_fitnesses: Vec<Vec<f64>> = fitnesses.into_iter()
            .chain(offspring_fitnesses.into_iter())
            .collect();

        // Non-dominated sort on combined population.
        let combined_ranks = non_dominated_sort(&combined_fitnesses);
        let combined_max_rank = *combined_ranks.iter().max().unwrap_or(&0);

        // Select next generation: fill front by front.
        let mut next_indices: Vec<usize> = Vec::with_capacity(config.pop_size);
        for r in 0..=combined_max_rank {
            let front: Vec<usize> = (0..combined_genomes.len())
                .filter(|&i| combined_ranks[i] == r)
                .collect();
            if next_indices.len() + front.len() <= config.pop_size {
                // Whole front fits.
                next_indices.extend(&front);
            } else {
                // Partial front: use crowding distance to pick the best.
                let remaining = config.pop_size - next_indices.len();
                let cd = crowding_distance(&combined_fitnesses, &front);
                let mut sorted: Vec<usize> = (0..front.len()).collect();
                sorted.sort_by(|&a, &b| cd[b].partial_cmp(&cd[a]).unwrap());
                for i in 0..remaining {
                    next_indices.push(front[sorted[i]]);
                }
                break;
            }
        }

        genomes = next_indices.iter().map(|&i| combined_genomes[i].clone()).collect();
        fitnesses = next_indices.iter().map(|&i| combined_fitnesses[i].clone()).collect();

        if gen % 25 == 0 || gen == config.generations {
            let gen_ranks = non_dominated_sort(&fitnesses);
            let front_size = gen_ranks.iter().filter(|&&r| r == 0).count();
            let div = population_diversity(&genomes, &mut rng);
            println!("  Gen {:>4}  Front: {:>3}  Diversity: {:.3}", gen, front_size, div);
        }
    }

    // Extract Pareto front.
    let final_ranks = non_dominated_sort(&fitnesses);
    let mut front: Vec<ParetoSolution> = Vec::new();
    for i in 0..genomes.len() {
        if final_ranks[i] == 0 {
            let maze = genomes[i].to_maze();
            let analysis = analyze(&maze);
            front.push(ParetoSolution {
                maze,
                analysis,
                objectives: fitnesses[i].clone(),
            });
        }
    }

    // Sort front by first objective for consistent output.
    front.sort_by(|a, b| b.objectives[0].partial_cmp(&a.objectives[0]).unwrap());
    front
}

/// Print the Pareto front as a table.
pub fn print_pareto_front(front: &[ParetoSolution], objectives: &[FitnessTarget]) {
    let obj_names: Vec<&str> = objectives.iter().map(|t| target_name(t)).collect();

    println!("\n=== Pareto Front ({} solutions) ===", front.len());

    // Header
    print!("  {:>4}", "#");
    for name in &obj_names {
        print!("  {:>12}", name);
    }
    println!("  {:>8}  {:>8}", "sol_len", "dead_end%");
    println!("  {}", "-".repeat(4 + obj_names.len() * 14 + 20));

    for (i, sol) in front.iter().enumerate() {
        print!("  {:>4}", i + 1);
        for val in &sol.objectives {
            print!("  {:>12.2}", val);
        }
        println!("  {:>8}  {:>7.1}%",
            sol.analysis.solution_length().unwrap_or(0),
            sol.analysis.dead_end_ratio() * 100.0);
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

    #[test]
    fn test_dominates() {
        assert!(dominates(&[3.0, 2.0], &[2.0, 1.0])); // strictly better on both
        assert!(dominates(&[3.0, 2.0], &[3.0, 1.0])); // equal on one, better on other
        assert!(!dominates(&[3.0, 1.0], &[2.0, 2.0])); // tradeoff: neither dominates
        assert!(!dominates(&[2.0, 2.0], &[3.0, 1.0])); // tradeoff
        assert!(!dominates(&[2.0, 2.0], &[2.0, 2.0])); // identical: no domination
    }

    #[test]
    fn test_non_dominated_sort() {
        // Front 0: (5,1) and (1,5) — non-dominated (tradeoff)
        // Front 1: (3,1) — dominated by (5,1)
        // Front 1: (1,3) — dominated by (1,5)
        let fitnesses = vec![
            vec![5.0, 1.0],
            vec![1.0, 5.0],
            vec![3.0, 1.0],
            vec![1.0, 3.0],
        ];
        let ranks = non_dominated_sort(&fitnesses);
        assert_eq!(ranks[0], 0); // (5,1) on front 0
        assert_eq!(ranks[1], 0); // (1,5) on front 0
        assert_eq!(ranks[2], 1); // (3,1) on front 1
        assert_eq!(ranks[3], 1); // (1,3) on front 1
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let fitnesses = vec![
            vec![1.0, 5.0],
            vec![3.0, 3.0],
            vec![5.0, 1.0],
        ];
        let front = vec![0, 1, 2];
        let cd = crowding_distance(&fitnesses, &front);
        // Boundary solutions should have infinite distance.
        assert!(cd[0].is_infinite());
        assert!(cd[2].is_infinite());
        // Middle solution should have finite distance.
        assert!(cd[1].is_finite());
        assert!(cd[1] > 0.0);
    }

    #[test]
    fn test_pareto_evolution_finds_tradeoffs() {
        let config = ParetoConfig {
            size: 8,
            pop_size: 30,
            generations: 20,
            mutation_rate: 0.3,
            objectives: vec![FitnessTarget::Difficulty, FitnessTarget::Tortuosity],
        };
        let front = evolve_pareto(&config);
        // Should find at least one Pareto-optimal solution.
        assert!(!front.is_empty(), "Pareto front is empty");
        // All solutions should be non-dominated with respect to each other.
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i == j { continue; }
                assert!(
                    !dominates(&front[i].objectives, &front[j].objectives),
                    "solution {} dominates {} — not a valid Pareto front",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_hamming_distance() {
        let a = make_genome("dfs", 10);
        let b = make_genome("kruskals", 10);
        let dist = a.hamming_distance(&b);
        // Two different spanning trees must differ in at least 2 edges (swap one in, one out).
        assert!(dist >= 2, "distance {} too small", dist);
        // Max distance is 2*(n²-1) = 198 for 10x10.
        assert!(dist <= 198, "distance {} too large", dist);
        // Self-distance should be 0.
        assert_eq!(a.hamming_distance(&a), 0);
    }

    #[test]
    fn test_population_diversity_range() {
        let mut rng = rand::thread_rng();
        let genomes: Vec<Genome> = (0..20)
            .map(|i| make_genome(ALGORITHMS[i % ALGORITHMS.len()], 8))
            .collect();
        let div = population_diversity(&genomes, &mut rng);
        assert!(div >= 0.0 && div <= 1.0, "diversity {} out of range", div);
        // A diverse population seeded from different algorithms should have non-trivial diversity.
        assert!(div > 0.05, "diversity {} unexpectedly low", div);
    }

    #[test]
    fn test_parse_objectives() {
        let objs = parse_objectives("difficulty,tortuosity").unwrap();
        assert_eq!(objs.len(), 2);
        assert!(parse_objectives("invalid").is_err());
    }
}
