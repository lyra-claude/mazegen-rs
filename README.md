# MazeGen

Generate, solve, and analyze perfect mazes in Rust.

Five generation algorithms, structural analysis, a BFS solver, and SVG export with distance heatmaps.

## Algorithms

- **DFS** (Depth First Search) — long winding corridors, high tortuosity, hardest to solve
- **Kruskal's** — short branchy passages, many dead ends
- **Wilson's** — uniform spanning tree (unbiased)
- **Eller's** — row-by-row generation, memory efficient
- **Prim's** — radiating growth pattern, easiest to solve

## Building & Testing

```
cargo build
cargo test    # 18 tests
```

## Usage

Generate a 15x15 maze:
```
cargo run -- -s 15 -a dfs
```

Solve and display the path (ANSI-colored terminal output):
```
cargo run -- -s 20 -a wilsons --solve
```

Analyze maze structure:
```
cargo run -- -s 20 -a dfs --analyze
```
```
=== Maze Analysis (20x20) ===
  Cells:       400
  Dead ends:   42 (10.5%)
  Corridors:   318
  Junctions:   40
  Avg branch:  2.00
  Solution:    155 steps
  Tortuosity:  4.08
  Turns:       96
  Difficulty:  71.2/100
```

Compare all algorithms:
```
cargo run -- -s 25 --compare --trials 30
```
```
=== Algorithm Comparison (25x25, 30 trials each) ===
Algorithm    DeadEnd% AvgBrnch   SolLen Tortuous   Diffi.
------------------------------------------------------------
dfs             10.4%     2.00    156.7     3.27     70.9
kruskals        30.3%     2.00     75.7     1.58     60.5
wilsons         27.3%     2.00     77.5     1.62     60.1
ellers          27.8%     2.00     79.3     1.65     58.3
prims           32.2%     2.00     52.2     1.09     47.6
```

Export as SVG with solution path:
```
cargo run -- -s 30 -a dfs --solve --svg maze.svg
```

Export with distance heatmap (cells colored by BFS distance from start):
```
cargo run -- -s 30 -a dfs --svg maze.svg --heatmap
```

## Features

| Feature | Flag |
|---------|------|
| Choose algorithm | `-a dfs\|kruskals\|wilsons\|ellers\|prims` |
| Set maze size | `-s N` |
| Solve & show path | `--solve` |
| Structural analysis | `--analyze` |
| Compare algorithms | `--compare [--trials N]` |
| SVG export | `--svg path.svg` |
| Distance heatmap | `--heatmap` (with `--svg`) |
| Animate generation | `--animate` |

## Credits

Original maze generation by [CianLR](https://github.com/CianLR/mazegen-rs).
Solver, analysis, SVG export, and heatmap by [Lyra](https://github.com/lyra-claude).
