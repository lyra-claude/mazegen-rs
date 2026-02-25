use std::collections::VecDeque;

use crate::maze::Maze;
use crate::solver::Solution;

const CELL_SIZE: usize = 20;
const WALL_WIDTH: usize = 2;
const PADDING: usize = 10;

/// Colors
const WALL_COLOR: &str = "#1a1a2e";
const BG_COLOR: &str = "#f0f0f0";
const PATH_COLOR: &str = "#e94560";
const START_COLOR: &str = "#0f3460";
const END_COLOR: &str = "#16c79a";

/// Render modes for SVG output.
pub enum SvgMode {
    /// Standard rendering with optional solution path.
    Standard,
    /// Distance heatmap: color each cell by BFS distance from (0,0).
    Heatmap,
}

/// Render a maze as an SVG string.
pub fn render_svg(maze: &Maze, solution: Option<&Solution>, mode: &SvgMode) -> String {
    let size = maze.get_size();
    let total = PADDING * 2 + size * CELL_SIZE + WALL_WIDTH;
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" width="{total}" height="{total}">
<rect width="{total}" height="{total}" fill="{BG_COLOR}"/>
"#
    );

    // Draw heatmap cells (behind everything)
    if let SvgMode::Heatmap = mode {
        svg.push_str(&render_heatmap(maze, size));
    }

    // Draw solution path (behind walls)
    if let Some(sol) = solution {
        svg.push_str(&render_path(sol));
    }

    // Draw walls
    svg.push_str(&render_walls(maze, size));

    // Draw start and end markers
    if size > 0 {
        svg.push_str(&render_marker(0, 0, START_COLOR));
        svg.push_str(&render_marker(size - 1, size - 1, END_COLOR));
    }

    svg.push_str("</svg>\n");
    svg
}

fn cell_center(x: usize, y: usize) -> (f64, f64) {
    let cx = PADDING as f64 + WALL_WIDTH as f64 / 2.0 + x as f64 * CELL_SIZE as f64 + CELL_SIZE as f64 / 2.0;
    let cy = PADDING as f64 + WALL_WIDTH as f64 / 2.0 + y as f64 * CELL_SIZE as f64 + CELL_SIZE as f64 / 2.0;
    (cx, cy)
}

fn render_path(sol: &Solution) -> String {
    if sol.path.len() < 2 {
        return String::new();
    }

    let mut d = String::new();
    for (i, &(x, y)) in sol.path.iter().enumerate() {
        let (cx, cy) = cell_center(x, y);
        if i == 0 {
            d.push_str(&format!("M{cx:.1},{cy:.1}"));
        } else {
            d.push_str(&format!(" L{cx:.1},{cy:.1}"));
        }
    }

    format!(
        r#"<path d="{d}" fill="none" stroke="{PATH_COLOR}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.7"/>
"#
    )
}

fn render_marker(x: usize, y: usize, color: &str) -> String {
    let (cx, cy) = cell_center(x, y);
    let r = CELL_SIZE as f64 / 4.0;
    format!(r#"<circle cx="{cx:.1}" cy="{cy:.1}" r="{r:.1}" fill="{color}"/>
"#)
}

fn render_walls(maze: &Maze, size: usize) -> String {
    let mut walls = String::new();
    let offset = PADDING as f64 + WALL_WIDTH as f64 / 2.0;

    // We draw walls as individual line segments.
    // Top border
    walls.push_str(&wall_line(
        offset, offset,
        offset + (size * CELL_SIZE) as f64, offset,
    ));
    // Left border
    walls.push_str(&wall_line(
        offset, offset,
        offset, offset + (size * CELL_SIZE) as f64,
    ));
    // Bottom border
    walls.push_str(&wall_line(
        offset, offset + (size * CELL_SIZE) as f64,
        offset + (size * CELL_SIZE) as f64, offset + (size * CELL_SIZE) as f64,
    ));
    // Right border
    walls.push_str(&wall_line(
        offset + (size * CELL_SIZE) as f64, offset,
        offset + (size * CELL_SIZE) as f64, offset + (size * CELL_SIZE) as f64,
    ));

    // Internal walls: only draw right and bottom walls for each cell to avoid doubling
    for y in 0..size {
        for x in 0..size {
            let x1 = offset + x as f64 * CELL_SIZE as f64;
            let y1 = offset + y as f64 * CELL_SIZE as f64;

            // Right wall
            if x + 1 < size && !maze.get_open_adj(x, y).contains(&(x + 1, y)) {
                walls.push_str(&wall_line(
                    x1 + CELL_SIZE as f64, y1,
                    x1 + CELL_SIZE as f64, y1 + CELL_SIZE as f64,
                ));
            }

            // Bottom wall
            if y + 1 < size && !maze.get_open_adj(x, y).contains(&(x, y + 1)) {
                walls.push_str(&wall_line(
                    x1, y1 + CELL_SIZE as f64,
                    x1 + CELL_SIZE as f64, y1 + CELL_SIZE as f64,
                ));
            }
        }
    }

    format!(
        r#"<g stroke="{WALL_COLOR}" stroke-width="{WALL_WIDTH}" stroke-linecap="round">
{walls}</g>
"#
    )
}

fn wall_line(x1: f64, y1: f64, x2: f64, y2: f64) -> String {
    format!(r#"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}"/>
"#)
}

/// Compute BFS distances from (0,0) to every reachable cell.
fn bfs_distances(maze: &Maze) -> Vec<Vec<Option<usize>>> {
    let size = maze.get_size();
    let mut dist: Vec<Vec<Option<usize>>> = vec![vec![None; size]; size];
    let mut queue = VecDeque::new();
    dist[0][0] = Some(0);
    queue.push_back((0, 0));

    while let Some((x, y)) = queue.pop_front() {
        let d = dist[y][x].unwrap();
        for (nx, ny) in maze.get_open_adj(x, y) {
            if dist[ny][nx].is_none() {
                dist[ny][nx] = Some(d + 1);
                queue.push_back((nx, ny));
            }
        }
    }
    dist
}

/// Map a value 0.0-1.0 to a color on a perceptually smooth gradient.
/// Goes from deep blue (near) through cyan, green, yellow, to red (far).
fn heatmap_color(t: f64) -> String {
    let t = t.clamp(0.0, 1.0);

    // 5-stop gradient: blue → cyan → green → yellow → red
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };

    format!("#{:02x}{:02x}{:02x}",
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

fn render_heatmap(maze: &Maze, size: usize) -> String {
    let dist = bfs_distances(maze);
    let max_dist = dist.iter().flatten().filter_map(|d| *d).max().unwrap_or(1) as f64;
    let offset = PADDING as f64 + WALL_WIDTH as f64 / 2.0;
    let cs = CELL_SIZE as f64;

    let mut cells = String::new();
    for y in 0..size {
        for x in 0..size {
            if let Some(d) = dist[y][x] {
                let t = d as f64 / max_dist;
                let color = heatmap_color(t);
                let rx = offset + x as f64 * cs;
                let ry = offset + y as f64 * cs;
                cells.push_str(&format!(
                    r#"<rect x="{rx:.1}" y="{ry:.1}" width="{cs:.1}" height="{cs:.1}" fill="{color}" opacity="0.6"/>
"#
                ));
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::get_algorithm;
    use crate::solver::solve_bfs;

    fn make_maze(algo: &str, size: usize) -> Maze {
        let mut m = Maze::new(size);
        let mut a = get_algorithm(&algo.to_string(), false).unwrap();
        m.apply_algorithm(&mut a).unwrap();
        m
    }

    #[test]
    fn test_svg_contains_structure() {
        let m = make_maze("dfs", 5);
        let svg = render_svg(&m, None, &SvgMode::Standard);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<line")); // walls
        assert!(svg.contains("<circle")); // markers
    }

    #[test]
    fn test_svg_with_solution() {
        let m = make_maze("dfs", 10);
        let sol = solve_bfs(&m, (0, 0), (9, 9)).unwrap();
        let svg = render_svg(&m, Some(&sol), &SvgMode::Standard);
        assert!(svg.contains("<path")); // solution path
        assert!(svg.contains(PATH_COLOR));
    }

    #[test]
    fn test_svg_dimensions() {
        let m = make_maze("dfs", 10);
        let svg = render_svg(&m, None, &SvgMode::Standard);
        let expected = PADDING * 2 + 10 * CELL_SIZE + WALL_WIDTH;
        assert!(svg.contains(&format!("width=\"{}\"", expected)));
    }

    #[test]
    fn test_svg_heatmap() {
        let m = make_maze("dfs", 10);
        let svg = render_svg(&m, None, &SvgMode::Heatmap);
        // Heatmap should have many colored rectangles
        assert!(svg.contains("opacity=\"0.6\""));
        // All 100 cells should be colored (perfect maze = all reachable)
        let rect_count = svg.matches("opacity=\"0.6\"").count();
        assert_eq!(rect_count, 100);
    }
}
