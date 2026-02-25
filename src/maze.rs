use std::boxed::Box;
use crate::algos::MazeAlgo;

#[derive(Copy, Clone)]
pub enum Walls {
    Left  = 1 << 0,
    Right = 1 << 1,
    Up    = 1 << 2,
    Down  = 1 << 3,
}

impl Walls {
    fn left(v: i8) -> bool {
        v & Walls::Left as i8 != 0
    }
    
    fn right(v: i8) -> bool {
        v & Walls::Right as i8 != 0
    }
    
    fn up(v: i8) -> bool {
        v & Walls::Up as i8 != 0
    }

    fn down(v: i8) -> bool {
        v & Walls::Down as i8 != 0
    }

    pub fn opposite(&self) -> Walls {
        match *self {
            Walls::Left => Walls::Right,
            Walls::Right => Walls::Left,
            Walls::Up => Walls::Down,
            Walls::Down => Walls::Up,
        }
    }
}


pub struct Maze {
    size: usize,
    maze: Vec<Vec<i8>>,
}

impl Maze {
    pub fn new(size: usize) -> Maze {
        let all_walls = Walls::Up as i8 | Walls::Down as i8 |
                        Walls::Left as i8 | Walls::Right as i8;
        let m = vec![vec![all_walls; size]; size];
        Maze { size: size, maze: m }
    }

    pub fn apply_algorithm(&mut self,
                           algo: &mut Box<dyn MazeAlgo>) -> Result<(), String> {
        algo.generate(self)
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn print_and_reset(&self) {
        self.print();
        // Reset cursor to first line of the maze
        println!("\x1b[{}F", self.size + 2);
    }

    pub fn print(&self) {
        // Top row
        let mut top = "┏━".to_string();
        for x in 0..(self.size - 1) {
            top.push_str(if Walls::right(self.maze[0][x]) { "┳" } else { "━" });
            top.push_str("━");
        }
        top.push('┓');
        println!("{}", top);
        // Middle rows
        for y in 0..(self.size - 1) {
            // Horizontal border
            let mut horz = if Walls::down(self.maze[y][0]) {
                    "┣━".to_string()
                } else {
                    "┃ ".to_string()
                };
            for x in 1..self.size {
                horz.push(
                    Maze::get_inner_junction(
                        self.maze[y][x - 1],
                        self.maze[y + 1][x]));
                horz.push_str(
                    if Walls::down(self.maze[y][x]) {"━"} else {" "});
            }
            horz.push(
                if Walls::down(self.maze[y][self.size - 1]) {'┫'} else {'┃'});
            println!("{}", horz);
        }
        // Final line
        let mut bot = "┗━".to_string();
        for x in 0..(self.size - 1) {
            bot.push_str(
                if Walls::right(self.maze[self.size - 1][x]) {
                    "┻"
                } else {
                    "━"
                });
            bot.push_str("━");
        }
        bot.push('┛');
        println!("{}", bot);
    }

    fn get_inner_junction(a: i8, d: i8) -> char {
        // The inner junction is the piece connecting four squares of a maze.
        // a b
        //  ?
        // c d
        //
        // The ? can be one of ┳, ┃, ╋, ... depending on the walls between
        // each of the cells. Because every cell stores all of its walls we
        // can determine which junction to use based on the down and right
        // of 'a' and the up and left of 'd' in the diagram
        //
        // Because the cells are bit vectors and there's no overlap of the
        // bits needed from 'a' or 'd' we can combine them into a single
        // i8 and use that as a lookup. Note this will break if the values
        // in the enum change. (Is this bad practice? Feels like it).
        let lookup: i8 =
            (a & (Walls::Down as i8 | Walls::Right as i8)) |
            (d & (Walls::Up as i8 | Walls::Left as i8));
        [' ', '╻', '╹', '┃', '╺', '┏', '┗', '┣',
         '╸', '┓', '┛', '┫', '━', '┳', '┻', '╋'][lookup as usize]
    }


    pub fn get_adjacent(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        vec![
            if x > 0 { Some((x - 1, y)) } else { None },
            if y > 0 { Some((x, y - 1)) } else { None },
            if x + 1 < self.size { Some((x + 1, y)) } else { None },
            if y + 1 < self.size { Some((x, y + 1)) } else { None }
        ].into_iter().flatten().collect()
    }

    pub fn get_open_adj(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let c = self.maze[y][x];
        vec![
            if x > 0 && !Walls::left(c) { Some((x - 1, y)) } else { None },
            if y > 0 && !Walls::up(c) { Some((x, y - 1)) } else { None },
            if x + 1 < self.size && !Walls::right(c) {
                Some((x + 1, y))
            } else {
                None
            },
            if y + 1 < self.size && !Walls::down(c) {
                Some((x, y + 1))
            } else {
                None
            }
        ].into_iter().flatten().collect()
    }

    /// Build a character grid representation of the maze.
    /// Grid is (2*size+1) x (2*size+1). Cell (x,y) is at grid[2y+1][2x+1].
    /// Walls are at even indices, cells at odd indices.
    pub fn to_grid(&self) -> Vec<Vec<char>> {
        let dim = 2 * self.size + 1;
        let mut grid = vec![vec![' '; dim]; dim];

        // Fill corners and walls
        for gy in 0..dim {
            for gx in 0..dim {
                let even_y = gy % 2 == 0;
                let even_x = gx % 2 == 0;

                if even_y && even_x {
                    // Junction
                    grid[gy][gx] = self.junction_char(gx / 2, gy / 2);
                } else if even_y && !even_x {
                    // Horizontal wall between (gx/2, gy/2-1) and (gx/2, gy/2)
                    let cx = gx / 2;
                    if gy == 0 || gy == dim - 1 {
                        grid[gy][gx] = '━';
                    } else {
                        let cy = gy / 2 - 1;
                        if Walls::down(self.maze[cy][cx]) {
                            grid[gy][gx] = '━';
                        }
                    }
                } else if !even_y && even_x {
                    // Vertical wall between (gx/2-1, gy/2) and (gx/2, gy/2)
                    let cy = gy / 2;
                    if gx == 0 || gx == dim - 1 {
                        grid[gy][gx] = '┃';
                    } else {
                        let cx = gx / 2 - 1;
                        if Walls::right(self.maze[cy][cx]) {
                            grid[gy][gx] = '┃';
                        }
                    }
                }
                // odd-y, odd-x = cell interior, stays as space
            }
        }
        grid
    }

    fn junction_char(&self, jx: usize, jy: usize) -> char {
        // Junction at grid position (2*jx, 2*jy) is the corner
        // where cells (jx-1,jy-1), (jx,jy-1), (jx-1,jy), (jx,jy) meet.
        let has_up = jy > 0 && (jx == 0 || jx == self.size
            || (jx > 0 && Walls::right(self.maze[jy - 1][jx - 1]))
            || (jx < self.size && Walls::left(self.maze[jy - 1][jx])));
        let has_down = jy < self.size && (jx == 0 || jx == self.size
            || (jx > 0 && Walls::right(self.maze[jy][jx - 1]))
            || (jx < self.size && Walls::left(self.maze[jy][jx])));
        let has_left = jx > 0 && (jy == 0 || jy == self.size
            || (jy > 0 && Walls::down(self.maze[jy - 1][jx - 1]))
            || (jy < self.size && Walls::up(self.maze[jy][jx - 1])));
        let has_right = jx < self.size && (jy == 0 || jy == self.size
            || (jy > 0 && Walls::down(self.maze[jy - 1][jx]))
            || (jy < self.size && Walls::up(self.maze[jy][jx])));

        let idx = (has_left as usize) << 3
                | (has_right as usize) << 2
                | (has_up as usize) << 1
                | (has_down as usize);

        [' ', '╻', '╹', '┃', '╺', '┏', '┗', '┣',
         '╸', '┓', '┛', '┫', '━', '┳', '┻', '╋'][idx]
    }

    /// Print the maze in grid format with optional solution path.
    pub fn print_with_path(&self, path: &[(usize, usize)]) {
        use std::collections::HashSet;

        let mut grid = self.to_grid();

        // Mark path cells and passages between them
        let path_set: HashSet<(usize, usize)> = path.iter().cloned().collect();

        for &(x, y) in path {
            grid[2 * y + 1][2 * x + 1] = '·';
        }
        // Mark passages between consecutive path cells
        for i in 1..path.len() {
            let (x1, y1) = path[i - 1];
            let (x2, y2) = path[i];
            let gy = y1 + y2 + 1;
            let gx = x1 + x2 + 1;
            grid[gy][gx] = '·';
        }

        // Print with ANSI colors
        let _ = path_set; // used above via grid marking
        for row in &grid {
            let mut line = String::new();
            for &ch in row {
                if ch == '·' {
                    line.push_str("\x1b[31m·\x1b[0m"); // red path
                } else if ch == ' ' {
                    line.push(' ');
                } else {
                    line.push_str("\x1b[90m"); // dim gray walls
                    line.push(ch);
                    line.push_str("\x1b[0m");
                }
            }
            println!("{}", line);
        }
    }

    pub fn remove_wall(&mut self, x: usize, y: usize, x2: usize, y2: usize) {
        let (min_x, min_y) = std::cmp::min((x, y), (x2, y2));
        let (max_x, max_y) = std::cmp::max((x, y), (x2, y2));
        let wall = if min_x < max_x { Walls::Right } else { Walls::Down };
        self.maze[min_y][min_x] &= !(wall as i8);
        self.maze[max_y][max_x] &= !(wall.opposite() as i8);
    }
}
