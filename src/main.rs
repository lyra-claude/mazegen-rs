#[macro_use]
extern crate clap;

mod algos;
mod maze;
mod solver;
mod analysis;
mod svg;

use maze::Maze;
use clap::{Arg, App};

fn get_args() -> clap::ArgMatches<'static> {
    let version = format!("{}.{}.{}{}",
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH"),
        option_env!("CARGO_PKG_VERSION_PRE").unwrap_or(""));
    App::new("MazeGen")
        .version(&*version)
        .author("CianLR <cian.ruane1@gmail.com>, Lyra <lyra@claude>")
        .about("Generates, solves, and analyzes perfect mazes")
        .arg(Arg::with_name("algorithm")
            .short("a")
            .long("algorithm")
            .takes_value(true)
            .possible_values(&algos::ALGORITHMS)
            .help("Sets the algorithm used to generate the maze"))
        .arg(Arg::with_name("size")
            .short("s")
            .long("size")
            .takes_value(true)
            .help("Sets the size of the maze to be generated"))
        .arg(Arg::with_name("animate")
            .long("animate")
            .help("Draw the process of creating the maze on screen"))
        .arg(Arg::with_name("solve")
            .long("solve")
            .help("Solve the maze and display the solution path"))
        .arg(Arg::with_name("analyze")
            .long("analyze")
            .help("Print structural analysis of the maze"))
        .arg(Arg::with_name("compare")
            .long("compare")
            .help("Compare all algorithms and print a summary table"))
        .arg(Arg::with_name("trials")
            .long("trials")
            .takes_value(true)
            .help("Number of trials for algorithm comparison (default: 50)"))
        .arg(Arg::with_name("svg")
            .long("svg")
            .takes_value(true)
            .help("Export maze as SVG to the given file path"))
        .arg(Arg::with_name("heatmap")
            .long("heatmap")
            .help("Color cells by distance from start in SVG output"))
        .get_matches()
}

fn main() -> Result<(), String> {
    let args = get_args();
    let algo = args.value_of("algorithm").unwrap_or("dfs").to_string();
    let size = value_t!(args, "size", usize).unwrap_or(15);
    let animate = args.is_present("animate");

    // Compare mode: run all algorithms and print table
    if args.is_present("compare") {
        let trials = value_t!(args, "trials", usize).unwrap_or(50);
        analysis::compare_algorithms(size, trials);
        return Ok(());
    }

    let mut m = Maze::new(size);
    let mut a = algos::get_algorithm(&algo, animate)?;
    m.apply_algorithm(&mut a)?;

    // Solve if needed
    let solution = if args.is_present("solve") || args.is_present("analyze") || args.is_present("svg") {
        solver::solve_bfs(&m, (0, 0), (size - 1, size - 1))
    } else {
        None
    };

    // Print maze (with path if solving)
    if args.is_present("solve") {
        if let Some(ref s) = solution {
            m.print_with_path(&s.path);
            println!("\nSolution: {} steps, {} turns, tortuosity {:.2}",
                s.length(), s.turn_count(), s.tortuosity());
        } else {
            m.print();
            println!("\nNo solution found.");
        }
    } else {
        m.print();
    }

    // Analyze
    if args.is_present("analyze") {
        println!();
        let a = analysis::analyze(&m);
        a.print_report();
    }

    // SVG export
    if let Some(path) = args.value_of("svg") {
        let mode = if args.is_present("heatmap") {
            svg::SvgMode::Heatmap
        } else {
            svg::SvgMode::Standard
        };
        let svg_content = svg::render_svg(&m, solution.as_ref(), &mode);
        std::fs::write(path, &svg_content)
            .map_err(|e| format!("Failed to write SVG: {}", e))?;
        println!("\nSVG written to {}", path);
    }

    Ok(())
}
