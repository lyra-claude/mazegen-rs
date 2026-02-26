#[macro_use]
extern crate clap;

mod algos;
mod maze;
mod solver;
mod analysis;
mod svg;
mod evolve;

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
        .arg(Arg::with_name("diameter")
            .long("diameter")
            .help("Find and display the longest path (tree diameter)"))
        .arg(Arg::with_name("evolve")
            .long("evolve")
            .help("Evolve a maze using a genetic algorithm to maximize a fitness target"))
        .arg(Arg::with_name("pop_size")
            .long("pop-size")
            .takes_value(true)
            .help("Population size for evolution (default: 80)"))
        .arg(Arg::with_name("generations")
            .long("generations")
            .takes_value(true)
            .help("Number of generations for evolution (default: 150)"))
        .arg(Arg::with_name("target")
            .long("target")
            .takes_value(true)
            .possible_values(&["difficulty", "tortuosity", "solution_length", "dead_end_ratio"])
            .help("Fitness target for evolution (default: difficulty)"))
        .arg(Arg::with_name("pareto")
            .long("pareto")
            .help("Multi-objective Pareto evolution (NSGA-II)"))
        .arg(Arg::with_name("objectives")
            .long("objectives")
            .takes_value(true)
            .help("Comma-separated objectives for Pareto mode (default: difficulty,tortuosity)"))
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

    // Evolve mode: use GA to breed optimized mazes
    if args.is_present("evolve") {
        let target = args.value_of("target").unwrap_or("difficulty");
        let config = evolve::EvolutionConfig {
            size,
            pop_size: value_t!(args, "pop_size", usize).unwrap_or(80),
            generations: value_t!(args, "generations", usize).unwrap_or(150),
            target: evolve::parse_target(target)?,
            ..evolve::EvolutionConfig::default()
        };
        let (maze, analysis) = evolve::evolve(&config);
        println!();
        maze.print();
        println!();
        analysis.print_report();

        // SVG export if requested
        if let Some(path) = args.value_of("svg") {
            let solution = solver::solve_bfs(&maze, (0, 0), (size - 1, size - 1));
            let mode = if args.is_present("heatmap") {
                svg::SvgMode::Heatmap
            } else {
                svg::SvgMode::Standard
            };
            let svg_content = svg::render_svg(&maze, solution.as_ref(), &mode);
            std::fs::write(path, &svg_content)
                .map_err(|e| format!("Failed to write SVG: {}", e))?;
            println!("\nSVG written to {}", path);
        }
        return Ok(());
    }

    // Pareto mode: multi-objective NSGA-II evolution
    if args.is_present("pareto") {
        let obj_str = args.value_of("objectives").unwrap_or("difficulty,tortuosity");
        let objectives = evolve::parse_objectives(obj_str)?;
        let config = evolve::ParetoConfig {
            size,
            pop_size: value_t!(args, "pop_size", usize).unwrap_or(80),
            generations: value_t!(args, "generations", usize).unwrap_or(150),
            objectives: objectives.clone(),
            ..evolve::ParetoConfig::default()
        };
        let front = evolve::evolve_pareto(&config);
        evolve::print_pareto_front(&front, &objectives);

        // Print the best maze (first on the front, highest on first objective).
        if let Some(best) = front.first() {
            println!("\nBest maze (by {}):", evolve::target_name_pub(&objectives[0]));
            best.maze.print();
            println!();
            best.analysis.print_report();
        }
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

    // Diameter
    if args.is_present("diameter") {
        let diam = solver::find_diameter(&m);
        println!("\nDiameter: {} steps (longest shortest path)", diam.length());
        println!("  From ({},{}) to ({},{})",
            diam.path[0].0, diam.path[0].1,
            diam.path.last().unwrap().0, diam.path.last().unwrap().1);
        println!("  Turns: {}, Tortuosity: {:.2}", diam.turn_count(), diam.tortuosity());
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
