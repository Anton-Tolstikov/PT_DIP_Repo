use clap::{Parser};
use random_state_generator::{DistType, Params, generate, save_csv, save_histogram};
use std::error::Error;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Distribution type
    #[arg(long, value_enum)]
    dist: DistType,

    /// Sample size
    #[arg(long, default_value_t = 1000)]
    n: usize,

    /// Mean (for normal, lognormal)
    #[arg(long)]
    mean: Option<f64>,
    /// Standard deviation (for normal, lognormal)
    #[arg(long)]
    std: Option<f64>,

    /// Minimum (for uniform, triangular)
    #[arg(long)]
    min: Option<f64>,
    /// Maximum (for uniform, triangular)
    #[arg(long)]
    max: Option<f64>,
    /// Mode (for triangular)
    #[arg(long)]
    mode: Option<f64>,

    /// Alpha (for beta)
    #[arg(long)]
    alpha: Option<f64>,
    /// Beta (for beta)
    #[arg(long)]
    beta: Option<f64>,

    /// Shape parameter (for gamma, pareto)
    #[arg(long)]
    shape: Option<f64>,
    /// Scale parameter (for gamma, pareto)
    #[arg(long)]
    scale: Option<f64>,

    /// Degrees of freedom (for t)
    #[arg(long)]
    df: Option<f64>,

    /// Output CSV file
    #[arg(long, default_value = "output.csv")]
    output: String,

    /// Optional histogram PNG file
    #[arg(long)]
    histogram: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let params = Params {
        mean: cli.mean,
        std: cli.std,
        min: cli.min,
        max: cli.max,
        mode: cli.mode,
        alpha: cli.alpha,
        beta: cli.beta,
        shape: cli.shape,
        scale: cli.scale,
        df: cli.df,
    };
    let data = generate(cli.dist, cli.n, &params)?;
    save_csv(&cli.output, &data)?;
    if let Some(path) = &cli.histogram {
        save_histogram(path, &data)?;
    }
    Ok(())
}

