use clap::ValueEnum;
use plotters::prelude::*;
use rand_distr::{Beta, Distribution, Gamma, LogNormal, Normal, Pareto, Triangular, Uniform, StudentT};
use std::error::Error;

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub enum DistType {
    Normal,
    Lognormal,
    Uniform,
    Triangular,
    Beta,
    Gamma,
    Pareto,
    T,
}

#[derive(Default, Clone, Debug)]
pub struct Params {
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mode: Option<f64>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub shape: Option<f64>,
    pub scale: Option<f64>,
    pub df: Option<f64>,
}

pub fn generate(dist: DistType, n: usize, p: &Params) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut rng = rand::rng();
    let data = match dist {
        DistType::Normal => {
            let mean = p.mean.unwrap_or(0.0);
            let std = p.std.unwrap_or(1.0);
            Normal::new(mean, std)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::Lognormal => {
            let mean = p.mean.unwrap_or(0.0);
            let std = p.std.unwrap_or(1.0);
            LogNormal::new(mean, std)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::Uniform => {
            let min = p.min.unwrap_or(0.0);
            let max = p.max.unwrap_or(1.0);
            Uniform::new(min, max)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::Triangular => {
            let min = p.min.unwrap_or(0.0);
            let max = p.max.unwrap_or(1.0);
            let mode = p.mode.unwrap_or((min + max) / 2.0);
            Triangular::new(min, max, mode)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::Beta => {
            let alpha = p.alpha.unwrap_or(0.5);
            let beta = p.beta.unwrap_or(0.5);
            Beta::new(alpha, beta)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::Gamma => {
            let shape = p.shape.unwrap_or(1.0);
            let scale = p.scale.unwrap_or(1.0);
            Gamma::new(shape, scale)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::Pareto => {
            let shape = p.shape.unwrap_or(1.0);
            let scale = p.scale.unwrap_or(1.0);
            Pareto::new(scale, shape)?.sample_iter(&mut rng).take(n).collect()
        }
        DistType::T => {
            let df = p.df.unwrap_or(1.0);
            StudentT::new(df)?.sample_iter(&mut rng).take(n).collect()
        }
    };
    Ok(data)
}

pub fn save_csv(path: &str, data: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["value"])?;
    for v in data {
        wtr.write_record([v.to_string()])?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn histogram_counts(data: &[f64], bins: usize) -> (f64, f64, Vec<usize>) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let step = if bins > 0 { (max - min) / bins as f64 } else { 0.0 };
    let mut counts = vec![0usize; bins];
    if step > 0.0 {
        for &v in data {
            let mut idx = ((v - min) / step) as isize;
            if idx < 0 { idx = 0; }
            if idx as usize >= bins { idx = bins as isize - 1; }
            counts[idx as usize] += 1;
        }
    }
    (min, step, counts)
}

pub fn save_histogram(path: &str, data: &[f64]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let bins = 50usize;
    let (min, step, counts) = histogram_counts(data, bins);
    let max_count = *counts.iter().max().unwrap_or(&0);
    let mut chart = ChartBuilder::on(&root)
        .caption("Histogram", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min..(min + step * bins as f64), 0..max_count)?;
    chart.configure_mesh().draw()?;
    chart.draw_series((0..bins).map(|i| {
        let x0 = min + i as f64 * step;
        let x1 = x0 + step;
        Rectangle::new([(x0, 0), (x1, counts[i])], BLUE.filled())
    }))?;
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[test]
    fn generate_produces_expected_length() {
        let params = Params { mean: Some(0.0), std: Some(1.0), ..Default::default() };
        let data = generate(DistType::Normal, 10, &params).expect("generation failed");
        assert_eq!(data.len(), 10);
    }

    #[test]
    fn histogram_counts_sum_matches_input_size() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (_min, _step, counts) = histogram_counts(&data, 5);
        assert_eq!(counts.iter().sum::<usize>(), data.len());
    }

    #[test]
    fn save_csv_and_histogram_create_files() {
        let dir = tempdir().unwrap();
        let csv_path = dir.path().join("out.csv");
        let png_path = dir.path().join("out.png");
        let data = vec![0.0, 1.0, 2.0];
        save_csv(csv_path.to_str().unwrap(), &data).unwrap();
        save_histogram(png_path.to_str().unwrap(), &data).unwrap();
        assert!(csv_path.exists());
        assert!(png_path.exists());
        let content = fs::read_to_string(csv_path).unwrap();
        assert!(content.contains("value"));
    }
}

