use eframe::{egui, App, NativeOptions};
use egui_plot::{Bar, BarChart, Plot};
use random_state_generator::{generate, histogram_counts, save_csv, save_histogram, DistType, Params};

struct GeneratorApp {
    dist: DistType,
    n: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    mode: f64,
    alpha: f64,
    beta: f64,
    shape: f64,
    scale: f64,
    df: f64,
    output: String,
    histogram: String,
    data: Vec<f64>,
    hist_min: f64,
    hist_step: f64,
    hist_counts: Vec<usize>,
    error: Option<String>,
}

impl Default for GeneratorApp {
    fn default() -> Self {
        Self {
            dist: DistType::Normal,
            n: 1000,
            mean: 0.0,
            std: 1.0,
            min: 0.0,
            max: 1.0,
            mode: 0.5,
            alpha: 0.5,
            beta: 0.5,
            shape: 1.0,
            scale: 1.0,
            df: 1.0,
            output: "output.csv".to_string(),
            histogram: String::new(),
            data: Vec::new(),
            hist_min: 0.0,
            hist_step: 1.0,
            hist_counts: Vec::new(),
            error: None,
        }
    }
}

impl App for GeneratorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Random State Generator");
            egui::ComboBox::from_label("Distribution")
                .selected_text(format!("{:?}", self.dist))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.dist, DistType::Normal, "Normal");
                    ui.selectable_value(&mut self.dist, DistType::Lognormal, "Lognormal");
                    ui.selectable_value(&mut self.dist, DistType::Uniform, "Uniform");
                    ui.selectable_value(&mut self.dist, DistType::Triangular, "Triangular");
                    ui.selectable_value(&mut self.dist, DistType::Beta, "Beta");
                    ui.selectable_value(&mut self.dist, DistType::Gamma, "Gamma");
                    ui.selectable_value(&mut self.dist, DistType::Pareto, "Pareto");
                    ui.selectable_value(&mut self.dist, DistType::T, "T");
                });
            ui.add(egui::DragValue::new(&mut self.n).prefix("n "));
            ui.add(egui::DragValue::new(&mut self.mean).prefix("mean "));
            ui.add(egui::DragValue::new(&mut self.std).prefix("std "));
            ui.add(egui::DragValue::new(&mut self.min).prefix("min "));
            ui.add(egui::DragValue::new(&mut self.max).prefix("max "));
            ui.add(egui::DragValue::new(&mut self.mode).prefix("mode "));
            ui.add(egui::DragValue::new(&mut self.alpha).prefix("alpha "));
            ui.add(egui::DragValue::new(&mut self.beta).prefix("beta "));
            ui.add(egui::DragValue::new(&mut self.shape).prefix("shape "));
            ui.add(egui::DragValue::new(&mut self.scale).prefix("scale "));
            ui.add(egui::DragValue::new(&mut self.df).prefix("df "));
            ui.horizontal(|ui| {
                ui.label("CSV file:");
                ui.text_edit_singleline(&mut self.output);
            });
            ui.horizontal(|ui| {
                ui.label("Histogram PNG:");
                ui.text_edit_singleline(&mut self.histogram);
            });
            if ui.button("Generate").clicked() {
                let params = Params {
                    mean: Some(self.mean),
                    std: Some(self.std),
                    min: Some(self.min),
                    max: Some(self.max),
                    mode: Some(self.mode),
                    alpha: Some(self.alpha),
                    beta: Some(self.beta),
                    shape: Some(self.shape),
                    scale: Some(self.scale),
                    df: Some(self.df),
                };
                match generate(self.dist, self.n, &params) {
                    Ok(d) => {
                        self.data = d;
                        if let Err(e) = save_csv(&self.output, &self.data) {
                            self.error = Some(e.to_string());
                        } else {
                            self.error = None;
                        }
                        if !self.histogram.is_empty() {
                            if let Err(e) = save_histogram(&self.histogram, &self.data) {
                                self.error = Some(e.to_string());
                            }
                        }
                        let (min, step, counts) = histogram_counts(&self.data, 50);
                        self.hist_min = min;
                        self.hist_step = step;
                        self.hist_counts = counts;
                    }
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
            }
            if !self.hist_counts.is_empty() {
                let bars: Vec<Bar> = self.hist_counts.iter().enumerate().map(|(i, c)| {
                    let x = self.hist_min + (i as f64 + 0.5) * self.hist_step;
                    Bar::new(x, *c as f64).width(self.hist_step)
                }).collect();
                Plot::new("histogram").show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new(bars));
                });
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Random State Generator",
        NativeOptions::default(),
        Box::new(|_cc| Box::new(GeneratorApp::default())),
    )
}

