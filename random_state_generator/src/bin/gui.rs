use chrono::Local;
use eframe::{egui, App, NativeOptions};
use egui_plot::{Bar, BarChart, Plot};
use random_state_generator::{
    generate, histogram_counts, save_csv, save_histogram, DistType, Params,
};

#[derive(Default)]
struct HistoryEntry {
    name: String,
    timestamp: String,
}

#[derive(PartialEq)]
enum ThemeChoice {
    PetroTrace,
    Gpn,
    Custom,
}

impl Default for ThemeChoice {
    fn default() -> Self {
        ThemeChoice::PetroTrace
    }
}

struct Theme {
    bg: egui::Color32,
    hist: egui::Color32,
}

impl ThemeChoice {
    fn theme(&self, custom: egui::Color32) -> Theme {
        match self {
            ThemeChoice::PetroTrace => Theme {
                bg: egui::Color32::from_rgb(0xF5, 0xF9, 0xF5),
                hist: egui::Color32::from_rgb(0x22, 0x8B, 0x22),
            },
            ThemeChoice::Gpn => Theme {
                bg: egui::Color32::from_rgb(0xF7, 0xFA, 0xFF),
                hist: egui::Color32::from_rgb(0x00, 0x88, 0xDA),
            },
            ThemeChoice::Custom => Theme {
                bg: egui::Color32::WHITE,
                hist: custom,
            },
        }
    }
}

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
    history: Vec<HistoryEntry>,
    theme_choice: ThemeChoice,
    custom_hist_color: egui::Color32,
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
            history: Vec::new(),
            theme_choice: ThemeChoice::PetroTrace,
            custom_hist_color: egui::Color32::from_rgb(0x33, 0x88, 0xFF),
        }
    }
}

impl GeneratorApp {
    fn generate(&mut self) {
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
                let entry_name = if self.output.is_empty() {
                    self.dist.to_string()
                } else {
                    self.output.clone()
                };
                self.history.push(HistoryEntry {
                    name: entry_name,
                    timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                });
            }
            Err(e) => self.error = Some(e.to_string()),
        }
    }

    fn save_data(&mut self) {
        if let Err(e) = save_csv(&self.output, &self.data) {
            self.error = Some(e.to_string());
        }
    }

    fn save_hist(&mut self) {
        if !self.histogram.is_empty() {
            if let Err(e) = save_histogram(&self.histogram, &self.data) {
                self.error = Some(e.to_string());
            }
        }
    }
}

impl App for GeneratorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let theme = self.theme_choice.theme(self.custom_hist_color);

        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::G)) {
            self.generate();
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::D)) {
            self.save_data();
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S)) {
            self.save_hist();
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::R)) {
            self.data.clear();
            self.hist_counts.clear();
        }

        egui::SidePanel::right("history")
            .frame(egui::Frame::default().fill(theme.bg))
            .show(ctx, |ui| {
                ui.heading("История генераций");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for h in &self.history {
                        ui.label(format!("{} - {}", h.name, h.timestamp));
                    }
                });
                ui.separator();
                ui.heading("Темы");
                ui.radio_value(&mut self.theme_choice, ThemeChoice::PetroTrace, "PetroTrace");
                ui.radio_value(&mut self.theme_choice, ThemeChoice::Gpn, "GPN");
                ui.radio_value(&mut self.theme_choice, ThemeChoice::Custom, "Пользовательская");
                if self.theme_choice == ThemeChoice::Custom {
                    ui.color_edit_button_srgba(&mut self.custom_hist_color);
                }
                ui.separator();
                ui.heading("Горячие клавиши");
                ui.label(
                    "Ctrl+G - Генерировать\nCtrl+D - Сохранить данные\nCtrl+S - Сохранить график\nCtrl+R - Очистить",
                );
            });

        egui::SidePanel::left("controls")
            .frame(egui::Frame::default().fill(theme.bg))
            .show(ctx, |ui| {
                ui.heading("Random State Generator");
                egui::ComboBox::from_label("Distribution")
                    .selected_text(format!("{}", self.dist))
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
                    self.generate();
                }
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::RED, err);
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(theme.bg))
            .show(ctx, |ui| {
                if !self.hist_counts.is_empty() {
                    let bars: Vec<Bar> = self
                        .hist_counts
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let x = self.hist_min + (i as f64 + 0.5) * self.hist_step;
                            Bar::new(x, *c as f64).width(self.hist_step)
                        })
                        .collect();
                    Plot::new("histogram").show(ui, |plot_ui| {
                        plot_ui.bar_chart(BarChart::new(bars).color(theme.hist));
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
