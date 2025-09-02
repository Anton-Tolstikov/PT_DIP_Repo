import os
import threading
import queue
import json
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ------------------------ Core processing functions ------------------------ #
def read_input_file(path: str, delimiter: Optional[str] = None) -> pd.DataFrame:
	ext = os.path.splitext(path)[1].lower()
	if ext in {".xls", ".xlsx"}:
		return pd.read_excel(path)
	if delimiter is None:
		return pd.read_csv(path, sep=r"\s+", engine="python")
	return pd.read_csv(path, sep=delimiter, engine="python")


def load_data(
	input_path: str,
	pp_params_path: str,
	gcos_params_path: Optional[str] = None,
	delimiter: Optional[str] = None,
	area_scale_divisor: float = 1000.0,
	area_col_name: str = "Area",
	hnn_col_name: str = "Hnn",
	keep_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
	df = read_input_file(input_path, delimiter=delimiter).copy()
	if area_col_name in df.columns and "Area" not in df.columns:
		df = df.rename(columns={area_col_name: "Area"})
	if hnn_col_name in df.columns and "Hnn" not in df.columns:
		df = df.rename(columns={hnn_col_name: "Hnn"})
	if "Area" in df.columns:
		df["Area"] = df["Area"] / area_scale_divisor

	# Keep only requested columns
	if keep_columns is not None:
		cols_req = {"Area", "Hnn"}
		cols_keep = set([c for c in keep_columns if c in df.columns]) | cols_req
		df = df.loc[:, [c for c in df.columns if c in cols_keep]].copy()

	pp_df = pd.read_excel(pp_params_path)
	gcos_df = pd.read_excel(gcos_params_path) if gcos_params_path else None
	return df, pp_df, gcos_df


def coerce_float(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series, errors="coerce")


def get_gcos_for_object(
	object_name: str,
	gcos_df: Optional[pd.DataFrame],
	gcos_name_col: str = "Name",
	gcos_value_col: str = "GCoS",
) -> Optional[float]:
	if gcos_df is None:
		return None
	if gcos_name_col not in gcos_df.columns or gcos_value_col not in gcos_df.columns:
		return None

	name_series = gcos_df[gcos_name_col].astype(str).str.strip().str.casefold()
	matches = gcos_df.loc[name_series == str(object_name).strip().casefold()]
	if matches.empty:
		return None

	try:
		value = float(matches.iloc[0][gcos_value_col])
	except Exception:
		return None

	return max(0.0, min(1.0, value))


def sample_pp_parameters(
	df_len: int,
	pp_df: pd.DataFrame,
	rng: np.random.Generator,
	required_cols: Tuple[str, str, str, str] = ("Kp", "Kn", "OilDens", "Kperesh"),
) -> pd.DataFrame:
	for col in required_cols:
		if col not in pp_df.columns:
			raise ValueError(f"PP parameters file is missing required column '{col}'")

	return pd.DataFrame({
		"Kp": rng.choice(coerce_float(pp_df["Kp"].dropna()).values, size=df_len, replace=True),
		"Kn": rng.choice(coerce_float(pp_df["Kn"].dropna()).values, size=df_len, replace=True),
		"OilDens": rng.choice(coerce_float(pp_df["OilDens"].dropna()).values, size=df_len, replace=True),
		"Kperesh": rng.choice(coerce_float(pp_df["Kperesh"].dropna()).values, size=df_len, replace=True),
	})


def calculate_parameters(df: pd.DataFrame, pp_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
	for col in ("Area", "Hnn"):
		if col not in df.columns:
			raise ValueError(f"Realizations file is missing required column '{col}'")

	df = df.copy()
	df["Area"] = coerce_float(df["Area"])
	df["Hnn"] = coerce_float(df["Hnn"])
	df["Area_x_Hnn"] = df["Area"] * df["Hnn"]

	sampled = sample_pp_parameters(len(df), pp_df, rng)
	df = pd.concat([df.reset_index(drop=True), sampled.reset_index(drop=True)], axis=1)

	df["Qoil"] = df["Area"] * df["Hnn"] * df["Kp"] * df["Kn"] * df["OilDens"] * df["Kperesh"]
	return df


def build_percentile_table(df: pd.DataFrame, percentiles: List[int], tracked_params: Optional[List[str]] = None) -> pd.DataFrame:
	# Always ensure these are present in the table for downstream logic
	required_for_calc = ["Area", "Hnn", "Kp", "Kn", "OilDens", "Kperesh"]
	default_tracked = ["Area", "Hnn", "Kp", "Kn", "OilDens", "Kperesh", "Qoil", "Qoil_GCoS"]
	track = default_tracked if (tracked_params is None or not tracked_params) else list(dict.fromkeys(tracked_params))
	track = list(dict.fromkeys(track + ["Area", "Hnn"]))  # always include Area/Hnn

	existing = [c for c in track if c in df.columns]
	table = pd.DataFrame({"Percentile": [f"P{100 - int(p)}" for p in percentiles]})

	for param in existing:
		series = pd.to_numeric(df[param], errors="coerce").dropna()
		if not series.empty:
			table[param] = np.percentile(series, percentiles)
		else:
			table[param] = np.nan

	# Ensure Qoil_Calc is available for selecting best rows, even if user didn't ask to track inputs
	if all(col in df.columns for col in required_for_calc):
		arr = {
			c: np.percentile(pd.to_numeric(df[c], errors="coerce").dropna(), percentiles)
			for c in required_for_calc
		}
		table["Qoil_Calc"] = arr["Area"] * arr["Hnn"] * arr["Kp"] * arr["Kn"] * arr["OilDens"] * arr["Kperesh"]
	else:
		table["Qoil_Calc"] = np.nan

	return table


def assign_target_percentiles(df: pd.DataFrame, percentiles_check: List[int]) -> pd.DataFrame:
	df = df.copy()
	qoil_percentiles = np.percentile(df["Qoil"].dropna(), percentiles_check)
	labels = [f"P{100 - int(p)}" for p in percentiles_check]

	def nearest_label(x: float) -> str:
		idx = int(np.argmin(np.abs(qoil_percentiles - x)))
		return labels[idx]

	df["Target_Qoil"] = df["Qoil"].apply(nearest_label)
	return df


def get_closest_realizations(df: pd.DataFrame, percentiles: List[int], k: int = 3) -> pd.DataFrame:
	df = df.copy()
	quant_levels = [(100 - p) / 100.0 for p in percentiles]
	targets = df["Qoil"].quantile(quant_levels).to_numpy()
	labels = [f"P{100 - p}" for p in percentiles]

	parts = []
	for label, target in zip(labels, targets):
		deviations = (df["Qoil"] - target).abs()
		closest = df.loc[deviations.nsmallest(k).index].copy()
		closest["Target_Qoil"] = label
		parts.append(closest)

	return pd.concat(parts, ignore_index=True)


def get_closest_by_area_hnn(
	df: pd.DataFrame,
	table_main: pd.DataFrame,
	qoil_percentiles: List[int],
	k: int = 3,
) -> pd.DataFrame:
	df = df.copy()
	results = []

	qoil_real_targets = [np.percentile(df["Qoil"].dropna(), p) for p in qoil_percentiles]
	tmp = table_main.copy()

	for p_label, q_target in zip(qoil_percentiles, qoil_real_targets):
		tmp["Diff"] = (tmp["Qoil_Calc"] - q_target).abs()
		closest_row = tmp.loc[tmp["Diff"].idxmin()].copy()
		area_hnn_target = float(closest_row["Area"] * closest_row["Hnn"])

		deviations = (df["Area_x_Hnn"] - area_hnn_target).abs()
		closest_df = df.loc[deviations.nsmallest(k).index].copy()
		closest_df["Target_Qoil"] = f"P{100 - int(p_label)}"
		results.append(closest_df)

	return pd.concat(results, ignore_index=True)


def get_best_percentile_rows(
	df: pd.DataFrame, table_main: pd.DataFrame, target_percentiles: List[int]
) -> pd.DataFrame:
	rows = []
	q_targets = [np.percentile(df["Qoil"].dropna(), p) for p in target_percentiles]
	has_gcos = "Qoil_GCoS" in df.columns
	q_gcos_targets = [np.percentile(df["Qoil_GCoS"].dropna(), p) for p in target_percentiles] if has_gcos else [np.nan] * len(target_percentiles)

	tmp = table_main.copy()
	for p, q_target, q_target_gcos in zip(target_percentiles, q_targets, q_gcos_targets):
		tmp["Diff"] = (tmp["Qoil_Calc"] - q_target).abs()
		best_row = tmp.loc[tmp["Diff"].idxmin()].copy()
		best_row["Percentile"] = f"P{100 - p}"
		best_row["Qoil"] = q_target
		best_row["Qoil_GCoS"] = q_target_gcos
		rows.append(best_row)

	return pd.DataFrame(rows[::-1]).reset_index(drop=True)


def apply_formatting(worksheet, df: pd.DataFrame, workbook) -> None:
	header_fmt = workbook.add_format({
		"bold": True, "font_color": "white", "bg_color": "#003366",
		"align": "center", "valign": "vcenter", "font_name": "Roboto Condensed Black",
		"font_size": 10, "border": 1
	})
	body_fmt = workbook.add_format({
		"bg_color": "#F0F0F0", "font_name": "Roboto", "font_size": 10,
		"align": "center", "valign": "vcenter", "border": 1
	})

	for col_idx, col_name in enumerate(df.columns):
		worksheet.write(0, col_idx, col_name, header_fmt)
	for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
		for col_idx, val in enumerate(row):
			worksheet.write(row_idx, col_idx, val, body_fmt)

	for idx, col in enumerate(df.columns):
		max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
		worksheet.set_column(idx, idx, max_len)


def write_excel_report(
	output_path: str,
	object_name: str,
	category: str,
	df: pd.DataFrame,
	percentiles_main: List[int],
	table_main: pd.DataFrame,
	full_percentiles_table: pd.DataFrame,
	closest_rows: pd.DataFrame,
	alt_closest_rows: pd.DataFrame,
	gcos_value: Optional[float] = None,
	bins: int = 10,
	tracked_params: List[str] = None,
) -> None:
	with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
		workbook = writer.book

		ws_final = workbook.add_worksheet("Кейс Нефть РБ")
		writer.sheets["Кейс Нефть РБ"] = ws_final

		headers = [
			"Пласт", "Категория", "Вариант", "F,тыс. м2", "Hэфф.нн, м",
			"Кп, д.ед.", "Кн, д.ед.", "Плотность нефти, г/см3",
			"Пересчетный коэффициент, д.ед", "НГЗ/НГР нефти, тыс.т", "GCoS", "НГЗ/НГР нефти в случае ГУ, тыс.т",
		]

		best_rows = get_best_percentile_rows(df, full_percentiles_table.copy(), percentiles_main)
		table_final = pd.DataFrame(columns=headers)
		for i, row in best_rows.iterrows():
			p_label = row["Percentile"]
			table_final.loc[i] = [
				object_name, category, p_label,
				row.get("Area", np.nan), row.get("Hnn", np.nan),
				row.get("Kp", np.nan), row.get("Kn", np.nan),
				row.get("OilDens", np.nan), row.get("Kperesh", np.nan),
				row.get("Qoil", np.nan), gcos_value, row.get("Qoil_GCoS", np.nan),
			]

		apply_formatting(ws_final, table_final, workbook)

		cols = ["Percentile"]
		headers = ["Процентиль"]
		if "Qoil" in table_main.columns:
			cols.append("Qoil")
			headers.append("Qoil, тыс.т")
		if "Qoil_GCoS" in table_main.columns:
			cols.append("Qoil_GCoS")
			headers.append("Qoil с учетом GCoS, тыс.т")
		reduced_table = table_main.loc[:, cols].copy()
		reduced_table.columns = headers

		start_row = len(table_final) + 3
		bordered_header = workbook.add_format({
			"bold": True, "font_color": "white", "bg_color": "#003366",
			"align": "center", "valign": "vcenter", "font_name": "Roboto Condensed Black",
			"font_size": 10, "border": 1
		})
		bordered_body = workbook.add_format({
			"border": 1, "align": "center", "valign": "vcenter",
			"font_name": "Roboto", "font_size": 10, "bg_color": "#F0F0F0"
		})

		for col_num, col_name in enumerate(reduced_table.columns):
			ws_final.write(start_row, col_num, col_name, bordered_header)
		for row_idx, (_, row) in enumerate(reduced_table.iterrows(), start=1):
			for col_idx, val in enumerate(row):
				ws_final.write(start_row + row_idx, col_idx, val, bordered_body)
		for idx, col in enumerate(reduced_table.columns):
			max_len = max(reduced_table[col].astype(str).map(len).max(), len(str(col))) + 4
			ws_final.set_column(idx, idx, max_len)

		ws_dist = workbook.add_worksheet("Распределения ПП")
		writer.sheets["Распределения ПП"] = ws_dist
		# Use user-specified parameters if provided; else default to common ones
		if tracked_params:
			parameters = [c for c in tracked_params if c in df.columns]
		else:
			parameters = [c for c in ["Hnn", "Area", "Qoil", "Kp", "Kn", "OilDens", "Kperesh"] if c in df.columns]

		header_fmt = workbook.add_format({
			"bold": True, "font_color": "white", "bg_color": "#003366",
			"align": "center", "valign": "vcenter", "font_name": "Roboto Condensed Black", "font_size": 10
		})
		body_fmt = workbook.add_format({
			"bg_color": "#F0F0F0", "font_name": "Roboto", "font_size": 10,
			"align": "center", "valign": "vcenter"
		})

		bins_count = 10
		for idx, param in enumerate(parameters):
			series_raw = pd.to_numeric(df[param], errors="coerce").dropna()
			if series_raw.empty:
				continue
			hist, edges = np.histogram(series_raw, bins=bins)
			hist_df = pd.DataFrame({"BinStart": edges[:-1], "BinEnd": edges[1:], "Count": hist})
			row_start = idx * (bins_count + 10)
			for col_num, col_val in enumerate(hist_df.columns):
				ws_dist.write(row_start, col_num, col_val, header_fmt)
			for r_idx, (_, row) in enumerate(hist_df.iterrows(), start=1):
				for c_idx, val in enumerate(row):
					ws_dist.write(row_start + r_idx, c_idx, val, body_fmt)

			chart = workbook.add_chart({"type": "column"})
			chart.add_series({
				"categories": ["Распределения ПП", row_start + 1, 0, row_start + len(hist_df), 0],
				"values": ["Распределения ПП", row_start + 1, 2, row_start + len(hist_df), 2],
				"fill": {"color": "#4F81BD"},
			})
			chart.set_title({"name": f"Гистограмма {param}"})
			chart.set_x_axis({"name": param})
			chart.set_y_axis({"name": "Count"})
			chart.set_legend({"none": True})
			ws_dist.insert_chart(row_start, 5, chart)

		df_safe = df.drop(columns=[c for c in ["Deviation", "Diff_Area_Hnn"] if c in df.columns], errors="ignore")

		ws_real = workbook.add_worksheet("Таблица реализаций")
		writer.sheets["Таблица реализаций"] = ws_real
		apply_formatting(ws_real, df_safe, workbook)

		ws_p100 = workbook.add_worksheet("Процентили 0-100")
		writer.sheets["Процентили 0-100"] = ws_p100
		apply_formatting(ws_p100, full_percentiles_table, workbook)

		ws_closest = workbook.add_worksheet("Процентильные (c ПП)")
		writer.sheets["Процентильные (c ПП)"] = ws_closest
		apply_formatting(ws_closest, closest_rows, workbook)

		ws_alt_closest = workbook.add_worksheet("Процентильные (без ПП)")
		writer.sheets["Процентильные (без ПП)"] = ws_alt_closest
		alt_clean = alt_closest_rows.drop(columns=[c for c in ["Deviation", "Diff_Area_Hnn"] if c in alt_closest_rows.columns], errors="ignore")
		apply_formatting(ws_alt_closest, alt_clean, workbook)

		qoil_series = df["Qoil"].dropna()
		hist, edges = np.histogram(qoil_series, bins=bins)

		qoil_chart = workbook.add_chart({"type": "column"})
		row_offset = start_row + len(reduced_table) + 4

		header_format = workbook.add_format({"bold": True, "border": 1, "align": "center", "valign": "vcenter"})
		cell_format = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "0.00"})

		ws_final.write(row_offset, 0, f"Гистограмма {category} (тыс.т)", header_format)
		ws_final.write(row_offset, 1, "Количество", header_format)
		for i, (bin_start, count) in enumerate(zip(edges[:-1], hist), start=1):
			ws_final.write(row_offset + i, 0, bin_start, cell_format)
			ws_final.write(row_offset + i, 1, count, cell_format)

		qoil_chart.add_series({
			"categories": ["Кейс Нефть РБ", row_offset + 1, 0, row_offset + len(hist), 0],
			"values": ["Кейс Нефть РБ", row_offset + 1, 1, row_offset + len(hist), 1],
			"fill": {"color": "#4F81BD"},
		})
		qoil_chart.set_title({"name": f"{category} нефти", "name_font": {"name": "Roboto", "size": 10, "bold": True}})
		qoil_chart.set_x_axis({
			"name": f"{category} нефти, тыс.т",
			"name_font": {"name": "Roboto", "size": 6, "bold": False},
			"num_font": {"name": "Roboto", "size": 6},
			"major_gridlines": {"visible": True},
			"label_position": "low",
		})
		qoil_chart.set_y_axis({
			"name": "Количество",
			"name_font": {"name": "Roboto", "size": 6, "bold": False},
			"num_font": {"name": "Roboto", "size": 6},
			"major_gridlines": {"visible": True},
			"label_position": "low",
		})
		qoil_chart.set_legend({"none": True})
		qoil_chart.set_size({"width": 300, "height": 400})
		ws_final.insert_chart(row_offset, 3, qoil_chart)

		if "Qoil_GCoS" in df.columns and df["Qoil_GCoS"].notna().any():
			hist_gcos, edges_gcos = np.histogram(df["Qoil_GCoS"].dropna(), bins=bins)
			ws_final.write(row_offset, 6, f"Гистограмма {category} с учетом GCoS (тыс.т)", header_format)
			ws_final.write(row_offset, 7, "Количество", header_format)
			for i, (bin_start, count) in enumerate(zip(edges_gcos[:-1], hist_gcos), start=1):
				ws_final.write(row_offset + i, 6, bin_start, cell_format)
				ws_final.write(row_offset + i, 7, count, cell_format)

			qoil_gcos_chart = workbook.add_chart({"type": "column"})
			qoil_gcos_chart.add_series({
				"categories": ["Кейс Нефть РБ", row_offset + 1, 6, row_offset + len(hist_gcos), 6],
				"values": ["Кейс Нефть РБ", row_offset + 1, 7, row_offset + len(hist_gcos), 7],
				"fill": {"color": "#C0504D"},
			})
			qoil_gcos_chart.set_title({"name": f"{category} нефти с учетом GCoS", "name_font": {"name": "Roboto", "size": 10}})
			qoil_gcos_chart.set_x_axis({
				"name": f"{category} нефти с учетом GCoS, тыс.т",
				"name_font": {"name": "Roboto", "size": 6, "bold": False},
				"num_font": {"name": "Roboto", "size": 6},
				"major_gridlines": {"visible": True},
				"label_position": "low",
			})
			qoil_gcos_chart.set_y_axis({
				"name": "Количество",
				"name_font": {"name": "Roboto", "size": 6, "bold": False},
				"num_font": {"name": "Roboto", "size": 6},
				"major_gridlines": {"visible": True},
				"label_position": "low",
			})
			qoil_gcos_chart.set_size({"width": 300, "height": 400})
			qoil_gcos_chart.set_legend({"none": True})
			ws_final.insert_chart(row_offset, 9, qoil_gcos_chart)


def to_percentile_list(csv_values: str) -> List[int]:
	return [int(v.strip()) for v in csv_values.split(",") if v.strip()]

def to_csv_list(csv_values: str) -> List[str]:
	return [v.strip() for v in csv_values.split(",") if v.strip()]


def run_pipeline(
	input_folder: str,
	output_folder: str,
	pp_params: str,
	gcos_params: Optional[str],
	gcos_name_col: str,
	gcos_value_col: str,
	file_pattern: Optional[str],
	delimiter: Optional[str],
	area_scale_div: float,
	area_col_name: str,
	hnn_col_name: str,
	keep_columns: Optional[List[str]],
	tracked_params: Optional[List[str]],
	percentiles_main: List[int],
	percentiles_step: int,
	bins: int,
	seed: int,
	k_closest: int,
	log: Callable[[str], None],
	on_progress: Callable[[int, int], None],
	stop_flag: threading.Event,
) -> None:
	os.makedirs(output_folder, exist_ok=True)
	rng = np.random.default_rng(seed)

	full_percentiles = list(range(0, 101, percentiles_step))
	percentiles_check = list(range(0, 101, 1))

	all_files = [f for f in os.listdir(input_folder) if not f.startswith(".")]
	if file_pattern:
		all_files = [f for f in all_files if file_pattern in f]

	total = len(all_files)
	done = 0

	errors: List[str] = []
	summary_rows: List[dict] = []

	log(f"Найдено файлов: {total}")
	for filename in all_files:
		if stop_flag.is_set():
			log("Остановлено пользователем.")
			break
		try:
			filepath = os.path.join(input_folder, filename)
			df_raw, pp_df, gcos_df = load_data(
				filepath, pp_params, gcos_params,
				delimiter=delimiter, area_scale_divisor=area_scale_div,
				area_col_name=area_col_name, hnn_col_name=hnn_col_name,
				keep_columns=keep_columns,
			)

			# Require only Area and Hnn
			for col in ("Area", "Hnn"):
				if col not in df_raw.columns:
					raise ValueError(f"Отсутствует обязательный столбец: {col}")

			object_name = os.path.splitext(filename)[0]
			gcos_value = get_gcos_for_object(object_name, gcos_df, gcos_name_col, gcos_value_col)

			df = calculate_parameters(df_raw, pp_df, rng)

			if gcos_value is not None:
				df["GCoS"] = gcos_value
				success_coef = rng.uniform(0.0, 1.0, size=len(df))
				df["Qoil_GCoS"] = np.where(success_coef <= gcos_value, df["Qoil"], 0.0)

			category = "Запасы" if gcos_value == 1 else "Ресурсы"

			df = assign_target_percentiles(df, percentiles_check)

			percentiles_table = build_percentile_table(df, full_percentiles, tracked_params=tracked_params)
			closest_realizations = get_closest_realizations(df, percentiles_main, k=k_closest)
			alt_closest_realizations = get_closest_by_area_hnn(df, percentiles_table.copy(), percentiles_main, k=k_closest)

			best_rows = get_best_percentile_rows(df, build_percentile_table(df, percentiles_check, tracked_params=tracked_params), percentiles_main)
			for _, row in best_rows.iterrows():
				summary_rows.append({
					"Объект": object_name,
					"Категория": category,
					"Процентиль": row["Percentile"],
					"F,тыс. м2": row.get("Area", np.nan),
					"Hэфф.нн, м": row.get("Hnn", np.nan),
					"Кп, д.ед.": row.get("Kp", np.nan),
					"Кн, д.ед.": row.get("Kn", np.nan),
					"Плотность нефти, г/см3": row.get("OilDens", np.nan),
					"Пересчетный коэффициент, д.ед": row.get("Kperesh", np.nan),
					"НГЗ/НГР нефти, тыс.т": row.get("Qoil", np.nan),
					"GCoS": gcos_value,
					"НГЗ/НГР нефти в случае ГУ, тыс.т": row.get("Qoil_GCoS", np.nan),
				})

			output_path = os.path.join(output_folder, f"{object_name}_report_oil.xlsx")
			write_excel_report(
				output_path=output_path,
				object_name=object_name,
				category=category,
				df=df,
				percentiles_main=percentiles_main,
				table_main=percentiles_table,
				full_percentiles_table=build_percentile_table(df, percentiles_check, tracked_params=tracked_params),
				closest_rows=closest_realizations,
				alt_closest_rows=alt_closest_realizations,
				gcos_value=gcos_value,
				bins=bins,
				tracked_params=tracked_params or [],
			)
			log(f"Сохранен отчет: {output_path}")

		except Exception as e:
			errors.append(f"{filename}: {e}")
			log(f"Ошибка: {filename}: {e}")

		done += 1
		on_progress(done, total)

	if summary_rows:
		summary_df = pd.DataFrame(summary_rows)
		summary_path = os.path.join(output_folder, "Сводная_таблица_оценок_Кейс_Нефть.xlsx")
		summary_df.to_excel(summary_path, index=False)
		log(f"Сводная таблица сохранена: {summary_path}")

	if errors:
		log("Ошибки при обработке:")
		for err in errors:
			log(err)
	else:
		log(f"Готово! Все файлы сохранены в {output_folder}")


# ------------------------ Tkinter GUI ------------------------ #

SETTINGS_PATH = Path.home() / ".oil_case_gui_settings.json"


class OilCaseGUI(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Oil Case Report Builder")
		self.geometry("900x720")

		self._build_vars()
		self._build_ui()
		
		self.log_queue = queue.Queue()
		self.worker_thread = None
		self.stop_event = threading.Event()

		self._load_settings()
		self.after(100, self._poll_log_queue)

	def _build_vars(self):
		self.var_input_folder = tk.StringVar()
		self.var_output_folder = tk.StringVar()
		self.var_pp_params = tk.StringVar()
		self.var_gcos_params = tk.StringVar()
		self.var_gcos_name_col = tk.StringVar(value="Name")
		self.var_gcos_value_col = tk.StringVar(value="GCoS")
		self.var_file_pattern = tk.StringVar()
		self.var_delimiter = tk.StringVar()
		self.var_area_scale = tk.DoubleVar(value=1000.0)
		self.var_percentiles_main = tk.StringVar(value="90,50,10")
		self.var_percentiles_step = tk.IntVar(value=10)
		self.var_bins = tk.IntVar(value=10)
		self.var_seed = tk.IntVar(value=42)
		self.var_k_closest = tk.IntVar(value=3)
		self.var_area_col = tk.StringVar(value="Area")
		self.var_hnn_col = tk.StringVar(value="Hnn")
		self.var_keep_cols = tk.StringVar(value="")
		self.var_tracked_params = tk.StringVar(value="Area,Hnn,Kp,Kn,OilDens,Kperesh,Qoil,Qoil_GCoS")

	def _build_ui(self):
		main = ttk.Frame(self, padding=10)
		main.pack(fill=tk.BOTH, expand=True)

		def add_row(row, label, widget, browse_cmd=None):
			ttk.Label(main, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
			widget.grid(row=row, column=1, sticky="ew", pady=4)
			if browse_cmd:
				ttk.Button(main, text="...", width=3, command=browse_cmd).grid(row=row, column=2, padx=(8, 0))
			return row + 1

		self.columnconfigure(0, weight=1)
		main.columnconfigure(1, weight=1)

		row = 0
		row = add_row(row, "Папка с файлами:", ttk.Entry(main, textvariable=self.var_input_folder),
			lambda: self._choose_dir(self.var_input_folder))
		row = add_row(row, "Папка результатов:", ttk.Entry(main, textvariable=self.var_output_folder),
			lambda: self._choose_dir(self.var_output_folder))
		row = add_row(row, "Файл подсчетных параметров (Excel):", ttk.Entry(main, textvariable=self.var_pp_params),
			lambda: self._choose_file(self.var_pp_params))
		row = add_row(row, "Файл GCoS (Excel, опц.):", ttk.Entry(main, textvariable=self.var_gcos_params),
			lambda: self._choose_file(self.var_gcos_params))

		row = add_row(row, "Колонка названия объекта GCoS:", ttk.Entry(main, textvariable=self.var_gcos_name_col))
		row = add_row(row, "Колонка значения GCoS:", ttk.Entry(main, textvariable=self.var_gcos_value_col))
		#row = add_row(row, "Фильтр по имени файла:", ttk.Entry(main, textvariable=self.var_file_pattern))
		row = add_row(row, "Разделитель столбцов в исходном файле:", ttk.Entry(main, textvariable=self.var_delimiter))
		row = add_row(row, "Колонка Area в исходных данных:", ttk.Entry(main, textvariable=self.var_area_col))
		row = add_row(row, "Колонка Hnn в исходных данных:", ttk.Entry(main, textvariable=self.var_hnn_col))
		row = add_row(row, "Список колонок для импорта (CSV, опц.):", ttk.Entry(main, textvariable=self.var_keep_cols))
		row = add_row(row, "Параметры для процентилей/гистограмм (CSV):", ttk.Entry(main, textvariable=self.var_tracked_params))
		row = add_row(row, "Масштаб Area (делитель, если требуется):", ttk.Entry(main, textvariable=self.var_area_scale))
		row = add_row(row, "Процентили (основные):", ttk.Entry(main, textvariable=self.var_percentiles_main))
		row = add_row(row, "Шаг процентилей (0..100):", ttk.Entry(main, textvariable=self.var_percentiles_step))
		row = add_row(row, "Кол-во бинов гистограмм:", ttk.Entry(main, textvariable=self.var_bins))
		row = add_row(row, "Seed:", ttk.Entry(main, textvariable=self.var_seed))
		row = add_row(row, "k ближайших реализаций:", ttk.Entry(main, textvariable=self.var_k_closest))

		btns = ttk.Frame(main)
		btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 8))
		btns.columnconfigure(0, weight=1)
		ttk.Button(btns, text="Запустить", command=self._on_start).grid(row=0, column=0, sticky="w")
		ttk.Button(btns, text="Остановить", command=self._on_stop).grid(row=0, column=1, sticky="w", padx=(8, 0))
		ttk.Button(btns, text="Сохранить настройки", command=self._save_settings).grid(row=0, column=2, sticky="e")

		self.progress = ttk.Progressbar(main, orient="horizontal", mode="determinate")
		self.progress.grid(row=row + 1, column=0, columnspan=3, sticky="ew")

		log_frame = ttk.LabelFrame(main, text="Лог")
		log_frame.grid(row=row + 2, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
		main.rowconfigure(row + 2, weight=1)

		self.txt_log = tk.Text(log_frame, height=16, wrap="word")
		self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		scroll = ttk.Scrollbar(log_frame, command=self.txt_log.yview)
		scroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.txt_log.config(yscrollcommand=scroll.set)

	def _choose_dir(self, var: tk.StringVar):
		path = filedialog.askdirectory()
		if path:
			var.set(path)

	def _choose_file(self, var: tk.StringVar):
		path = filedialog.askopenfilename(filetypes=[("Excel/CSV/Text", "*.xlsx *.xls *.csv *.txt *.dat"), ("All", "*.*")])
		if path:
			var.set(path)

	def _append_log(self, msg: str):
		self.txt_log.insert(tk.END, msg + "\n")
		self.txt_log.see(tk.END)

	def _enqueue_log(self, msg: str):
		self.log_queue.put(msg)

	def _poll_log_queue(self):
		try:
			while True:
				msg = self.log_queue.get_nowait()
				self._append_log(msg)
		except queue.Empty:
			pass
		self.after(100, self._poll_log_queue)

	def _on_start(self):
		if self.worker_thread and self.worker_thread.is_alive():
			messagebox.showinfo("Идет обработка", "Процесс уже запущен.")
			return

		try:
			input_folder = self.var_input_folder.get().strip()
			output_folder = self.var_output_folder.get().strip()
			pp_params = self.var_pp_params.get().strip()
			gcos_params = self.var_gcos_params.get().strip() or None
			gcos_name_col = self.var_gcos_name_col.get().strip() or "Name"
			gcos_value_col = self.var_gcos_value_col.get().strip() or "GCoS"
			file_pattern = self.var_file_pattern.get().strip() or None
			delimiter = self.var_delimiter.get().strip() or None
			area_col_name = self.var_area_col.get().strip() or "Area"
			hnn_col_name = self.var_hnn_col.get().strip() or "Hnn"
			keep_columns = to_csv_list(self.var_keep_cols.get()) if self.var_keep_cols.get().strip() else None
			tracked_params = to_csv_list(self.var_tracked_params.get()) if self.var_tracked_params.get().strip() else None
			area_scale_div = float(self.var_area_scale.get())
			percentiles_main = to_percentile_list(self.var_percentiles_main.get())
			percentiles_step = int(self.var_percentiles_step.get())
			bins = int(self.var_bins.get())
			seed = int(self.var_seed.get())
			k_closest = int(self.var_k_closest.get())

			if not input_folder or not os.path.isdir(input_folder):
				raise ValueError("Укажите корректную папку входных данных")
			if not output_folder:
				raise ValueError("Укажите папку для результатов")
			if not os.path.isfile(pp_params):
				raise ValueError("Укажите корректный файл параметров ПП")

			all_files = [f for f in os.listdir(input_folder) if not f.startswith(".")]
			if file_pattern:
				all_files = [f for f in all_files if file_pattern in f]
			total = max(1, len(all_files))
			self.progress.configure(maximum=total, value=0)

		except Exception as e:
			messagebox.showerror("Ошибка параметров", str(e))
			return

		self.txt_log.delete("1.0", tk.END)
		self._enqueue_log("Старт обработки...")

		self.stop_event.clear()
		self.worker_thread = threading.Thread(
			target=self._worker,
			args=(input_folder, output_folder, pp_params, gcos_params, gcos_name_col,
			      gcos_value_col, file_pattern, delimiter, area_col_name, hnn_col_name,
			      keep_columns, tracked_params, area_scale_div, percentiles_main,
			      percentiles_step, bins, seed, k_closest),
			daemon=True
		)
		self.worker_thread.start()

	def _worker(self, input_folder, output_folder, pp_params, gcos_params, gcos_name_col,
	            gcos_value_col, file_pattern, delimiter, area_col_name, hnn_col_name,
	            keep_columns, tracked_params, area_scale_div, percentiles_main, percentiles_step, bins, seed, k_closest):
		def log(msg: str):
			self._enqueue_log(msg)

		def on_progress(done: int, total: int):
			self.progress.configure(maximum=total, value=done)

		try:
			run_pipeline(
				input_folder=input_folder,
				output_folder=output_folder,
				pp_params=pp_params,
				gcos_params=gcos_params,
				gcos_name_col=gcos_name_col,
				gcos_value_col=gcos_value_col,
				file_pattern=file_pattern,
				delimiter=delimiter,
				area_scale_div=area_scale_div,
				area_col_name=area_col_name,
				hnn_col_name=hnn_col_name,
				keep_columns=keep_columns,
				tracked_params=tracked_params,
				percentiles_main=percentiles_main,
				percentiles_step=percentiles_step,
				bins=bins,
				seed=seed,
				k_closest=k_closest,
				log=log,
				on_progress=on_progress,
				stop_flag=self.stop_event,
			)
			self._enqueue_log("Завершено.")
		except Exception as e:
			self._enqueue_log(f"Критическая ошибка: {e}")

	def _on_stop(self):
		if self.worker_thread and self.worker_thread.is_alive():
			self.stop_event.set()
			self._enqueue_log("Остановка запрошена...")
		else:
			self._enqueue_log("Нет активного процесса.")

	def _save_settings(self):
		data = {
			"input_folder": self.var_input_folder.get(),
			"output_folder": self.var_output_folder.get(),
			"pp_params": self.var_pp_params.get(),
			"gcos_params": self.var_gcos_params.get(),
			"gcos_name_col": self.var_gcos_name_col.get(),
			"gcos_value_col": self.var_gcos_value_col.get(),
			"file_pattern": self.var_file_pattern.get(),
			"delimiter": self.var_delimiter.get(),
			"area_col_name": self.var_area_col.get(),
			"hnn_col_name": self.var_hnn_col.get(),
			"keep_columns": self.var_keep_cols.get(),
			"tracked_params": self.var_tracked_params.get(),
			"area_scale": self.var_area_scale.get(),
			"percentiles_main": self.var_percentiles_main.get(),
			"percentiles_step": self.var_percentiles_step.get(),
			"bins": self.var_bins.get(),
			"seed": self.var_seed.get(),
			"k_closest": self.var_k_closest.get(),
		}
		try:
			SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
			self._enqueue_log(f"Настройки сохранены: {SETTINGS_PATH}")
		except Exception as e:
			messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")

	def _load_settings(self):
		try:
			if SETTINGS_PATH.exists():
				data = json.loads(SETTINGS_PATH.read_text())
				self.var_input_folder.set(data.get("input_folder", ""))
				self.var_output_folder.set(data.get("output_folder", ""))
				self.var_pp_params.set(data.get("pp_params", ""))
				self.var_gcos_params.set(data.get("gcos_params", ""))
				self.var_gcos_name_col.set(data.get("gcos_name_col", "Name"))
				self.var_gcos_value_col.set(data.get("gcos_value_col", "GCoS"))
				self.var_file_pattern.set(data.get("file_pattern", ""))
				self.var_delimiter.set(data.get("delimiter", ""))
				self.var_area_col.set(data.get("area_col_name", "Area"))
				self.var_hnn_col.set(data.get("hnn_col_name", "Hnn"))
				self.var_keep_cols.set(data.get("keep_columns", ""))
				self.var_tracked_params.set(data.get("tracked_params", "Area,Hnn,Kp,Kn,OilDens,Kperesh,Qoil,Qoil_GCoS"))
				self.var_area_scale.set(float(data.get("area_scale", 1000.0)))
				self.var_percentiles_main.set(data.get("percentiles_main", "90,50,10"))
				self.var_percentiles_step.set(int(data.get("percentiles_step", 10)))
				self.var_bins.set(int(data.get("bins", 10)))
				self.var_seed.set(int(data.get("seed", 42)))
				self.var_k_closest.set(int(data.get("k_closest", 3)))
				self._enqueue_log("Настройки загружены.")
		except Exception as e:
			self._enqueue_log(f"Не удалось загрузить настройки: {e}")

def main():
	app = OilCaseGUI()
	app.mainloop()

if __name__ == "__main__":
	main()