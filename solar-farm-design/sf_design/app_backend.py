from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from matplotlib.backends.backend_pdf import PdfPages
from pvlib.location import Location

from sf_design.avsystem import AVSystemEWFT, AVSystemEWSAT, AVSystemNSFT, AVSystemNSSAT
from sf_design.solar_gen import fixed_tilt_monofacial_system, run_modelchain


DEFAULT_CROP_LIBRARY = {
    "soybean": {"par_requirement": 336, "revenue_per_m2": 18.0},
    "lettuce": {"par_requirement": 420, "revenue_per_m2": 22.0},
    "potato": {"par_requirement": 504, "revenue_per_m2": 20.0},
    "cabbage": {"par_requirement": 672, "revenue_per_m2": 16.0},
    "spinach": {"par_requirement": 300, "revenue_per_m2": 19.0},
    "tomato": {"par_requirement": 600, "revenue_per_m2": 28.0},
    "pepper": {"par_requirement": 580, "revenue_per_m2": 26.0},
    "strawberry": {"par_requirement": 520, "revenue_per_m2": 32.0},
    "rice": {"par_requirement": 540, "revenue_per_m2": 15.0},
    "wheat": {"par_requirement": 460, "revenue_per_m2": 12.0},
}


LAYOUTS = {
    "ew-ft": AVSystemEWFT,
    "ns-ft": AVSystemNSFT,
    "ew-sat": AVSystemEWSAT,
    "ns-sat": AVSystemNSSAT,
}


@dataclass
class SimulationOutputs:
    summary: Dict[str, float | str]
    crop_table: List[Dict[str, float | str]]
    image_paths: Dict[str, str]
    output_dir: str
    report_path: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _infer_num_int(weather_index: pd.DatetimeIndex) -> int:
    delta_seconds = (weather_index[1] - weather_index[0]).total_seconds()
    return int(round(3600 / delta_seconds))


def load_crop_library(crop_file: str | None = None) -> Dict[str, Dict[str, float]]:
    if not crop_file:
        return {key: value.copy() for key, value in DEFAULT_CROP_LIBRARY.items()}

    path = Path(crop_file)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                crop: {
                    "par_requirement": float(values["par_requirement"]),
                    "revenue_per_m2": float(values["revenue_per_m2"]),
                }
                for crop, values in data.items()
            }
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        required = {"crop", "par_requirement", "revenue_per_m2"}
        if not required.issubset(frame.columns):
            raise ValueError("Crop CSV must contain crop, par_requirement, revenue_per_m2 columns.")
        return {
            row["crop"]: {
                "par_requirement": float(row["par_requirement"]),
                "revenue_per_m2": float(row["revenue_per_m2"]),
            }
            for _, row in frame.iterrows()
        }

    raise ValueError("Crop library file must be JSON or CSV.")


def generate_clear_sky_weather(latitude: float, longitude: float, timezone: str,
                               year: int, freq: str = "30min",
                               irradiance_scale: float = 1.0) -> pd.DataFrame:
    location = Location(latitude=latitude, longitude=longitude, tz=timezone)
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0, tz=timezone)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=30, tz=timezone)
    times = pd.date_range(start=start, end=end, freq=freq)
    weather = location.get_clearsky(times)
    scale = max(irradiance_scale, 0.0)
    for column in ("ghi", "dni", "dhi"):
        weather[column] = weather[column] * scale
    weather["GHI"] = weather["ghi"]
    return weather


def _read_weather_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if {"Year", "Month", "Day", "Hour", "Minute"}.issubset(raw.columns):
        raw["Time"] = pd.to_datetime(raw[["Year", "Month", "Day", "Hour", "Minute"]])
        raw = raw.drop(columns=["Year", "Month", "Day", "Hour", "Minute"])
    elif "Time" in raw.columns:
        raw["Time"] = pd.to_datetime(raw["Time"])
    else:
        raise ValueError("Weather CSV must contain Time or Year/Month/Day/Hour/Minute columns.")
    raw = raw.loc[:, ~raw.columns.str.contains("^Unnamed")]
    weather = raw.set_index("Time").sort_index()
    rename_map = {}
    for column in weather.columns:
        lower = column.lower()
        if lower == "ghi":
            rename_map[column] = "ghi"
        elif lower == "dni":
            rename_map[column] = "dni"
        elif lower == "dhi":
            rename_map[column] = "dhi"
        elif lower == "ghi":
            rename_map[column] = "ghi"
    weather = weather.rename(columns=rename_map)
    return weather


def load_weather_data(params: dict) -> pd.DataFrame:
    weather_file = params.get("weather_file")
    if weather_file:
        weather = _read_weather_csv(Path(weather_file))
        scale = max(float(params["irradiance_scale"]), 0.0)
        location = Location(latitude=params["latitude"], longitude=params["longitude"], tz=params["timezone"])
        if "ghi" not in weather.columns and "GHI" in weather.columns:
            weather["ghi"] = weather["GHI"]
        if "ghi" not in weather.columns:
            raise ValueError("Imported weather data must include GHI.")

        solar_position = location.get_solarposition(weather.index)
        if "dni" not in weather.columns or "dhi" not in weather.columns:
            erbs = pvlib.irradiance.erbs(
                ghi=weather["ghi"],
                zenith=solar_position["apparent_zenith"],
                datetime_or_doy=weather.index,
            )
            weather["dni"] = weather.get("dni", erbs["dni"])
            weather["dhi"] = weather.get("dhi", erbs["dhi"])

        weather["ghi"] = weather["ghi"] * scale
        weather["dni"] = weather["dni"] * scale
        weather["dhi"] = weather["dhi"] * scale
        weather["GHI"] = weather["ghi"]
        return weather[["ghi", "dni", "dhi", "GHI"]]

    return generate_clear_sky_weather(
        latitude=params["latitude"],
        longitude=params["longitude"],
        timezone=params["timezone"],
        year=params["year"],
        freq=params["weather_frequency"],
        irradiance_scale=params["irradiance_scale"],
    )


def _build_av_system(layout: str, params: dict, num_int: int):
    system = LAYOUTS[layout]()
    system.set_timeinterval(num_int)
    system.define_parameters(
        W_r=params["row_spacing"],
        A=params["panel_length"],
        H=params["height"],
        n_row=params["row_count"],
        W_p=params["panel_width"],
        L_c=params["clearance"],
        day_i=params["day_start"],
        day_f=params["day_end"],
    )

    if layout in ("ew-ft", "ns-ft"):
        system.set_surface_angle(tilt_angle=params["tilt"])
    elif layout == "ew-sat":
        system.set_surface_angle(
            tilt_morning=params["tilt_morning"],
            tilt_noon=params["tilt_noon"],
            tilt_evening=params["tilt_evening"],
        )
    else:
        system.set_surface_angle(
            tilt_morning=params["tilt_morning"],
            tilt_noon=params["tilt_noon"],
        )
    return system


def _crop_suitability(system, crop_library: Dict[str, Dict[str, float]]) -> List[Dict[str, float | str]]:
    calc_area_m2 = max((system.A_f - system.A_i) * (system.W_f - system.W_i), 0.0)
    rows: List[Dict[str, float | str]] = []
    for crop, crop_meta in crop_library.items():
        par_require = crop_meta["par_requirement"]
        count_area = np.sum(system.radiation_par[system.n_Ai:system.n_Af, system.n_Wi:system.n_Wf] > par_require)
        count_total = (system.n_Af - system.n_Ai) * (system.n_Wf - system.n_Wi)
        area_pct = float(count_area / count_total) if count_total else 0.0
        suitable_area = area_pct * calc_area_m2
        crop_revenue = suitable_area * crop_meta["revenue_per_m2"]
        if area_pct >= 0.7:
            fit = "High"
        elif area_pct >= 0.4:
            fit = "Medium"
        else:
            fit = "Low"
        rows.append({
            "crop": crop,
            "par_requirement": par_require,
            "revenue_per_m2": crop_meta["revenue_per_m2"],
            "suitable_area_pct": area_pct,
            "suitable_area_m2": suitable_area,
            "fit": fit,
            "estimated_crop_revenue": crop_revenue,
        })
    rows.sort(key=lambda row: row["estimated_crop_revenue"], reverse=True)
    return rows


def _economic_analysis(annual_energy_kwh: float, dc_capacity_kw: float,
                       best_crop_revenue: float, params: dict) -> Dict[str, float]:
    electricity_revenue = annual_energy_kwh * params["electricity_price"]
    annual_revenue = electricity_revenue + best_crop_revenue
    capex = dc_capacity_kw * params["capex_per_kw"]
    annual_opex = dc_capacity_kw * params["opex_per_kw_year"]
    lifetime_years = int(params["project_lifetime_years"])
    discount_rate = float(params["discount_rate"])
    degradation = float(params["degradation_rate"])

    discounted_cashflows = []
    cumulative_undiscounted = -capex
    payback_year = None
    for year in range(1, lifetime_years + 1):
        energy_factor = (1 - degradation) ** (year - 1)
        year_revenue = annual_revenue * energy_factor
        year_cashflow = year_revenue - annual_opex
        discounted_cashflow = year_cashflow / ((1 + discount_rate) ** year)
        discounted_cashflows.append(discounted_cashflow)
        cumulative_undiscounted += year_cashflow
        if payback_year is None and cumulative_undiscounted >= 0:
            payback_year = year

    npv = -capex + sum(discounted_cashflows)
    total_lifetime_energy = sum(
        annual_energy_kwh * ((1 - degradation) ** (year - 1))
        for year in range(1, lifetime_years + 1)
    )
    lcoe = ((capex + annual_opex * lifetime_years) / total_lifetime_energy) if total_lifetime_energy else np.nan

    return {
        "annual_electricity_revenue": electricity_revenue,
        "annual_total_revenue": annual_revenue,
        "capital_cost": capex,
        "annual_opex": annual_opex,
        "npv": npv,
        "simple_payback_years": float(payback_year) if payback_year is not None else np.nan,
        "lcoe": lcoe,
    }


def _create_pdf_report(report_path: Path, summary: dict, crop_rows: List[dict], image_paths: Dict[str, str]) -> None:
    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = ["Solar Farm Analysis Report", ""]
        for key, value in summary.items():
            lines.append(f"{key}: {value}")
        ax.text(0.03, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        crop_lines = ["Crop Suitability", "", "crop | fit | suitable_area_pct | suitable_area_m2 | est_crop_revenue"]
        for row in crop_rows:
            crop_lines.append(
                f"{row['crop']} | {row['fit']} | {row['suitable_area_pct']:.1%} | "
                f"{row['suitable_area_m2']:.1f} | {row['estimated_crop_revenue']:.2f}"
            )
        ax.text(0.03, 0.98, "\n".join(crop_lines), va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_name in ("combined", "pv_energy", "shading", "radiation_percentage", "radiation_par", "area_agri"):
            image_path = image_paths.get(image_name)
            if not image_path or not Path(image_path).exists():
                continue
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(plt.imread(image_path))
            ax.axis("off")
            ax.set_title(image_name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def run_av_analysis(params: dict) -> SimulationOutputs:
    output_dir = _ensure_dir(Path(params["output_dir"]))
    figure_dir = _ensure_dir(output_dir / "figures")
    result_dir = _ensure_dir(output_dir / "results")

    crop_library = load_crop_library(params.get("crop_file"))
    weather = load_weather_data(params)
    location = Location(latitude=params["latitude"], longitude=params["longitude"], tz=params["timezone"])
    num_int = _infer_num_int(weather.index)
    system = _build_av_system(params["layout"], params, num_int)

    system.set_meshgrid(dA=params["mesh_size"])
    system.construct_site(site_location=location, L_st=params["standard_longitude"])
    system.calc_shading_percentage(n_proc=params["processes"])
    system.calc_irradiance_components(weather["GHI"])
    system.calc_irradiance_percentage(PAR_require=params["default_par"])
    system.save_results(str(result_dir))

    shading_path = figure_dir / "shading_percentage.png"
    radiation_pct_path = figure_dir / "radiation_percentage.png"
    radiation_par_path = figure_dir / "radiation_par.png"
    area_path = figure_dir / "area_agri.png"
    combined_path = figure_dir / "combined.png"
    system.plot_shading_percentage(fig_width=8, fig_height=6).savefig(shading_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    system.plot_irradiance_percentage(fig_width=8, fig_height=6).savefig(radiation_pct_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    system.plot_irradiance_par(fig_width=8, fig_height=6).savefig(radiation_par_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    system.plot_area_agri(fig_width=8, fig_height=6).savefig(area_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    system.plot_combined(fig_width=10, fig_height=8).savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    crop_rows = _crop_suitability(system, crop_library)

    panel_area = params["panel_length"] * params["panel_width"]
    array_capacity_watts = panel_area * params["panel_power_density_kw_m2"] * 1000.0
    total_dc_capacity_kw = (array_capacity_watts * params["row_count"]) / 1000.0
    inverter_capacity_watts = (array_capacity_watts * params["row_count"]) / 1.2
    module_parameters = {
        "pdc0": array_capacity_watts,
        "gamma_pdc": params.get("gamma_pdc", -0.0047),
    }
    inverter_parameters = {"pdc0": inverter_capacity_watts}
    pv_system = fixed_tilt_monofacial_system(
        weather_df=weather[["ghi", "dni", "dhi"]],
        num_array=params["row_count"],
        surface_tilt_list=[params["tilt"]] * params["row_count"],
        surface_azimuth_list=[90 if params["layout"].startswith("ew") else 180] * params["row_count"],
        module_parameters_list=[module_parameters] * params["row_count"],
        inverter_parameters=inverter_parameters,
        temperature_model="sapm",
        array_type="open_rack_glass_polymer",
        module_type="glass_polymer",
        modules_per_string=1,
        strings_per_inverter=1,
        albedo=0.2,
        name="Desktop AV energy system",
    )
    pv_ac, pv_dc = run_modelchain(pv_system, location, weather[["ghi", "dni", "dhi"]])
    annual_energy_kwh = float(pv_ac.sum() / 1000.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    pv_ac.resample("D").sum().plot(ax=ax, lw=1.0)
    ax.set_title("Daily PV Energy")
    ax.set_ylabel("Wh")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    pv_plot_path = figure_dir / "pv_energy.png"
    fig.savefig(pv_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    best_crop = max(crop_rows, key=lambda row: row["estimated_crop_revenue"])
    economics = _economic_analysis(annual_energy_kwh, total_dc_capacity_kw,
                                   float(best_crop["estimated_crop_revenue"]), params)

    summary = {
        "layout": params["layout"],
        "weather_source": params.get("weather_file") or "generated_clear_sky",
        "annual_energy_kwh": annual_energy_kwh,
        "average_shading": float(system.average_shading),
        "average_radiation": float(system.average_radiation),
        "average_par": float(system.average_par),
        "default_crop_area_percentage": float(system.area_percentage),
        "best_crop": best_crop["crop"],
        "best_crop_fit": best_crop["fit"],
        "best_crop_revenue_estimate": float(best_crop["estimated_crop_revenue"]),
        **economics,
    }

    image_paths = {
        "combined": str(combined_path),
        "shading": str(shading_path),
        "radiation_percentage": str(radiation_pct_path),
        "radiation_par": str(radiation_par_path),
        "area_agri": str(area_path),
        "pv_energy": str(pv_plot_path),
    }
    report_path = output_dir / "report.pdf"
    _create_pdf_report(report_path, summary, crop_rows, image_paths)

    _save_json(output_dir / "summary.json", summary)
    _save_json(output_dir / "crop_table.json", {"rows": crop_rows})
    pv_ac.to_csv(output_dir / "solar_gen_ac.csv", header=["ac_power_watts"])
    pv_dc.to_csv(output_dir / "solar_gen_dc.csv")

    return SimulationOutputs(
        summary=summary,
        crop_table=crop_rows,
        image_paths=image_paths,
        output_dir=str(output_dir),
        report_path=str(report_path),
    )
