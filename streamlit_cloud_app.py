from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR / "solar-farm-design"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from sf_design.app_backend import DEFAULT_CROP_LIBRARY, run_av_analysis


st.set_page_config(page_title="农光互补分析平台", page_icon="☀️", layout="wide")


def save_upload(uploaded_file, target_dir: Path) -> str | None:
    if uploaded_file is None:
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_path)


def fmt_number(value, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


st.title("农光互补分析平台")
st.caption("输入农场与光伏参数，上传天气数据或作物库，生成发电、作物适配、经济性和 PDF 报告。")

with st.sidebar:
    st.header("输入参数")

    layout = st.selectbox("布局类型", ["ew-ft", "ns-ft", "ew-sat", "ns-sat"], index=0)
    latitude = st.number_input("纬度", value=42.45, format="%.4f")
    longitude = st.number_input("经度", value=-76.50, format="%.4f")
    timezone = st.text_input("时区", value="US/Eastern")
    year = st.number_input("年份", value=2019, step=1)

    st.subheader("阵列参数")
    row_count = st.number_input("光伏排数", min_value=1, value=4, step=1)
    row_spacing = st.number_input("排间距（米）", min_value=0.1, value=4.0, step=0.5)
    panel_length = st.number_input("组件长度（米）", min_value=0.1, value=20.0, step=1.0)
    panel_width = st.number_input("组件宽度（米）", min_value=0.1, value=4.0, step=0.5)
    height = st.number_input("安装高度（米）", min_value=0.1, value=2.0, step=0.5)
    clearance = st.number_input("边界留白（米）", min_value=0.0, value=5.0, step=0.5)
    day_start = st.number_input("生长季起始日", min_value=1, max_value=365, value=60, step=1)
    day_end = st.number_input("生长季结束日", min_value=1, max_value=365, value=274, step=1)
    tilt = st.number_input("固定倾角（度）", value=20.0, step=1.0)
    tilt_morning = st.number_input("早晨倾角（度）", value=60.0, step=1.0)
    tilt_noon = st.number_input("中午倾角（度）", value=0.0, step=1.0)
    tilt_evening = st.number_input("傍晚倾角（度）", value=-60.0, step=1.0)

    st.subheader("模型参数")
    mesh_size = st.number_input("网格尺寸（米）", min_value=0.1, value=1.0, step=0.5)
    processes = st.number_input("并行进程数", min_value=1, value=1, step=1)
    default_par = st.number_input("默认 PAR 阈值", min_value=0.0, value=420.0, step=10.0)
    irradiance_scale = st.number_input("光照倍率", min_value=0.0, value=1.0, step=0.1)
    weather_frequency = st.selectbox("天气步长", ["15min", "30min", "1h"], index=1)
    standard_longitude = st.number_input("标准经线", value=-75.0, step=1.0)
    panel_power_density_kw_m2 = st.number_input("功率密度（kW/m2）", min_value=0.01, value=0.22, step=0.01, format="%.3f")

    st.subheader("经济参数")
    electricity_price = st.number_input("售电电价", min_value=0.0, value=0.65, step=0.05)
    capex_per_kw = st.number_input("单位建设成本（元/kW）", min_value=0.0, value=4200.0, step=100.0)
    opex_per_kw_year = st.number_input("单位运维成本（元/kW·年）", min_value=0.0, value=120.0, step=10.0)
    project_lifetime_years = st.number_input("项目寿命（年）", min_value=1, value=20, step=1)
    discount_rate = st.number_input("贴现率", min_value=0.0, value=0.08, step=0.01, format="%.3f")
    degradation_rate = st.number_input("衰减率", min_value=0.0, value=0.005, step=0.001, format="%.3f")

    st.subheader("文件导入")
    weather_file = st.file_uploader("上传天气 CSV", type=["csv"])
    crop_file = st.file_uploader("上传作物参数库（JSON/CSV）", type=["json", "csv"])
    output_name = st.text_input("结果子目录名", value="streamlit_run")

    run_button = st.button("开始计算", type="primary", use_container_width=True)


tab_summary, tab_crops, tab_images, tab_about = st.tabs(["结果摘要", "作物分析", "图像结果", "说明"])

with tab_about:
    st.markdown(
        """
        **说明**

        - 不上传天气文件时，系统会自动生成晴空天气数据。
        - 不上传作物库时，系统会使用内置的 10 种作物参数。
        - 计算完成后可直接下载 PDF 报告。
        """
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "作物": name,
                    "PAR阈值": meta["par_requirement"],
                    "收益/平方米": meta["revenue_per_m2"],
                }
                for name, meta in DEFAULT_CROP_LIBRARY.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


if run_button:
    output_dir = ROOT_DIR / "app_outputs" / output_name
    upload_dir = output_dir / "uploads"
    params = {
        "layout": layout,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "timezone": timezone,
        "year": int(year),
        "row_count": int(row_count),
        "row_spacing": float(row_spacing),
        "panel_length": float(panel_length),
        "panel_width": float(panel_width),
        "height": float(height),
        "clearance": float(clearance),
        "day_start": int(day_start),
        "day_end": int(day_end),
        "tilt": float(tilt),
        "tilt_morning": float(tilt_morning),
        "tilt_noon": float(tilt_noon),
        "tilt_evening": float(tilt_evening),
        "mesh_size": float(mesh_size),
        "processes": int(processes),
        "default_par": float(default_par),
        "irradiance_scale": float(irradiance_scale),
        "panel_power_density_kw_m2": float(panel_power_density_kw_m2),
        "electricity_price": float(electricity_price),
        "weather_frequency": weather_frequency,
        "standard_longitude": float(standard_longitude),
        "capex_per_kw": float(capex_per_kw),
        "opex_per_kw_year": float(opex_per_kw_year),
        "project_lifetime_years": int(project_lifetime_years),
        "discount_rate": float(discount_rate),
        "degradation_rate": float(degradation_rate),
        "weather_file": save_upload(weather_file, upload_dir),
        "crop_file": save_upload(crop_file, upload_dir),
        "output_dir": str(output_dir),
    }

    with st.spinner("正在运行仿真，请稍候..."):
        try:
            st.session_state["outputs"] = run_av_analysis(params)
        except Exception as exc:
            st.error(f"运行失败：{exc}")


outputs = st.session_state.get("outputs")

if outputs:
    with tab_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年发电量 (kWh)", fmt_number(outputs.summary.get("annual_energy_kwh", 0), 1))
        c2.metric("最佳作物", str(outputs.summary.get("best_crop", "-")))
        c3.metric("年总收入", fmt_number(outputs.summary.get("annual_total_revenue", 0), 2))
        c4.metric("回收期 (年)", fmt_number(outputs.summary.get("simple_payback_years", 0), 1))

        st.json(outputs.summary)

        report_path = Path(outputs.report_path)
        if report_path.exists():
            st.download_button(
                "下载 PDF 报告",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/pdf",
            )

        summary_path = Path(outputs.output_dir) / "summary.json"
        if summary_path.exists():
            st.download_button(
                "下载 JSON 摘要",
                data=summary_path.read_bytes(),
                file_name=summary_path.name,
                mime="application/json",
            )

        st.success(f"结果目录：{outputs.output_dir}")

    with tab_crops:
        crop_df = pd.DataFrame(outputs.crop_table)
        if not crop_df.empty:
            crop_df["suitable_area_pct"] = (crop_df["suitable_area_pct"] * 100).round(2)
            crop_df = crop_df.rename(
                columns={
                    "crop": "作物",
                    "fit": "适配度",
                    "par_requirement": "PAR阈值",
                    "revenue_per_m2": "单位收益",
                    "suitable_area_pct": "适宜面积占比(%)",
                    "suitable_area_m2": "适宜面积(m2)",
                    "estimated_crop_revenue": "作物收益估算",
                }
            )
            st.dataframe(crop_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无作物结果。")

    with tab_images:
        image_cols = st.columns(2)
        images = [
            ("综合结果图", "combined"),
            ("PV 能量图", "pv_energy"),
            ("遮阴比例图", "shading"),
            ("辐照比例图", "radiation_percentage"),
            ("PAR 图", "radiation_par"),
            ("农业适宜面积图", "area_agri"),
        ]
        for idx, (label, key) in enumerate(images):
            image_path = outputs.image_paths.get(key)
            if image_path and Path(image_path).exists():
                image_cols[idx % 2].image(image_path, caption=label, use_container_width=True)
