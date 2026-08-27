# =========================================================
# OPTIBF — MODULAR VERSION
# =========================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from io import BytesIO

from core.brake_model import RD_MM

# =========================================================
# IMPORTS FROM CORE
# =========================================================

from core.brake_model import (
    calculate_bf,
    calculate_bf_asymmetry
)
from core.optimization import run_optimization

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="OptiBF",
    layout="wide"
)

st.title("OptiBF")

st.subheader(
    "Brake Factor Optimization Platform"
)

st.caption(
    "Alpha Version • Developed by Cassio Belo Clemente de Souza"
)

# =========================================================
# THEORY SECTION
# =========================================================

st.markdown("---")

st.header(
    "Brake Geometry and Mathematical Model"
)

col1, col2, col3 = st.columns(3)

# =========================================================
# FIGURE 1
# =========================================================

with col1:

    st.subheader(
        "Brake Geometry"
    )

    st.image(
        "brake_geometry.png",
        width=260
    )

    st.caption(
        "Source: Adapted from DAY, Andrew. "
        "Braking of Road Vehicles. Oxford, UK: Elsevier, 2014."
    )

# =========================================================
# FIGURE 2
# =========================================================

with col2:

    st.subheader(
        "Brake Self-Energizing Effect"
    )

    st.image(
        "brake_self_energizing.png",
        width=320
    )

    st.caption(
        "Source: Based on Souza et al., "
        "'Influence of Wheel Stiffness on Brake Drum "
        "Deformation in Commercial Vehicles', "
        "SAE Technical Paper, 2026."
    )

# =========================================================
# FIGURE 3
# =========================================================

with col3:

    st.subheader(
        "Mathematical Formulation"
    )

    st.image(
        "brake_equations.png",
        width=420
    )

    st.caption(
        "Source: DAY, Andrew. "
        "Braking of Road Vehicles. Oxford, UK: Elsevier, 2014."
    )

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Input Parameters"
)

# =========================================================
# FRICTION RANGE
# =========================================================

st.sidebar.subheader(
    "Friction Range"
)

mu_range = st.sidebar.slider(
    "μ Range",
    min_value=0.05,
    max_value=0.60,
    value=(0.05, 0.40),
    step=0.05
)

mu_min = mu_range[0]
mu_max = mu_range[1]

MU_VALUES_FULL = np.round(
    np.arange(
        mu_min,
        mu_max + 0.05,
        0.05
    ),
    3
)

MU_VALUES_OPT = MU_VALUES_FULL.copy()

# =========================================================
# FRICTION ASYMMETRY
# =========================================================

st.sidebar.subheader(
    "Friction Asymmetry"
)

friction_asymmetry_percent = st.sidebar.slider(
    "Left / Right Friction Asymmetry (%)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=1.0
)

friction_asymmetry = (
    friction_asymmetry_percent / 100.0
)

# =========================================================
# DRUM RADIUS
# =========================================================

RD_mm = st.sidebar.number_input(
    "Drum Radius RD (mm)",
    value=RD_MM
)

RD = RD_mm / 1000

# =========================================================
# CURRENT GEOMETRY
# =========================================================

st.sidebar.subheader(
    "Current Geometry"
)

L1_current_mm = st.sidebar.number_input(
    "Current L1 (mm)",
    value=40.0
)

L3_current_mm = st.sidebar.number_input(
    "Current L3 (mm)",
    value=143.15
)

L4_current_mm = st.sidebar.number_input(
    "Current L4 (mm)",
    value=157.0
)

theta1_current_deg = st.sidebar.number_input(
    "Current Theta1 (°)",
    value=28.0
)

theta2_current_deg = st.sidebar.number_input(
    "Current Theta2 (°)",
    value=145.0
)

# =========================================================
# CONVERSIONS
# =========================================================

L1_current = L1_current_mm / 1000
L3_current = L3_current_mm / 1000
L4_current = L4_current_mm / 1000

theta1_current = math.radians(
    theta1_current_deg
)

theta2_current = math.radians(
    theta2_current_deg
)

# =========================================================
# CURRENT PREVIEW
# =========================================================

rows_current_preview = []

for mu in MU_VALUES_FULL:

    result = calculate_bf(

        L1_current,

        L3_current,

        L4_current,

        theta1_current,

        theta2_current,

        mu,

        RD
    )

    if result is not None:

        rows_current_preview.append({

            "mu": mu,

            "BF": result["BF"],

            "CL": result["CL"],

            "CT": result["CT"],

            "CL/CT": result["CL_CT"]
        })

df_preview = pd.DataFrame(
    rows_current_preview
)

# =========================================================
# CURRENT CL/CT
# =========================================================

CURRENT_CLCT = df_preview[
    "CL/CT"
].max()

# =========================================================
# CURRENT GEOMETRY - FRICTION ASYMMETRY
# =========================================================

rows_asymmetry_current = []

for mu in MU_VALUES_FULL:

    result_asymmetry = calculate_bf_asymmetry(

        L1=L1_current,

        L3=L3_current,

        L4=L4_current,

        theta1=theta1_current,

        theta2=theta2_current,

        mu=mu,

        delta=friction_asymmetry,

        RD=RD
    )

    if result_asymmetry is not None:

        rows_asymmetry_current.append({

            "mu": mu,

            "mu_left": result_asymmetry["mu_left"],

            "mu_right": result_asymmetry["mu_right"],

            "BF_left": result_asymmetry["BF_left"],

            "BF_right": result_asymmetry["BF_right"],

            "BF_mean": result_asymmetry["BF_mean"],

            "Delta BF (%)":
                result_asymmetry["delta_BF_percent"]
        })

df_asymmetry_current = pd.DataFrame(
    rows_asymmetry_current
)

# =========================================================
# CURRENT GEOMETRY - FRICTION ASYMMETRY SUMMARY
# =========================================================

st.subheader(
    "Brake Factor Robustness to Friction Asymmetry"
)

if not df_asymmetry_current.empty:

    max_delta_bf_current = (
        df_asymmetry_current["Delta BF (%)"].max()
    )

    mean_delta_bf_current = (
        df_asymmetry_current["Delta BF (%)"].mean()
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Friction Asymmetry",
        f"{friction_asymmetry_percent:.1f}%"
    )

    r2.metric(
        "Mean ΔBF",
        f"{mean_delta_bf_current:.2f}%"
    )

    r3.metric(
        "Maximum ΔBF",
        f"{max_delta_bf_current:.2f}%"
    )

    st.dataframe(
        df_asymmetry_current,
        use_container_width=True
    )

# =========================================================
# OPTIMIZATION OBJECTIVE
# =========================================================

st.sidebar.subheader(
    "Optimization Objective"
)

optimization_mode = st.sidebar.radio(
    "Objective",
    [
        "Maximize Brake Factor",
        "Minimize Brake Factor"
    ]
)

# =========================================================
# CL/CT CONSTRAINT
# =========================================================

st.sidebar.subheader(
    "CL/CT Constraint"
)

clct_mode = st.sidebar.radio(
    "Constraint Mode",
    [
        "No Constraint",
        "Use Current CL/CT",
        "Custom CL/CT"
    ]
)

if clct_mode == "No Constraint":

    CL_CT_limit = None

elif clct_mode == "Use Current CL/CT":

    CL_CT_limit = CURRENT_CLCT

    st.sidebar.number_input(
        "Maximum CL/CT",
        value=float(CURRENT_CLCT),
        disabled=True
    )

else:

    CL_CT_limit = st.sidebar.number_input(
        "Maximum CL/CT",
        value=float(round(CURRENT_CLCT, 2)),
        step=0.05
    )

# =========================================================
# GEOMETRY LIMITS
# =========================================================

# =========================================================
# L1
# =========================================================

freeze_L1 = st.sidebar.checkbox(
    "Freeze current L1",
    value=False
)

if freeze_L1:

    L1_range_mm = (
        int(L1_current_mm),
        int(L1_current_mm)
    )

    st.sidebar.slider(
        "L1 Range (mm)",
        10,
        60,
        L1_range_mm,
        disabled=True
    )

else:

    L1_range_mm = st.sidebar.slider(
        "L1 Range (mm)",
        10,
        60,
        (35, 45)
    )

# =========================================================
# L3
# =========================================================

freeze_L3 = st.sidebar.checkbox(
    "Freeze current L3",
    value=False
)

if freeze_L3:

    L3_range_mm = (
        int(L3_current_mm),
        int(L3_current_mm)
    )

    st.sidebar.slider(
        "L3 Range (mm)",
        100,
        200,
        L3_range_mm,
        disabled=True
    )

else:

    L3_range_mm = st.sidebar.slider(
        "L3 Range (mm)",
        100.0,
        200.0,
        (138.15, 143.15),
        step = 0.05
    )

# =========================================================
# L4
# =========================================================

freeze_L4 = st.sidebar.checkbox(
    "Freeze current L4",
    value=False
)

if freeze_L4:

    L4_range_mm = (
        int(L4_current_mm),
        int(L4_current_mm)
    )

    st.sidebar.slider(
        "L4 Range (mm)",
        100,
        220,
        L4_range_mm,
        disabled=True
    )

else:

    L4_range_mm = st.sidebar.slider(
        "L4 Range (mm)",
        100,
        220,
        (152, 162)
    )

# =========================================================
# THETA1
# =========================================================

freeze_theta1 = st.sidebar.checkbox(
    "Freeze current Theta1",
    value=False
)

if freeze_theta1:

    theta1_range_deg = (
        int(theta1_current_deg),
        int(theta1_current_deg)
    )

    st.sidebar.slider(
        "Theta1 Range (°)",
        0,
        60,
        theta1_range_deg,
        disabled=True
    )

else:

    theta1_range_deg = st.sidebar.slider(
        "Theta1 Range (°)",
        0,
        60,
        (18, 38)
    )

# =========================================================
# THETA2
# =========================================================

freeze_theta2 = st.sidebar.checkbox(
    "Freeze current Theta2",
    value=False
)

if freeze_theta2:

    theta2_range_deg = (
        int(theta2_current_deg),
        int(theta2_current_deg)
    )

    st.sidebar.slider(
        "Theta2 Range (°)",
        90,
        180,
        theta2_range_deg,
        disabled=True
    )

else:

    theta2_range_deg = st.sidebar.slider(
        "Theta2 Range (°)",
        90,
        180,
        (135, 155)
    )

# =========================================================
# CONVERSIONS
# =========================================================

L1_min = L1_range_mm[0] / 1000
L1_max = L1_range_mm[1] / 1000

L3_min = L3_range_mm[0] / 1000
L3_max = L3_range_mm[1] / 1000

L4_min = L4_range_mm[0] / 1000
L4_max = L4_range_mm[1] / 1000

theta1_min = math.radians(
    theta1_range_deg[0]
)

theta1_max = math.radians(
    theta1_range_deg[1]
)

theta2_min = math.radians(
    theta2_range_deg[0]
)

theta2_max = math.radians(
    theta2_range_deg[1]
)

# =========================================================
# RUN BUTTON
# =========================================================

if st.button("Run Optimization"):

    bounds = [

        (L1_min, L1_max),

        (L3_min, L3_max),

        (L4_min, L4_max),

        (theta1_min, theta1_max),

        (theta2_min, theta2_max),
    ]

    with st.spinner(
        "Optimizing..."
    ):

        result = run_optimization(

            bounds=bounds,

            MU_VALUES_OPT=MU_VALUES_OPT,

            RD=RD,

            CL_CT_limit=CL_CT_limit,

            optimization_mode=optimization_mode
        )

    x_opt = result.x

    L1_opt = x_opt[0]
    L3_opt = x_opt[1]
    L4_opt = x_opt[2]

    theta1_opt = x_opt[3]
    theta2_opt = x_opt[4]

    # =====================================================
    # OPTIMIZED RESULTS
    # =====================================================

    rows = []

    for mu in MU_VALUES_FULL:

        result_bf = calculate_bf(

            L1_opt,

            L3_opt,

            L4_opt,

            theta1_opt,

            theta2_opt,

            mu,

            RD
        )

        current_row = df_preview[
            df_preview["mu"] == mu
        ].iloc[0]

        bf_current = current_row["BF"]

        gain_percent = (

            (
                result_bf["BF"]
                -
                bf_current
            )

            /

            bf_current

        ) * 100

        rows.append({

            "mu": mu,

            "BF": result_bf["BF"],

            "CL": result_bf["CL"],

            "CT": result_bf["CT"],

            "CL/CT": result_bf["CL_CT"],

            "BF Gain (%)": gain_percent
        })

    df = pd.DataFrame(rows)

    # =====================================================
    # HOT / COLD CONDITIONS
    # =====================================================

    hot_mu = MU_VALUES_FULL[0]
    cold_mu = MU_VALUES_FULL[-1]

    hot_current = df_preview[
        np.isclose(
            df_preview["mu"],
            hot_mu
        )
    ].iloc[0]["BF"]

    hot_opt = df[
        np.isclose(
            df["mu"],
            hot_mu
        )
    ].iloc[0]["BF"]

    hot_gain = (
        (hot_opt - hot_current)
        /
        hot_current
    ) * 100

    cold_current = df_preview[
        np.isclose(
            df_preview["mu"],
            cold_mu
        )
    ].iloc[0]["BF"]

    cold_opt = df[
        np.isclose(
            df["mu"],
            cold_mu
        )
    ].iloc[0]["BF"]

    cold_gain = (
        (cold_opt - cold_current)
        /
        cold_current
    ) * 100

    # =====================================================
    # OPTIMIZED GEOMETRY
    # =====================================================

    st.subheader(
        "Optimized Geometry"
    )

    st.info(
        f"Objective: {optimization_mode} | "
        f"CL/CT Constraint: {clct_mode}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "L1",
        f"{L1_opt*1000:.2f} mm"
    )

    c1.metric(
        "L3",
        f"{L3_opt*1000:.2f} mm"
    )

    c2.metric(
        "L4",
        f"{L4_opt*1000:.2f} mm"
    )

    c2.metric(
        "Theta1",
        f"{math.degrees(theta1_opt):.2f}°"
    )

    c3.metric(
        "Theta2",
        f"{math.degrees(theta2_opt):.2f}°"
    )

    # =====================================================
    # PERFORMANCE COMPARISON
    # =====================================================

    st.subheader(
        "Performance Comparison"
    )

    st.markdown(
        f"### Low Friction Condition (μ = {hot_mu:.2f})"
    )

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Current BF",
        f"{hot_current:.4f}"
    )

    h2.metric(
        "Optimized BF",
        f"{hot_opt:.4f}"
    )

    h3.metric(
        "Gain (%)",
        f"{hot_gain:.2f}%"
    )

    st.markdown(
        f"### High Friction Condition (μ = {cold_mu:.2f})"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current BF",
        f"{cold_current:.4f}"
    )

    c2.metric(
        "Optimized BF",
        f"{cold_opt:.4f}"
    )

    c3.metric(
        "Gain (%)",
        f"{cold_gain:.2f}%"
    )

    # =====================================================
    # TABLES
    # =====================================================

    st.subheader("Results")

    tab1, tab2 = st.tabs([
        "Current Geometry",
        "Optimized Geometry"
    ])

    with tab1:

        st.dataframe(df_preview)

    with tab2:

        st.dataframe(df)

    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        df_preview["mu"],
        df_preview["BF"],
        "-o",
        linewidth=3,
        color="dimgray",
        label="Current BF"
    )

    ax.plot(
        df_preview["mu"],
        df_preview["CL"],
        "--o",
        linewidth=2,
        color="gray",
        label="Current CL"
    )

    ax.plot(
        df_preview["mu"],
        df_preview["CT"],
        ":o",
        linewidth=2,
        color="darkgray",
        label="Current CT"
    )

    ax.plot(
        df["mu"],
        df["BF"],
        "-o",
        linewidth=3,
        color="blue",
        label="Optimized BF"
    )

    ax.plot(
        df["mu"],
        df["CL"],
        "--o",
        linewidth=2,
        color="dodgerblue",
        label="Optimized CL"
    )

    ax.plot(
        df["mu"],
        df["CT"],
        ":o",
        linewidth=2,
        color="navy",
        label="Optimized CT"
    )

    ax.set_xlabel(
        "Friction Coefficient (μ)"
    )

    ax.set_ylabel(
        "Brake Factor"
    )

    ax.set_title(
        "Current vs Optimized Performance"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    ax.legend()

    st.pyplot(fig)

    # =====================================================
    # EXCEL EXPORT
    # =====================================================

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # CURRENT GEOMETRY

        df_preview.to_excel(

            writer,

            sheet_name="Current Geometry",

            index=False
        )

        # OPTIMIZED GEOMETRY

        df.to_excel(

            writer,

            sheet_name="Optimized Geometry",

            index=False
        )

        # SUMMARY

        summary_df = pd.DataFrame({

            "Parameter": [

                "RD (mm)",

                "L1 Current (mm)",
                "L1 Optimized (mm)",

                "L3 Current (mm)",
                "L3 Optimized (mm)",

                "L4 Current (mm)",
                "L4 Optimized (mm)",

                "Theta1 Current (deg)",
                "Theta1 Optimized (deg)",

                "Theta2 Current (deg)",
                "Theta2 Optimized (deg)",

                "Maximum CL/CT",
                "Optimization Objective",
                "CL/CT Constraint Mode",
                "Mu Minimum",
                "Mu Maximum",

                "Hot Current BF",
                "Hot Optimized BF",
                "Hot Gain (%)",

                "Cold Current BF",
                "Cold Optimized BF",
                "Cold Gain (%)"
            ],

            "Value": [

                RD_mm,

                L1_current_mm,
                L1_opt * 1000,

                L3_current_mm,
                L3_opt * 1000,

                L4_current_mm,
                L4_opt * 1000,

                theta1_current_deg,
                math.degrees(theta1_opt),

                theta2_current_deg,
                math.degrees(theta2_opt),

                "No Constraint" if CL_CT_limit is None else CL_CT_limit,
                
                optimization_mode,
                clct_mode,
                mu_min,
                mu_max,

                hot_current,
                hot_opt,
                hot_gain,

                cold_current,
                cold_opt,
                cold_gain
            ]
        })

        summary_df.to_excel(

            writer,

            sheet_name="Optimization Summary",

            index=False
        )

    output.seek(0)

    st.download_button(

        label="Download Excel Report",

        data=output,

        file_name="optibf_report.xlsx",

        mime=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    )