"""
OptiBF - Geometry-Focused Sensitivity Analysis

Purpose
-------
Evaluate the influence of the five brake-geometry variables
with the friction coefficient fixed at FIXED_MU.

Outputs:
    - Morris: BF and Delta BF
    - Sobol: BF and Delta BF
    - SS-ANOVA: BF and Delta BF
    - CSV files with numerical results
    - PNG plots

Execution:
    Recommended:
        python -m core.geometry_sensitivity_analysis

    Also supports direct execution / double-click from Windows
    because the project root is added to sys.path before importing
    the local "core" package.

Output structure:
    Brake Factor Optimizer/
        sensitivity_results/
            mu_fixed/
                csv/
                plots/
"""

import math
import sys
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from SALib.sample import morris
from SALib.sample import sobol
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze


# =========================================================
# PROJECT PATH
# =========================================================

# This file is inside:
#
# Brake Factor Optimizer/
#     core/
#         geometry_sensitivity_analysis.py
#
# parents[0] -> core
# parents[1] -> Brake Factor Optimizer

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# Make the project root visible to Python BEFORE importing
# the local "core" package. This also supports double-click
# execution from Windows Explorer.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# Local project import
from core.brake_model import (
    calculate_bf,
    calculate_bf_asymmetry,
    RD
)


# =========================================================
# OUTPUT PATHS
# =========================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "sensitivity_results"
)

MU_FIXED_DIR = (
    RESULTS_DIR
    / "mu_fixed"
)

OUTPUT_DIR = (
    MU_FIXED_DIR
    / "csv"
)

PLOT_DIR = (
    MU_FIXED_DIR
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42

N_TRAJECTORIES = 100

SOBOL_N = 1024

SOBOL_SEED = 12345

ROBUSTNESS_DELTA = 0.10

# Friction is deliberately fixed in this analysis.
FIXED_MU = 0.30


# =========================================================
# GEOMETRY-ONLY PROBLEM
# =========================================================

PROBLEM_GEOMETRY = {

    "num_vars": 5,

    "names": [
        "L1_mm",
        "L3_mm",
        "L4_mm",
        "theta1_deg",
        "theta2_deg"
    ],

    "bounds": [
        [35.0, 45.0],
        [138.15, 143.15],
        [152.0, 162.0],
        [18.0, 38.0],
        [135.0, 155.0]
    ]
}


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(
    X,
    output="BF"
):
    """
    Evaluate the brake model for a matrix of geometry samples.

    Parameters
    ----------
    X : ndarray
        Columns:
        L1 [mm], L3 [mm], L4 [mm],
        theta1 [deg], theta2 [deg]

    output : str
        "BF" or "Delta_BF"

    Returns
    -------
    ndarray
        Model response for each sample.
    """

    results = []

    for row in X:

        L1 = row[0] / 1000.0
        L3 = row[1] / 1000.0
        L4 = row[2] / 1000.0

        theta1 = math.radians(
            row[3]
        )

        theta2 = math.radians(
            row[4]
        )

        # -------------------------------------------------
        # BRAKE FACTOR
        # -------------------------------------------------

        if output == "BF":

            result = calculate_bf(
                L1=L1,
                L3=L3,
                L4=L4,
                theta1=theta1,
                theta2=theta2,
                mu=FIXED_MU,
                RD=RD
            )

            if result is None:

                results.append(
                    np.nan
                )

            else:

                results.append(
                    result["BF"]
                )

        # -------------------------------------------------
        # ROBUSTNESS TO FRICTION ASYMMETRY
        # -------------------------------------------------

        elif output == "Delta_BF":

            result = calculate_bf_asymmetry(
                L1=L1,
                L3=L3,
                L4=L4,
                theta1=theta1,
                theta2=theta2,
                mu=FIXED_MU,
                delta=ROBUSTNESS_DELTA,
                RD=RD
            )

            if result is None:

                results.append(
                    np.nan
                )

            else:

                results.append(
                    result["delta_BF_percent"]
                )

        else:

            raise ValueError(
                f"Unknown output: {output}"
            )

    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# MORRIS
# =========================================================

def run_morris(
    output_name
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"MORRIS - Geometry Only - "
        f"{output_name}"
    )

    print(
        "=" * 60
    )

    X = morris.sample(
        PROBLEM_GEOMETRY,
        N=N_TRAJECTORIES,
        num_levels=8,
        optimal_trajectories=None,
        local_optimization=False,
        seed=RANDOM_SEED
    )

    print(
        f"Generated {len(X)} evaluations."
    )

    Y = evaluate_model(
        X,
        output_name
    )

    valid = np.isfinite(
        Y
    )

    print(
        f"Valid evaluations: "
        f"{np.sum(valid)} / {len(Y)}"
    )

    if not np.all(valid):

        raise ValueError(
            "Morris received invalid evaluations."
        )

    result = morris_analyze.analyze(
        PROBLEM_GEOMETRY,
        X,
        Y,
        conf_level=0.95,
        print_to_console=False,
        num_levels=8,
        num_resamples=1000,
        seed=RANDOM_SEED,
        scaled=True
    )

    df = pd.DataFrame({

        "Variable":
            PROBLEM_GEOMETRY["names"],

        "mu_star":
            result["mu_star"],

        "mu_star_conf":
            result["mu_star_conf"],

        "sigma":
            result["sigma"],

        "mu":
            result["mu"]

    })

    df = df.sort_values(
        "mu_star",
        ascending=False
    )

    filename = (
        "morris_geometry_"
        + output_name.lower()
        + ".csv"
    )

    csv_path = (
        OUTPUT_DIR
        / filename
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"[OK] CSV:\n{csv_path}"
    )

    plot_morris(
        df,
        output_name
    )

    return df


# =========================================================
# SOBOL
# =========================================================

def run_sobol(
    output_name
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"SOBOL - Geometry Only - "
        f"{output_name}"
    )

    print(
        "=" * 60
    )

    X = sobol.sample(
        PROBLEM_GEOMETRY,
        SOBOL_N,
        calc_second_order=True,
        scramble=True,
        seed=SOBOL_SEED
    )

    print(
        f"Generated {len(X)} evaluations."
    )

    Y = evaluate_model(
        X,
        output_name
    )

    valid = np.isfinite(
        Y
    )

    print(
        f"Valid evaluations: "
        f"{np.sum(valid)} / {len(Y)}"
    )

    if not np.all(valid):

        raise ValueError(
            "Sobol received invalid evaluations."
        )

    Si = sobol_analyze.analyze(
        PROBLEM_GEOMETRY,
        Y,
        calc_second_order=True,
        conf_level=0.95,
        print_to_console=False
    )

    df = pd.DataFrame({

        "Variable":
            PROBLEM_GEOMETRY["names"],

        "S1":
            Si["S1"],

        "S1_conf":
            Si["S1_conf"],

        "ST":
            Si["ST"],

        "ST_conf":
            Si["ST_conf"]

    })

    df = df.sort_values(
        "ST",
        ascending=False
    )

    filename = (
        "sobol_geometry_"
        + output_name.lower()
        + ".csv"
    )

    csv_path = (
        OUTPUT_DIR
        / filename
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"[OK] CSV:\n{csv_path}"
    )

    plot_sobol(
        df,
        output_name
    )

    return df


# =========================================================
# SS-ANOVA SAMPLE
# =========================================================

def generate_anova_sample(
    levels=3
):

    axes = [

        np.linspace(
            bound[0],
            bound[1],
            levels
        )

        for bound in
        PROBLEM_GEOMETRY["bounds"]
    ]

    mesh = np.meshgrid(
        *axes,
        indexing="ij"
    )

    X = np.column_stack(
        [
            m.ravel()
            for m in mesh
        ]
    )

    print(
        f"Generated {len(X)} "
        f"factorial evaluations."
    )

    return X


# =========================================================
# SS-ANOVA
# =========================================================

def run_ss_anova(
    output_name
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"SS-ANOVA - Geometry Only - "
        f"{output_name}"
    )

    print(
        "=" * 60
    )

    X = generate_anova_sample(
        levels=3
    )

    Y = evaluate_model(
        X,
        output_name
    )

    valid = np.isfinite(
        Y
    )

    print(
        f"Valid evaluations: "
        f"{np.sum(valid)} / {len(Y)}"
    )

    if not np.all(valid):

        raise ValueError(
            "SS-ANOVA received invalid evaluations."
        )

    # -----------------------------------------------------
    # Normalize geometry variables to [-1, +1]
    # -----------------------------------------------------

    X_norm = np.zeros_like(
        X
    )

    for i, bounds in enumerate(
        PROBLEM_GEOMETRY["bounds"]
    ):

        low, high = bounds

        X_norm[:, i] = (

            2.0
            * (
                X[:, i]
                - low
            )
            / (
                high
                - low
            )
            - 1.0

        )

    # -----------------------------------------------------
    # Design matrix
    # -----------------------------------------------------

    columns = [
        np.ones(
            len(X_norm)
        )
    ]

    names = [
        "Intercept"
    ]

    # Main effects
    for i, name in enumerate(
        PROBLEM_GEOMETRY["names"]
    ):

        columns.append(
            X_norm[:, i]
        )

        names.append(
            name
        )

    # All second-order interactions
    for i, j in combinations(
        range(
            PROBLEM_GEOMETRY["num_vars"]
        ),
        2
    ):

        columns.append(
            X_norm[:, i]
            * X_norm[:, j]
        )

        names.append(
            f"{PROBLEM_GEOMETRY['names'][i]} × "
            f"{PROBLEM_GEOMETRY['names'][j]}"
        )

    A = np.column_stack(
        columns
    )

    # -----------------------------------------------------
    # Regression
    # -----------------------------------------------------

    model = LinearRegression(
        fit_intercept=False
    )

    model.fit(
        A,
        Y
    )

    coefficients = model.coef_

    # -----------------------------------------------------
    # Variance contribution
    # -----------------------------------------------------

    contributions = []

    for j in range(
        len(names)
    ):

        term = (
            A[:, j]
            * coefficients[j]
        )

        ss = np.sum(
            (
                term
                - np.mean(term)
            ) ** 2
        )

        contributions.append(
            ss
        )

    contributions = np.asarray(
        contributions
    )

    # Remove intercept
    contributions = contributions[1:]
    names = names[1:]

    total = np.sum(
        contributions
    )

    if total > 0:

        contribution_pct = (
            contributions
            / total
            * 100.0
        )

    else:

        contribution_pct = np.zeros(
            len(contributions)
        )

    df = pd.DataFrame({

        "Effect":
            names,

        "SS":
            contributions,

        "Contribution_percent":
            contribution_pct

    })

    df = df.sort_values(
        "Contribution_percent",
        ascending=False
    )

    filename = (
        "ss_anova_geometry_"
        + output_name.lower()
        + ".csv"
    )

    csv_path = (
        OUTPUT_DIR
        / filename
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"[OK] CSV:\n{csv_path}"
    )

    plot_ss_anova(
        df,
        output_name
    )

    return df


# =========================================================
# PLOT HELPERS
# =========================================================

def label_variable(
    variable
):

    labels = {

        "L1_mm":
            "L1",

        "L3_mm":
            "L3",

        "L4_mm":
            "L4",

        "theta1_deg":
            r"$\theta_1$",

        "theta2_deg":
            r"$\theta_2$"

    }

    return labels.get(
        variable,
        variable
    )


# =========================================================
# PLOT MORRIS
# =========================================================

def plot_morris(
    df,
    output_name
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        df["mu_star"],
        df["sigma"],
        s=80
    )

    for _, row in df.iterrows():

        ax.annotate(
            label_variable(
                row["Variable"]
            ),
            (
                row["mu_star"],
                row["sigma"]
            ),
            xytext=(
                7,
                7
            ),
            textcoords="offset points"
        )

    ax.set_xlabel(
        r"$\mu^*$"
    )

    ax.set_ylabel(
        r"$\sigma$"
    )

    ax.set_title(
        f"Morris — Geometry Only — "
        f"{output_name} "
        f"(μ = {FIXED_MU:.2f})"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    filename = (
        "morris_geometry_"
        + output_name.lower()
        + ".png"
    )

    path = (
        PLOT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] PNG:\n{path}"
    )


# =========================================================
# PLOT SOBOL
# =========================================================

def plot_sobol(
    df,
    output_name
):

    x = np.arange(
        len(df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        df["S1"],
        width,
        label=r"$S_1$"
    )

    ax.bar(
        x + width / 2,
        df["ST"],
        width,
        label=r"$S_T$"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            label_variable(
                value
            )
            for value
            in df["Variable"]
        ]
    )

    ax.set_ylabel(
        "Sensitivity Index"
    )

    ax.set_xlabel(
        "Geometry variable"
    )

    ax.set_title(
        f"Sobol — Geometry Only — "
        f"{output_name} "
        f"(μ = {FIXED_MU:.2f})"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    filename = (
        "sobol_geometry_"
        + output_name.lower()
        + ".png"
    )

    path = (
        PLOT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] PNG:\n{path}"
    )


# =========================================================
# PLOT SS-ANOVA
# =========================================================

def plot_ss_anova(
    df,
    output_name,
    top_n=15
):

    plot_df = df.head(
        top_n
    ).copy()

    plot_df = plot_df.sort_values(
        "Contribution_percent",
        ascending=True
    )

    labels = []

    for effect in plot_df[
        "Effect"
    ]:

        parts = [

            label_variable(
                part.strip()
            )

            for part
            in str(effect).split("×")

        ]

        labels.append(
            " × ".join(parts)
        )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        labels,
        plot_df[
            "Contribution_percent"
        ]
    )

    ax.set_xlabel(
        "Contribution to variance (%)"
    )

    ax.set_ylabel(
        "Effect"
    )

    ax.set_title(
        f"SS-ANOVA — Geometry Only — "
        f"{output_name} "
        f"(μ = {FIXED_MU:.2f})"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    max_value = plot_df[
        "Contribution_percent"
    ].max()

    for i, value in enumerate(
        plot_df[
            "Contribution_percent"
        ]
    ):

        ax.text(
            value
            + max(
                max_value * 0.01,
                0.01
            ),
            i,
            f"{value:.2f}%",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    filename = (
        "ss_anova_geometry_"
        + output_name.lower()
        + ".png"
    )

    path = (
        PLOT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] PNG:\n{path}"
    )


# =========================================================
# VERIFY OUTPUTS
# =========================================================

def verify_outputs():

    print(
        "\n" + "=" * 60
    )

    print(
        "VERIFYING OUTPUT FILES"
    )

    print(
        "=" * 60
    )

    expected_csv = [

        "morris_geometry_bf.csv",

        "morris_geometry_delta_bf.csv",

        "sobol_geometry_bf.csv",

        "sobol_geometry_delta_bf.csv",

        "ss_anova_geometry_bf.csv",

        "ss_anova_geometry_delta_bf.csv"

    ]

    expected_png = [

        "morris_geometry_bf.png",

        "morris_geometry_delta_bf.png",

        "sobol_geometry_bf.png",

        "sobol_geometry_delta_bf.png",

        "ss_anova_geometry_bf.png",

        "ss_anova_geometry_delta_bf.png"

    ]

    print(
        "\nCSV:"
    )

    for filename in expected_csv:

        path = (
            OUTPUT_DIR
            / filename
        )

        if path.exists():

            print(
                f"[OK] {path}"
            )

        else:

            print(
                f"[MISSING] {path}"
            )

    print(
        "\nPNG:"
    )

    for filename in expected_png:

        path = (
            PLOT_DIR
            / filename
        )

        if path.exists():

            print(
                f"[OK] {path}"
            )

        else:

            print(
                f"[MISSING] {path}"
            )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 60
    )

    print(
        "OptiBF - Geometry-Focused "
        "Sensitivity Analysis"
    )

    print(
        "=" * 60
    )

    print(
        f"\nProject directory:"
        f"\n{PROJECT_ROOT}"
    )

    print(
        f"\nSensitivity results:"
        f"\n{RESULTS_DIR}"
    )

    print(
        f"\nμ-fixed results:"
        f"\n{MU_FIXED_DIR}"
    )

    print(
        f"\nCSV output:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nPlot output:"
        f"\n{PLOT_DIR}"
    )

    print(
        f"\nBrake drum radius RD:"
        f" {RD * 1000:.2f} mm"
    )

    print(
        f"Fixed friction coefficient:"
        f" μ = {FIXED_MU:.2f}"
    )

    print(
        f"Friction asymmetry:"
        f" {ROBUSTNESS_DELTA * 100:.1f}%"
    )

    print(
        "\nVariables:"
    )

    for name, bounds in zip(
        PROBLEM_GEOMETRY["names"],
        PROBLEM_GEOMETRY["bounds"]
    ):

        print(
            f"  {name}: "
            f"{bounds[0]} -> {bounds[1]}"
        )

    # -----------------------------------------------------
    # MORRIS
    # -----------------------------------------------------

    run_morris(
        "BF"
    )

    run_morris(
        "Delta_BF"
    )

    # -----------------------------------------------------
    # SOBOL
    # -----------------------------------------------------

    run_sobol(
        "BF"
    )

    run_sobol(
        "Delta_BF"
    )

    # -----------------------------------------------------
    # SS-ANOVA
    # -----------------------------------------------------

    run_ss_anova(
        "BF"
    )

    run_ss_anova(
        "Delta_BF"
    )

    # -----------------------------------------------------
    # VERIFY
    # -----------------------------------------------------

    verify_outputs()

    print(
        "\n" + "=" * 60
    )

    print(
        "COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            "\n" + "=" * 60
        )

        print(
            "EXECUTION ERROR"
        )

        print(
            "=" * 60
        )

        print(
            f"\nType: "
            f"{type(error).__name__}"
        )

        print(
            f"\nMessage:\n{error}"
        )

        print(
            "\nFull traceback:"
        )

        traceback.print_exc()

    finally:

        input(
            "\n\nPress ENTER to close..."
        )