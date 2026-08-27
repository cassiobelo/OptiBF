import math
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from SALib.sample import morris
from SALib.sample import sobol

from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze

from core.brake_model import (
    calculate_bf,
    calculate_bf_asymmetry,
    RD
)


# =========================================================
# CONFIGURATION
# =========================================================

RESULTS_DIR = PROJECT_ROOT / "sensitivity_results"
GLOBAL_DIR = RESULTS_DIR / "global"
OUTPUT_DIR = GLOBAL_DIR / "csv"
PLOT_DIR = GLOBAL_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

N_TRAJECTORIES = 100

ROBUSTNESS_DELTA = 0.10


# =========================================================
# SOBOL CONFIGURATION
# =========================================================

SOBOL_N = 1024
SOBOL_SEED = 12345


# =========================================================
# MORRIS / SOBOL PROBLEM DEFINITION
# =========================================================

PROBLEM = {
    "num_vars": 6,

    "names": [
        "L1_mm",
        "L3_mm",
        "L4_mm",
        "theta1_deg",
        "theta2_deg",
        "mu"
    ],

    "bounds": [
        [35.0, 45.0],
        [138.15, 143.15],
        [152.0, 162.0],
        [18.0, 38.0],
        [135.0, 155.0],
        [0.05, 0.40]
    ]
}


# =========================================================
# GENERATE MORRIS SAMPLE
# =========================================================

def generate_morris_sample():

    print("========================================")
    print("OptiBF - Morris Sensitivity Analysis")
    print("========================================")

    print("\nGenerating Morris trajectories...")

    X = morris.sample(
        PROBLEM,
        N=N_TRAJECTORIES,
        num_levels=8,
        optimal_trajectories=None,
        local_optimization=False,
        seed=RANDOM_SEED
    )

    print(
        f"Generated {len(X)} model evaluations."
    )

    return X


# =========================================================
# EVALUATE BF
# =========================================================

def evaluate_bf(X):

    results = []

    print("\nEvaluating BF...")

    for row in X:

        L1 = row[0] / 1000.0
        L3 = row[1] / 1000.0
        L4 = row[2] / 1000.0

        theta1 = math.radians(row[3])
        theta2 = math.radians(row[4])

        mu = row[5]

        # RD is imported directly from brake_model.py.
        result = calculate_bf(
            L1=L1,
            L3=L3,
            L4=L4,
            theta1=theta1,
            theta2=theta2,
            mu=mu,
            RD=RD
        )

        if result is None:
            results.append(np.nan)
        else:
            results.append(result["BF"])

    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# EVALUATE DELTA BF
# =========================================================

def evaluate_delta_bf(X):

    results = []

    print(
        "\nEvaluating robustness "
        f"(friction asymmetry = "
        f"{ROBUSTNESS_DELTA * 100:.1f}%)..."
    )

    for row in X:

        L1 = row[0] / 1000.0
        L3 = row[1] / 1000.0
        L4 = row[2] / 1000.0

        theta1 = math.radians(row[3])
        theta2 = math.radians(row[4])

        mu = row[5]

        result = calculate_bf_asymmetry(
            L1=L1,
            L3=L3,
            L4=L4,
            theta1=theta1,
            theta2=theta2,
            mu=mu,
            delta=ROBUSTNESS_DELTA,
            RD=RD
        )

        if result is None:
            results.append(np.nan)
        else:
            results.append(
                result["delta_BF_percent"]
            )

    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# MORRIS ANALYSIS
# =========================================================

def analyze_morris(
    X,
    Y,
    response_name
):

    valid = np.isfinite(Y)

    X_valid = X[valid]
    Y_valid = Y[valid]

    print(
        f"\nValid evaluations for "
        f"{response_name}: "
        f"{len(Y_valid)} / {len(Y)}"
    )

    if len(Y_valid) != len(Y):

        print(
            "Warning: invalid evaluations "
            "were removed."
        )

    result = morris_analyze.analyze(
        PROBLEM,
        X_valid,
        Y_valid,
        conf_level=0.95,
        print_to_console=False,
        num_levels=8,
        num_resamples=1000,
        seed=RANDOM_SEED,
        scaled=True
    )

    df_result = pd.DataFrame({

        "Variable":
            PROBLEM["names"],

        "mu_star":
            result["mu_star"],

        "mu_star_conf":
            result["mu_star_conf"],

        "sigma":
            result["sigma"],

        "mu":
            result["mu"]

    })

    df_result = df_result.sort_values(
        "mu_star",
        ascending=False
    )

    print(
        "\n========================================"
    )

    print(
        f"MORRIS RESULTS - {response_name}"
    )

    print(
        "========================================"
    )

    print(
        df_result.to_string(index=False)
    )

    return df_result


# =========================================================
# SOBOL ANALYSIS
# =========================================================

def analyze_sobol(
    X,
    Y,
    problem,
    output_name
):

    print("\n========================================")
    print(f"SOBOL RESULTS - {output_name}")
    print("========================================")

    Y = np.asarray(Y, dtype=float)

    if not np.all(np.isfinite(Y)):

        raise ValueError(
            "Sobol analysis received "
            "invalid model evaluations."
        )

    Si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=True,
        conf_level=0.95,
        print_to_console=False
    )

    results = pd.DataFrame({

        "Variable":
            problem["names"],

        "S1":
            Si["S1"],

        "S1_conf":
            Si["S1_conf"],

        "ST":
            Si["ST"],

        "ST_conf":
            Si["ST_conf"]

    })

    results = results.sort_values(
        "ST",
        ascending=False
    )

    print(
        results.to_string(index=False)
    )

    return results, Si


# =========================================================
# EVALUATE SOBOL MODEL
# =========================================================

def evaluate_sobol_model(
    X,
    output="BF"
):

    results = []

    for row in X:

        L1 = row[0] / 1000.0
        L3 = row[1] / 1000.0
        L4 = row[2] / 1000.0

        theta1 = math.radians(row[3])
        theta2 = math.radians(row[4])

        mu = row[5]

        if output == "BF":

            result = calculate_bf(
                L1=L1,
                L3=L3,
                L4=L4,
                theta1=theta1,
                theta2=theta2,
                mu=mu,
                RD=RD
            )

            if result is None:
                results.append(np.nan)
            else:
                results.append(result["BF"])

        elif output == "Delta_BF":

            result = calculate_bf_asymmetry(
                L1=L1,
                L3=L3,
                L4=L4,
                theta1=theta1,
                theta2=theta2,
                mu=mu,
                delta=ROBUSTNESS_DELTA,
                RD=RD
            )

            if result is None:
                results.append(np.nan)
            else:
                results.append(
                    result["delta_BF_percent"]
                )

        else:

            raise ValueError(
                f"Unknown Sobol output: {output}"
            )

    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# SS-ANOVA SAMPLE
# =========================================================

def generate_anova_sample(
    problem,
    levels=3
):

    """
    Generates a complete factorial design.

    For 6 variables and 3 levels:

        3^6 = 729 evaluations

    This design is used for the current
    SS-Anova implementation.
    """

    axes = [
        np.linspace(
            bound[0],
            bound[1],
            levels
        )
        for bound in problem["bounds"]
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
# EVALUATE SS-ANOVA MODEL
# =========================================================

def evaluate_anova_model(
    X,
    output="BF"
):

    results = []

    for row in X:

        L1 = row[0] / 1000.0
        L3 = row[1] / 1000.0
        L4 = row[2] / 1000.0

        theta1 = math.radians(row[3])
        theta2 = math.radians(row[4])

        mu = row[5]

        if output == "BF":

            result = calculate_bf(
                L1=L1,
                L3=L3,
                L4=L4,
                theta1=theta1,
                theta2=theta2,
                mu=mu,
                RD=RD
            )

            if result is None:
                results.append(np.nan)
            else:
                results.append(result["BF"])

        elif output == "Delta BF":

            result = calculate_bf_asymmetry(
                L1=L1,
                L3=L3,
                L4=L4,
                theta1=theta1,
                theta2=theta2,
                mu=mu,
                delta=ROBUSTNESS_DELTA,
                RD=RD
            )

            if result is None:
                results.append(np.nan)
            else:
                results.append(
                    result["delta_BF_percent"]
                )

        else:

            raise ValueError(
                f"Unknown SS-Anova output: {output}"
            )

    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# SS-ANOVA ANALYSIS
# =========================================================

def analyze_ss_anova(
    X,
    Y,
    problem
):

    """
    Current SS-Anova implementation.

    Decomposes the response using a linear model
    containing:

    - intercept;
    - six main effects;
    - all second-order interactions.

    The contribution of each term is estimated from
    the variance of its fitted component.

    The resulting percentages are intended as a
    sensitivity/decomposition result and should be
    validated against the chosen SS-Anova methodology
    before being used as a final dissertation result.
    """

    valid = np.isfinite(Y)

    X = X[valid]
    Y = Y[valid]

    # -----------------------------------------------------
    # NORMALIZE FACTORS TO [-1, +1]
    # -----------------------------------------------------

    X_norm = np.zeros_like(X)

    for i, bounds in enumerate(
        problem["bounds"]
    ):

        low, high = bounds

        X_norm[:, i] = (
            2.0
            * (X[:, i] - low)
            / (high - low)
            - 1.0
        )

    # -----------------------------------------------------
    # DESIGN MATRIX
    # -----------------------------------------------------

    columns = []
    names = []

    # Intercept
    columns.append(
        np.ones(len(X_norm))
    )

    names.append("Intercept")

    # Main effects
    for i, name in enumerate(
        problem["names"]
    ):

        columns.append(
            X_norm[:, i]
        )

        names.append(name)

    # Second-order interactions
    interaction_indices = list(
        combinations(
            range(problem["num_vars"]),
            2
        )
    )

    for i, j in interaction_indices:

        columns.append(
            X_norm[:, i]
            * X_norm[:, j]
        )

        names.append(
            f"{problem['names'][i]} × "
            f"{problem['names'][j]}"
        )

    A = np.column_stack(columns)

    # -----------------------------------------------------
    # REGRESSION
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
    # CONTRIBUTION OF EACH TERM
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

        contributions.append(ss)

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

    result = pd.DataFrame({

        "Effect":
            names,

        "SS":
            contributions,

        "Contribution_percent":
            contribution_pct

    })

    result = result.sort_values(
        "Contribution_percent",
        ascending=False
    )

    return result


# =========================================================
# PLOT MORRIS
# =========================================================

def plot_morris(
    df_result,
    response_name
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        df_result["mu_star"],
        df_result["sigma"]
    )

    for _, row in df_result.iterrows():

        plt.annotate(
            row["Variable"],
            (
                row["mu_star"],
                row["sigma"]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

    plt.xlabel("μ*")
    plt.ylabel("σ")

    plt.title(
        f"Morris Sensitivity - {response_name}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_file = PLOT_DIR / ("morris_" + response_name.lower().replace(" ", "_") + ".png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}")


# =========================================================
# PLOT SS-ANOVA
# =========================================================

def plot_ss_anova(
    df_result,
    response_name,
    top_n=None
):

    df = df_result.copy()

    if top_n is not None:
        df = df.head(top_n)

    df = df.sort_values(
        "Contribution_percent",
        ascending=True
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        df["Effect"],
        df["Contribution_percent"]
    )

    plt.xlabel(
        "Contribution to variance (%)"
    )

    plt.ylabel(
        "Effect"
    )

    plt.title(
        f"SS-Anova Sensitivity - "
        f"{response_name}"
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    filename = (
        "ss_anova_"
        + response_name.lower()
        .replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    print(f"Saved plot: {PLOT_DIR / filename}")


# =========================================================
# PLOT SOBOL
# =========================================================

def plot_sobol(
    df_result,
    response_name
):

    x = np.arange(
        len(df_result)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        df_result["S1"],
        width,
        label="First-order (S1)"
    )

    plt.bar(
        x + width / 2,
        df_result["ST"],
        width,
        label="Total-order (ST)"
    )

    plt.xticks(
        x,
        df_result["Variable"],
        rotation=45
    )

    plt.ylabel(
        "Sobol index"
    )

    plt.title(
        f"Sobol Sensitivity - {response_name}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output_file = PLOT_DIR / ("sobol_" + response_name.lower().replace(" ", "_") + ".png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}")


# =========================================================
# SAVE MORRIS RESULTS
# =========================================================

def save_results(
    df_result,
    response_name
):

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    filename = (
        "morris_"
        + response_name.lower().replace(
            " ",
            "_"
        )
        + ".csv"
    )

    output_file = (
        OUTPUT_DIR /
        filename
    )

    df_result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


# =========================================================
# SAVE SOBOL RESULTS
# =========================================================

def save_sobol_results(
    df_result,
    response_name
):

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    filename = (
        "sobol_"
        + response_name.lower().replace(
            " ",
            "_"
        )
        + ".csv"
    )

    output_file = (
        OUTPUT_DIR /
        filename
    )

    df_result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


# =========================================================
# SAVE SS-ANOVA RESULTS
# =========================================================

def save_ss_anova_results(
    df_result,
    response_name
):

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    filename = (
        "ss_anova_"
        + response_name.lower()
        .replace(" ", "_")
        + ".csv"
    )

    output_file = (
        OUTPUT_DIR /
        filename
    )

    df_result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


# =========================================================
# GENERATE SOBOL SAMPLES
# =========================================================

def generate_sobol_samples(
    problem,
    N=SOBOL_N,
    seed=SOBOL_SEED
):

    print("\n========================================")
    print("SOBOL SAMPLING")
    print("========================================")

    print(
        f"Base sample size: {N}"
    )

    print(
        f"Number of variables: "
        f"{problem['num_vars']}"
    )

    X = sobol.sample(
        problem,
        N,
        calc_second_order=True,
        scramble=True,
        seed=seed
    )

    print(
        f"Generated "
        f"{len(X)} model evaluations."
    )

    return X


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n========================================")
    print("OptiBF - Sensitivity Analysis")
    print("========================================")

    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Global sensitivity results: {GLOBAL_DIR}")
    print(f"CSV output: {OUTPUT_DIR}")
    print(f"Plot output: {PLOT_DIR}")

    print(
        f"\nBrake drum radius RD: "
        f"{RD * 1000:.2f} mm"
    )

    print(
        f"Friction asymmetry delta: "
        f"{ROBUSTNESS_DELTA * 100:.1f}%"
    )

    # =====================================================
    # MORRIS ANALYSIS
    # =====================================================

    X = generate_morris_sample()

    # -----------------------------------------------------
    # BF
    # -----------------------------------------------------

    Y_BF = evaluate_bf(X)

    results_BF = analyze_morris(
        X,
        Y_BF,
        "BF"
    )

    save_results(
        results_BF,
        "BF"
    )

    plot_morris(
        results_BF,
        "BF"
    )

    # -----------------------------------------------------
    # DELTA BF
    # -----------------------------------------------------

    Y_Delta_BF = evaluate_delta_bf(X)

    results_Delta_BF = analyze_morris(
        X,
        Y_Delta_BF,
        "Delta BF"
    )

    save_results(
        results_Delta_BF,
        "Delta BF"
    )

    plot_morris(
        results_Delta_BF,
        "Delta BF"
    )

    # =====================================================
    # SOBOL ANALYSIS
    # =====================================================

    print("\n========================================")
    print("OptiBF - Sobol Sensitivity Analysis")
    print("========================================")

    X_Sobol = generate_sobol_samples(
        PROBLEM
    )

    # -----------------------------------------------------
    # BF
    # -----------------------------------------------------

    print("\nEvaluating BF for Sobol...")

    Y_BF_Sobol = evaluate_sobol_model(
        X_Sobol,
        output="BF"
    )

    valid_BF = np.isfinite(
        Y_BF_Sobol
    )

    print(
        f"Valid evaluations for BF: "
        f"{np.sum(valid_BF)} / "
        f"{len(Y_BF_Sobol)}"
    )

    if not np.all(valid_BF):

        raise ValueError(
            "Sobol BF analysis requires "
            "all model evaluations to be valid."
        )

    results_BF_Sobol, Si_BF = analyze_sobol(
        X_Sobol,
        Y_BF_Sobol,
        PROBLEM,
        "BF"
    )

    save_sobol_results(
        results_BF_Sobol,
        "BF"
    )

    plot_sobol(
        results_BF_Sobol,
        "BF"
    )

    # -----------------------------------------------------
    # DELTA BF
    # -----------------------------------------------------

    print(
        "\nEvaluating Delta BF "
        f"(friction asymmetry = "
        f"{ROBUSTNESS_DELTA * 100:.1f}%)..."
    )

    Y_Delta_BF_Sobol = evaluate_sobol_model(
        X_Sobol,
        output="Delta_BF"
    )

    valid_Delta_BF = np.isfinite(
        Y_Delta_BF_Sobol
    )

    print(
        f"Valid evaluations for Delta BF: "
        f"{np.sum(valid_Delta_BF)} / "
        f"{len(Y_Delta_BF_Sobol)}"
    )

    if not np.all(valid_Delta_BF):

        raise ValueError(
            "Sobol Delta BF analysis requires "
            "all model evaluations to be valid."
        )

    results_Delta_BF_Sobol, Si_Delta_BF = analyze_sobol(
        X_Sobol,
        Y_Delta_BF_Sobol,
        PROBLEM,
        "Delta BF"
    )

    save_sobol_results(
        results_Delta_BF_Sobol,
        "Delta BF"
    )

    plot_sobol(
        results_Delta_BF_Sobol,
        "Delta BF"
    )

    # =====================================================
    # SS-ANOVA ANALYSIS
    # =====================================================

    print("\n========================================")
    print("OptiBF - SS-ANOVA Sensitivity Analysis")
    print("========================================")

    X_ANOVA = generate_anova_sample(
        PROBLEM,
        levels=3
    )

    # -----------------------------------------------------
    # BF
    # -----------------------------------------------------

    print(
        "\nEvaluating BF for SS-Anova..."
    )

    Y_BF_ANOVA = evaluate_anova_model(
        X_ANOVA,
        output="BF"
    )

    results_BF_ANOVA = analyze_ss_anova(
        X_ANOVA,
        Y_BF_ANOVA,
        PROBLEM
    )

    print(
        "\n========================================"
    )

    print(
        "SS-ANOVA RESULTS - BF"
    )

    print(
        "========================================"
    )

    print(
        results_BF_ANOVA.to_string(
            index=False
        )
    )

    save_ss_anova_results(
        results_BF_ANOVA,
        "BF"
    )

    plot_ss_anova(
        results_BF_ANOVA,
        "BF",
        top_n=15
    )

    # -----------------------------------------------------
    # DELTA BF
    # -----------------------------------------------------

    print(
        "\nEvaluating Delta BF "
        "for SS-Anova..."
    )

    Y_Delta_BF_ANOVA = evaluate_anova_model(
        X_ANOVA,
        output="Delta BF"
    )

    results_Delta_BF_ANOVA = analyze_ss_anova(
        X_ANOVA,
        Y_Delta_BF_ANOVA,
        PROBLEM
    )

    print(
        "\n========================================"
    )

    print(
        "SS-ANOVA RESULTS - Delta BF"
    )

    print(
        "========================================"
    )

    print(
        results_Delta_BF_ANOVA.to_string(
            index=False
        )
    )

    save_ss_anova_results(
        results_Delta_BF_ANOVA,
        "Delta BF"
    )

    plot_ss_anova(
        results_Delta_BF_ANOVA,
        "Delta BF",
        top_n=15
    )

    print("\n========================================")
    print("VERIFYING OUTPUT FILES")
    print("========================================")

    expected_csv = [
        "morris_bf.csv", "morris_delta_bf.csv",
        "sobol_bf.csv", "sobol_delta_bf.csv",
        "ss_anova_bf.csv", "ss_anova_delta_bf.csv"
    ]

    expected_png = [
        "morris_bf.png", "morris_delta_bf.png",
        "sobol_bf.png", "sobol_delta_bf.png",
        "ss_anova_bf.png", "ss_anova_delta_bf.png"
    ]

    print("\nCSV:")
    for filename in expected_csv:
        path = OUTPUT_DIR / filename
        print(f"[OK] {path}" if path.exists() else f"[MISSING] {path}")

    print("\nPNG:")
    for filename in expected_png:
        path = PLOT_DIR / filename
        print(f"[OK] {path}" if path.exists() else f"[MISSING] {path}")

    print("\n========================================")
    print("SENSITIVITY ANALYSIS COMPLETED")
    print("========================================")


if __name__ == "__main__":

    try:
        main()
    except Exception as error:
        print("\n========================================")
        print("EXECUTION ERROR")
        print("========================================")
        print(f"\nType: {type(error).__name__}")
        print(f"\nMessage: {error}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n\nPress ENTER to close...")