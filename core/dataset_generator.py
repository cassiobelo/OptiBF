import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import qmc

from core.brake_model import (
    calculate_bf,
    calculate_bf_asymmetry,
    RD,
    BOUNDS,
    ROBUSTNESS_DELTA
)

# =========================================================
# CONFIGURATION
# =========================================================

N_SAMPLES = 1000

# Model parameters, design space and robustness settings
# are centralized in core.brake_model.
RANDOM_SEED = 42


# =========================================================
# LHS GENERATION
# =========================================================

def generate_lhs(n_samples, bounds, seed=42):
    sampler = qmc.LatinHypercube(
        d=len(bounds),
        seed=seed
    )

    sample = sampler.random(
        n=n_samples
    )

    lower = np.array([
        limits[0]
        for limits in bounds.values()
    ])

    upper = np.array([
        limits[1]
        for limits in bounds.values()
    ])

    sample_scaled = qmc.scale(
        sample,
        lower,
        upper
    )

    columns = list(bounds.keys())

    return pd.DataFrame(
        sample_scaled,
        columns=columns
    )


# =========================================================
# BRAKE MODEL EVALUATION
# =========================================================

def evaluate_dataset(df):
    results = []

    for _, row in df.iterrows():

        L1 = row["L1_mm"] / 1000.0
        L3 = row["L3_mm"] / 1000.0
        L4 = row["L4_mm"] / 1000.0

        theta1 = math.radians(
            row["theta1_deg"]
        )

        theta2 = math.radians(
            row["theta2_deg"]
        )

        mu = row["mu"]

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

            results.append({
                "CL": np.nan,
                "CT": np.nan,
                "CL_CT": np.nan,
                "BF": np.nan,
                "valid": False
            })

        else:

            results.append({
                "CL": result["CL"],
                "CT": result["CT"],
                "CL_CT": result["CL_CT"],
                "BF": result["BF"],
                "valid": True
            })

    results_df = pd.DataFrame(results)

    return pd.concat(
        [
            df.reset_index(drop=True),
            results_df
        ],
        axis=1
    )


# =========================================================
# ROBUSTNESS ANALYSIS
# =========================================================

def evaluate_robustness(df, delta=ROBUSTNESS_DELTA):
    robustness_results = []

    for _, row in df.iterrows():

        L1 = row["L1_mm"] / 1000.0
        L3 = row["L3_mm"] / 1000.0
        L4 = row["L4_mm"] / 1000.0

        theta1 = math.radians(
            row["theta1_deg"]
        )

        theta2 = math.radians(
            row["theta2_deg"]
        )

        mu = row["mu"]

        result = calculate_bf_asymmetry(
            L1=L1,
            L3=L3,
            L4=L4,
            theta1=theta1,
            theta2=theta2,
            mu=mu,
            delta=delta,
            RD=RD
        )

        if result is None:

            robustness_results.append({
                "mu_left": np.nan,
                "mu_right": np.nan,
                "BF_left": np.nan,
                "BF_right": np.nan,
                "BF_mean": np.nan,
                "Delta_BF_percent": np.nan,
                "robustness_valid": False
            })

        else:

            robustness_results.append({
                "mu_left": result["mu_left"],
                "mu_right": result["mu_right"],
                "BF_left": result["BF_left"],
                "BF_right": result["BF_right"],
                "BF_mean": result["BF_mean"],
                "Delta_BF_percent":
                    result["delta_BF_percent"],
                "robustness_valid": True
            })

    robustness_df = pd.DataFrame(
        robustness_results
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            robustness_df
        ],
        axis=1
    )


# =========================================================
# ROBUSTNESS SENSITIVITY TO FRICTION ASYMMETRY
# =========================================================

def evaluate_asymmetry_levels(
        df,
        asymmetry_levels=(0.00, 0.05, 0.10, 0.15, 0.20)
):
    summary_results = []

    for delta in asymmetry_levels:

        print(
            f"\nEvaluating friction asymmetry: "
            f"{delta * 100:.1f}%"
        )

        robustness_df = evaluate_robustness(
            df,
            delta=delta
        )

        valid = robustness_df[
            robustness_df["robustness_valid"]
        ]

        if len(valid) == 0:
            continue

        delta_bf = valid[
            "Delta_BF_percent"
        ]

        summary_results.append({

            "Friction_Asymmetry_percent":
                delta * 100.0,

            "Valid_Samples":
                len(valid),

            "Delta_BF_mean_percent":
                delta_bf.mean(),

            "Delta_BF_std_percent":
                delta_bf.std(),

            "Delta_BF_min_percent":
                delta_bf.min(),

            "Delta_BF_median_percent":
                delta_bf.median(),

            "Delta_BF_max_percent":
                delta_bf.max(),

            "Delta_BF_P90_percent":
                delta_bf.quantile(0.90),

            "Delta_BF_P95_percent":
                delta_bf.quantile(0.95)
        })

    return pd.DataFrame(
        summary_results
    )


# =========================================================
# DATASET QUALITY CHECK
# =========================================================

def report_dataset_quality(df):
    total = len(df)

    valid = df["valid"].sum()

    invalid = total - valid

    print("\n========================================")
    print("DATASET QUALITY")
    print("========================================")

    print(f"Total samples:     {total}")
    print(f"Valid samples:     {valid}")
    print(f"Invalid samples:   {invalid}")

    print(
        f"Validity rate:     {100 * valid / total:.2f}%"
    )

    if valid > 0:
        valid_df = df[
            df["valid"]
        ]

        print("\nBF statistics:")
        print(
            valid_df["BF"].describe()
        )

        print("\nCL/CT statistics:")
        print(
            valid_df["CL_CT"].describe()
        )


# =========================================================
# MAIN
# =========================================================

def main():
    print("========================================")
    print("OptiBF - Dataset Generator")
    print("========================================")

    print("\nGenerating LHS dataset...")

    dataset = generate_lhs(
        n_samples=N_SAMPLES,
        bounds=BOUNDS,
        seed=RANDOM_SEED
    )

    print(
        f"Generated {len(dataset)} samples."
    )

    print("\nEvaluating Brake Factor model...")

    dataset = evaluate_dataset(
        dataset
    )

    report_dataset_quality(
        dataset
    )

    # Remove invalid points from final ML dataset
    dataset_valid = dataset[
        dataset["valid"]
    ].copy()

    dataset_valid.drop(
        columns=["valid"],
        inplace=True
    )

    # =====================================================
    # SAVE DATASET
    # =====================================================

    # Dataset/output location is specific to this generation run.
    output_dir = Path("data")

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
            output_dir /
            "optibf_dataset_lhs_1000.csv"
    )

    dataset_valid.to_csv(
        output_file,
        index=False
    )

    # =====================================================
    # ROBUSTNESS ANALYSIS
    # =====================================================

    print("\n========================================")
    print("ROBUSTNESS ANALYSIS")
    print("========================================")

    print(
        f"Friction asymmetry: "
        f"{ROBUSTNESS_DELTA * 100:.1f}%"
    )

    dataset_robustness = evaluate_robustness(
        dataset_valid,
        delta=ROBUSTNESS_DELTA
    )

    robustness_valid = dataset_robustness[
        dataset_robustness["robustness_valid"]
    ].copy()

    print(
        f"Valid robustness samples: "
        f"{len(robustness_valid)}"
    )

    if len(robustness_valid) > 0:
        print("\nDelta BF statistics:")

        print(
            robustness_valid[
                "Delta_BF_percent"
            ].describe()
        )

        print("\nBF left statistics:")

        print(
            robustness_valid[
                "BF_left"
            ].describe()
        )

        print("\nBF right statistics:")

        print(
            robustness_valid[
                "BF_right"
            ].describe()
        )

    # =====================================================
    # SAVE ROBUSTNESS DATASET
    # =====================================================

    robustness_output_file = (
            output_dir /
            "optibf_dataset_lhs_1000_robustness.csv"
    )

    robustness_valid.drop(
        columns=["robustness_valid"],
        inplace=True
    )

    robustness_valid.to_csv(
        robustness_output_file,
        index=False
    )

    print("\n========================================")
    print("ROBUSTNESS DATASET SAVED")
    print("========================================")

    print(
        f"File: {robustness_output_file}"
    )

    print(
        f"Samples saved: "
        f"{len(robustness_valid)}"
    )

    # =====================================================
    # ASYMMETRY LEVEL SENSITIVITY
    # =====================================================

    print("\n========================================")
    print("ASYMMETRY LEVEL SENSITIVITY")
    print("========================================")

    asymmetry_levels = (
        0.00,
        0.05,
        0.10,
        0.15,
        0.20
    )

    asymmetry_summary = evaluate_asymmetry_levels(
        dataset_valid,
        asymmetry_levels=asymmetry_levels
    )

    print("\nSummary of Delta BF:")

    print(
        asymmetry_summary.to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE ASYMMETRY SUMMARY
    # =====================================================

    asymmetry_output_file = (
            output_dir /
            "optibf_asymmetry_sensitivity.csv"
    )

    asymmetry_summary.to_csv(
        asymmetry_output_file,
        index=False
    )

    print("\n========================================")
    print("ASYMMETRY SENSITIVITY SAVED")
    print("========================================")

    print(
        f"File: {asymmetry_output_file}"
    )


if __name__ == "__main__":
    main()