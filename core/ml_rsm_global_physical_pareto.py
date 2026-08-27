# =========================================================
# OptiBF - GLOBAL PHYSICAL PARETO ANALYSIS
# XGBoost vs RSM vs FULL PHYSICAL MODEL
# =========================================================
#
# Purpose
# -------
# Evaluate ALL 30,000 common candidates with the physical
# model, construct the GLOBAL physical Pareto front, and
# identify which Pareto points had previously been identified
# by XGBoost, RSM, both, or neither.
#
# This is intentionally different from
# ml_rsm_multiobjective_fair_comparison.py:
#
#   Previous diagnostic:
#       surrogate Pareto -> physical validation -> Pareto
#
#   This analysis:
#       ALL 30,000 candidates -> physical model -> GLOBAL Pareto
#
# Therefore this is the definitive physical reference for the
# ML vs RSM optimization comparison.
# =========================================================

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.brake_model import RD, calculate_bf, calculate_bf_asymmetry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "ml_rsm_comparison"
OUTPUT_DIR = INPUT_DIR / "global_physical_pareto"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MU = 0.30
ASYMMETRY_DELTA = 0.10

REFERENCE = {
    "BF": 1.55877649,
    "Delta_BF": 28.12134928,
}

# ---------------------------------------------------------
# PARETO
# ---------------------------------------------------------

def pareto_mask(bf, delta):
    """
    Objective:
        maximize BF
        minimize Delta BF

    A point is dominated if another point has:
        BF >= current BF
        Delta <= current Delta
    with at least one strict improvement.
    """
    values = np.column_stack([bf, delta])
    n = len(values)

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        bf_i, delta_i = values[i]

        dominated = (
            (values[:, 0] >= bf_i)
            & (values[:, 1] <= delta_i)
            & (
                (values[:, 0] > bf_i)
                | (values[:, 1] < delta_i)
            )
        )

        dominated[i] = False

        if np.any(dominated):
            is_pareto[i] = False

    return is_pareto


# ---------------------------------------------------------
# PHYSICAL MODEL
# ---------------------------------------------------------

def evaluate_physical(df):
    bf_values = []
    delta_values = []
    cl_values = []
    ct_values = []
    clct_values = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        bf = calculate_bf(
            L1=row["L1_mm"] / 1000.0,
            L3=row["L3_mm"] / 1000.0,
            L4=row["L4_mm"] / 1000.0,
            theta1=math.radians(row["theta1_deg"]),
            theta2=math.radians(row["theta2_deg"]),
            mu=MU,
            RD=RD,
        )

        delta = calculate_bf_asymmetry(
            L1=row["L1_mm"] / 1000.0,
            L3=row["L3_mm"] / 1000.0,
            L4=row["L4_mm"] / 1000.0,
            theta1=math.radians(row["theta1_deg"]),
            theta2=math.radians(row["theta2_deg"]),
            mu=MU,
            delta=ASYMMETRY_DELTA,
            RD=RD,
        )

        if bf is None:
            bf_values.append(np.nan)
            cl_values.append(np.nan)
            ct_values.append(np.nan)
            clct_values.append(np.nan)
        else:
            bf_values.append(bf["BF"])
            cl_values.append(bf["CL"])
            ct_values.append(bf["CT"])
            clct_values.append(bf["CL_CT"])

        if delta is None:
            delta_values.append(np.nan)
        else:
            delta_values.append(delta["delta_BF_percent"])

        if i % 5000 == 0 or i == total:
            print(f"  Physical evaluations: {i}/{total}")

    result = df.copy()
    result["BF_physical"] = bf_values
    result["Delta_BF_physical"] = delta_values
    result["CL_physical"] = cl_values
    result["CT_physical"] = ct_values
    result["CL_CT_physical"] = clct_values

    result["BF_change_percent"] = (
        (result["BF_physical"] - REFERENCE["BF"])
        / REFERENCE["BF"]
    ) * 100.0

    result["Delta_BF_change_pp"] = (
        result["Delta_BF_physical"] - REFERENCE["Delta_BF"]
    )

    result["Win_Win_Physical"] = (
        (result["BF_physical"] > REFERENCE["BF"])
        & (result["Delta_BF_physical"] < REFERENCE["Delta_BF"])
    )

    return result


# ---------------------------------------------------------
# ORIGIN OF CANDIDATES
# ---------------------------------------------------------

def add_method_origin(df):
    xgb = pd.read_csv(INPUT_DIR / "xgb_pareto.csv")
    rsm = pd.read_csv(INPUT_DIR / "rsm_pareto.csv")

    xgb_ids = set(xgb["candidate_id"].astype(int))
    rsm_ids = set(rsm["candidate_id"].astype(int))

    origins = []

    for cid in df["candidate_id"].astype(int):
        in_xgb = cid in xgb_ids
        in_rsm = cid in rsm_ids

        if in_xgb and in_rsm:
            origins.append("Both")
        elif in_xgb:
            origins.append("XGBoost")
        elif in_rsm:
            origins.append("RSM")
        else:
            origins.append("Neither")

    result = df.copy()
    result["Surrogate_Origin"] = origins

    result["XGBoost_Pareto"] = result["candidate_id"].astype(int).isin(xgb_ids)
    result["RSM_Pareto"] = result["candidate_id"].astype(int).isin(rsm_ids)

    return result


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("OptiBF - GLOBAL PHYSICAL PARETO ANALYSIS")
    print("=" * 70)

    print("\nLoading common 30,000 candidates...")

    candidates_file = INPUT_DIR / "common_candidates_30000.csv"

    if not candidates_file.exists():
        raise FileNotFoundError(
            f"Missing file: {candidates_file}\n"
            "Run ml_rsm_multiobjective_fair_comparison.py first."
        )

    candidates = pd.read_csv(candidates_file)

    print(f"Candidates loaded: {len(candidates)}")

    if len(candidates) != 30000:
        print(
            f"WARNING: expected 30000 candidates, "
            f"found {len(candidates)}."
        )

    print("\nEvaluating ALL candidates with the physical model...")

    physical = evaluate_physical(candidates)

    valid = (
        np.isfinite(physical["BF_physical"])
        & np.isfinite(physical["Delta_BF_physical"])
    )

    physical = physical.loc[valid].copy().reset_index(drop=True)

    print(f"Valid physical evaluations: {len(physical)}")

    print("\nIdentifying origin of each candidate...")

    physical = add_method_origin(physical)

    print("\nConstructing GLOBAL physical Pareto front...")

    mask = pareto_mask(
        physical["BF_physical"].to_numpy(),
        physical["Delta_BF_physical"].to_numpy(),
    )

    physical["Global_Physical_Pareto"] = mask

    global_pareto = (
        physical.loc[mask]
        .copy()
        .sort_values("BF_physical")
        .reset_index(drop=True)
    )

    print(f"GLOBAL physical Pareto solutions: {len(global_pareto)}")

    # -----------------------------------------------------
    # WIN-WIN GLOBAL PHYSICAL PARETO
    # -----------------------------------------------------

    global_winwin = global_pareto[
        global_pareto["Win_Win_Physical"]
    ].copy()

    print(
        f"Global physical Pareto Win-Win solutions: "
        f"{len(global_winwin)}"
    )

    # -----------------------------------------------------
    # ORIGIN SUMMARY
    # -----------------------------------------------------

    origin_summary = (
        global_pareto["Surrogate_Origin"]
        .value_counts()
        .rename_axis("Surrogate_Origin")
        .reset_index(name="Count")
    )

    print("\nGLOBAL PHYSICAL PARETO — ORIGIN")
    print(origin_summary.to_string(index=False))

    print("\nGLOBAL PHYSICAL PARETO — WIN-WIN ORIGIN")

    if len(global_winwin):
        win_origin_summary = (
            global_winwin["Surrogate_Origin"]
            .value_counts()
            .rename_axis("Surrogate_Origin")
            .reset_index(name="Count")
        )
        print(win_origin_summary.to_string(index=False))
    else:
        win_origin_summary = pd.DataFrame(
            columns=["Surrogate_Origin", "Count"]
        )
        print("No physical Win-Win solutions.")

    # -----------------------------------------------------
    # BEST PHYSICAL WIN-WIN BY EACH OBJECTIVE
    # -----------------------------------------------------

    if len(global_winwin):

        max_bf = global_winwin.loc[
            global_winwin["BF_physical"].idxmax()
        ]

        min_delta = global_winwin.loc[
            global_winwin["Delta_BF_physical"].idxmin()
        ]

        print("\nBEST GLOBAL PHYSICAL WIN-WIN")
        print(
            f"Maximum BF: candidate {int(max_bf.candidate_id)} | "
            f"origin={max_bf.Surrogate_Origin} | "
            f"BF={max_bf.BF_physical:.8f} | "
            f"Delta BF={max_bf.Delta_BF_physical:.8f}%"
        )

        print(
            f"Minimum Delta BF: candidate {int(min_delta.candidate_id)} | "
            f"origin={min_delta.Surrogate_Origin} | "
            f"BF={min_delta.BF_physical:.8f} | "
            f"Delta BF={min_delta.Delta_BF_physical:.8f}%"
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    physical.to_csv(
        OUTPUT_DIR / "all_30000_physical_evaluated.csv",
        index=False,
    )

    global_pareto.to_csv(
        OUTPUT_DIR / "global_physical_pareto.csv",
        index=False,
    )

    global_winwin.to_csv(
        OUTPUT_DIR / "global_physical_pareto_winwin.csv",
        index=False,
    )

    origin_summary.to_csv(
        OUTPUT_DIR / "global_physical_pareto_origin_summary.csv",
        index=False,
    )

    win_origin_summary.to_csv(
        OUTPUT_DIR / "global_physical_pareto_winwin_origin_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    # All candidates in light gray
    plt.scatter(
        physical["BF_physical"],
        physical["Delta_BF_physical"],
        s=8,
        alpha=0.12,
        label="30,000 candidatos físicos",
    )

    # Global physical Pareto
    plt.plot(
        global_pareto["BF_physical"],
        global_pareto["Delta_BF_physical"],
        linewidth=2.2,
        label="Pareto física global",
    )

    # Highlight Pareto origin
    for origin, marker in [
        ("XGBoost", "o"),
        ("RSM", "s"),
        ("Both", "D"),
    ]:
        subset = global_pareto[
            global_pareto["Surrogate_Origin"] == origin
        ]

        if len(subset):
            plt.scatter(
                subset["BF_physical"],
                subset["Delta_BF_physical"],
                s=65,
                marker=marker,
                label=f"Pareto física — {origin}",
                zorder=5,
            )

    # Reference
    plt.scatter(
        REFERENCE["BF"],
        REFERENCE["Delta_BF"],
        marker="*",
        s=220,
        label="Referência",
        zorder=10,
    )

    # Win-Win boundary
    plt.axvline(
        REFERENCE["BF"],
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        REFERENCE["Delta_BF"],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Brake Factor (BF)")
    plt.ylabel("ΔBF (%)")
    plt.title(
        "Pareto Física Global — XGBoost vs. RSM"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_physical_pareto.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(
        OUTPUT_DIR / "all_30000_physical_evaluated.csv"
    )
    print(
        OUTPUT_DIR / "global_physical_pareto.csv"
    )
    print(
        OUTPUT_DIR / "global_physical_pareto_winwin.csv"
    )
    print(
        OUTPUT_DIR / "global_physical_pareto_origin_summary.csv"
    )
    print(
        OUTPUT_DIR / "global_physical_pareto_winwin_origin_summary.csv"
    )
    print(
        OUTPUT_DIR / "global_physical_pareto.png"
    )

    print("\n========================================")
    print("GLOBAL PHYSICAL PARETO ANALYSIS COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()
