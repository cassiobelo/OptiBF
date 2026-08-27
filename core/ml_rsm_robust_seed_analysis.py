# =========================================================
# OptiBF - ROBUST ML vs RSM GLOBAL PARETO COMPARISON
# Multiple candidate-space seeds
# =========================================================
#
# IMPORTANT:
# This version reproduces the ORIGINAL fair-comparison setup.
#
# Reference experiment:
#   seed = 42
#   30,000 candidates
#   BOUNDS and XGB hyperparameters copied from
#   ml_rsm_multiobjective_fair_comparison.py
#
# Seeds:
#   42, 43, 44, 45, 46
#
# For seed 42, the script explicitly loads the original
# common_candidates_30000.csv so that the baseline is exactly
# the same candidate space used in the original analysis.
#
# For seeds 43-46, the same candidate-generation function
# and the same restricted design space are used.
#
# =========================================================

from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Import the SAME physical model used by the project.
from core.brake_model import RD, calculate_bf, calculate_bf_asymmetry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = (
    DATA_DIR
    / "ml_rsm_comparison"
    / "seed_robustness"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# EXPERIMENT SETTINGS — COPIED FROM ORIGINAL FAIR COMPARISON
# =========================================================

SEEDS = [42, 43, 44, 45, 46]

N_CANDIDATES = 30000

RANDOM_SEED = 42

MU = 0.30
ASYMMETRY_DELTA = 0.10

# EXACT original restricted design space.
BOUNDS = {
    "L1_mm": (35.0, 45.0),
    "L3_mm": (138.15, 143.15),
    "L4_mm": (152.0, 162.0),
    "theta1_deg": (18.0, 38.0),
    "theta2_deg": (135.0, 155.0),
}

FEATURES = [
    "L1_mm",
    "L3_mm",
    "L4_mm",
    "theta1_deg",
    "theta2_deg",
    "mu",
]

REFERENCE_BF = 1.55877649
REFERENCE_DELTA = 28.12134928


# EXACT original XGBoost tuning.
XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.0609418819,
    "n_estimators": 800,
    "subsample": 0.6030189277,
    "colsample_bytree": 0.9852181122,
}


BF_DATASET = (
    DATA_DIR / "optibf_dataset_lhs_1000.csv"
)

DELTA_DATASET = (
    DATA_DIR
    / "optibf_dataset_lhs_1000_robustness.csv"
)

ORIGINAL_COMMON = (
    DATA_DIR
    / "ml_rsm_comparison"
    / "common_candidates_30000.csv"
)


# =========================================================
# MODEL TRAINING
# =========================================================

def train_xgb(df, target):

    params = {
        **XGB_PARAMS,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }

    model = XGBRegressor(**params)

    model.fit(
        df[FEATURES],
        df[target],
    )

    return model


def train_rsm(df, target):

    poly = PolynomialFeatures(
        degree=2,
        include_bias=False,
    )

    X_poly = poly.fit_transform(
        df[FEATURES]
    )

    model = LinearRegression()

    model.fit(
        X_poly,
        df[target],
    )

    return poly, model


# =========================================================
# CANDIDATE GENERATION
# =========================================================

def generate_candidates(seed):

    rng = np.random.default_rng(seed)

    data = {}

    for name, (lo, hi) in BOUNDS.items():

        data[name] = rng.uniform(
            lo,
            hi,
            N_CANDIDATES,
        )

    data["mu"] = np.full(
        N_CANDIDATES,
        MU,
    )

    df = pd.DataFrame(data)

    df.insert(
        0,
        "candidate_id",
        np.arange(len(df)),
    )

    return df


# =========================================================
# PHYSICAL MODEL
# =========================================================

def evaluate_physical(df):

    bf_values = []
    delta_values = []

    total = len(df)

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        bf = calculate_bf(
            L1=row["L1_mm"] / 1000.0,
            L3=row["L3_mm"] / 1000.0,
            L4=row["L4_mm"] / 1000.0,
            theta1=math.radians(
                row["theta1_deg"]
            ),
            theta2=math.radians(
                row["theta2_deg"]
            ),
            mu=MU,
            RD=RD,
        )

        delta = calculate_bf_asymmetry(
            L1=row["L1_mm"] / 1000.0,
            L3=row["L3_mm"] / 1000.0,
            L4=row["L4_mm"] / 1000.0,
            theta1=math.radians(
                row["theta1_deg"]
            ),
            theta2=math.radians(
                row["theta2_deg"]
            ),
            mu=MU,
            delta=ASYMMETRY_DELTA,
            RD=RD,
        )

        bf_values.append(
            bf["BF"]
        )

        delta_values.append(
            delta["delta_BF_percent"]
        )

        if i % 5000 == 0:

            print(
                f"      Physical: "
                f"{i}/{total}"
            )

    result = df.copy()

    result["BF_physical"] = bf_values

    result["Delta_BF_physical"] = (
        delta_values
    )

    return result


# =========================================================
# PARETO
# =========================================================

def pareto_mask(
    bf,
    delta,
):

    values = np.column_stack(
        [
            bf,
            delta,
        ]
    )

    n = len(values)

    keep = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):

        dominated = (
            (values[:, 0] >= values[i, 0])
            &
            (values[:, 1] <= values[i, 1])
            &
            (
                (values[:, 0] > values[i, 0])
                |
                (values[:, 1] < values[i, 1])
            )
        )

        dominated[i] = False

        if np.any(dominated):

            keep[i] = False

    return keep


# =========================================================
# CLASSIFICATION METRICS
# =========================================================

def classification_metrics(
    actual,
    predicted,
):

    tp = int(
        np.sum(
            actual & predicted
        )
    )

    fp = int(
        np.sum(
            ~actual & predicted
        )
    )

    fn = int(
        np.sum(
            actual & ~predicted
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
    )


# =========================================================
# ONE SEED
# =========================================================

def analyze_seed(
    seed,
    xgb_bf,
    xgb_delta,
    rsm_bf,
    rsm_delta,
):

    print("\n" + "-" * 60)

    print(
        f"SEED {seed}"
    )

    print("-" * 60)

    # -----------------------------------------------------
    # Candidate space
    # -----------------------------------------------------

    if (
        seed == 42
        and ORIGINAL_COMMON.exists()
    ):

        print(
            "Using ORIGINAL "
            "common_candidates_30000.csv"
        )

        candidates = pd.read_csv(
            ORIGINAL_COMMON
        )

    else:

        candidates = (
            generate_candidates(seed)
        )

    # -----------------------------------------------------
    # Surrogate predictions
    # -----------------------------------------------------

    X = candidates[FEATURES]

    candidates["BF_XGB"] = (
        xgb_bf.predict(X)
    )

    candidates["Delta_BF_XGB"] = (
        xgb_delta.predict(X)
    )

    rsm_bf_poly, rsm_bf_model = rsm_bf

    rsm_delta_poly, rsm_delta_model = (
        rsm_delta
    )

    candidates["BF_RSM"] = (
        rsm_bf_model.predict(
            rsm_bf_poly.transform(X)
        )
    )

    candidates["Delta_BF_RSM"] = (
        rsm_delta_model.predict(
            rsm_delta_poly.transform(X)
        )
    )

    # -----------------------------------------------------
    # Physical evaluation
    # -----------------------------------------------------

    candidates = evaluate_physical(
        candidates
    )

    # -----------------------------------------------------
    # Pareto fronts
    # -----------------------------------------------------

    physical_p = pareto_mask(
        candidates["BF_physical"].to_numpy(),
        candidates["Delta_BF_physical"].to_numpy(),
    )

    xgb_p = pareto_mask(
        candidates["BF_XGB"].to_numpy(),
        candidates["Delta_BF_XGB"].to_numpy(),
    )

    rsm_p = pareto_mask(
        candidates["BF_RSM"].to_numpy(),
        candidates["Delta_BF_RSM"].to_numpy(),
    )

    # -----------------------------------------------------
    # Win-Win
    # -----------------------------------------------------

    physical_win = (
        (candidates["BF_physical"] > REFERENCE_BF)
        &
        (
            candidates["Delta_BF_physical"]
            < REFERENCE_DELTA
        )
    ).to_numpy()

    xgb_win = (
        (candidates["BF_XGB"] > REFERENCE_BF)
        &
        (
            candidates["Delta_BF_XGB"]
            < REFERENCE_DELTA
        )
    ).to_numpy()

    rsm_win = (
        (candidates["BF_RSM"] > REFERENCE_BF)
        &
        (
            candidates["Delta_BF_RSM"]
            < REFERENCE_DELTA
        )
    ).to_numpy()

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    xgb_p_metrics = classification_metrics(
        physical_p,
        xgb_p,
    )

    rsm_p_metrics = classification_metrics(
        physical_p,
        rsm_p,
    )

    xgb_w_metrics = classification_metrics(
        physical_win,
        xgb_win,
    )

    rsm_w_metrics = classification_metrics(
        physical_win,
        rsm_win,
    )

    rows = []

    for method, p, w in [
        (
            "XGBoost",
            xgb_p_metrics,
            xgb_w_metrics,
        ),
        (
            "RSM",
            rsm_p_metrics,
            rsm_w_metrics,
        ),
    ]:

        rows.append(
            {
                "Seed": seed,
                "Method": method,
                "N_candidates": len(candidates),

                "Physical_Pareto":
                    int(physical_p.sum()),

                "Surrogate_Pareto":
                    int(
                        xgb_p.sum()
                        if method == "XGBoost"
                        else rsm_p.sum()
                    ),

                "Pareto_TP": p[0],
                "Pareto_FP": p[1],
                "Pareto_FN": p[2],

                "Pareto_Precision": p[3],
                "Pareto_Recall": p[4],
                "Pareto_F1": p[5],

                "Physical_WinWin":
                    int(physical_win.sum()),

                "Surrogate_WinWin":
                    int(
                        xgb_win.sum()
                        if method == "XGBoost"
                        else rsm_win.sum()
                    ),

                "WinWin_TP": w[0],
                "WinWin_FP": w[1],
                "WinWin_FN": w[2],

                "WinWin_Precision": w[3],
                "WinWin_Recall": w[4],
                "WinWin_F1": w[5],
            }
        )

    # -----------------------------------------------------
    # Save physical Pareto for each seed
    # -----------------------------------------------------

    candidates["Physical_Pareto"] = (
        physical_p
    )

    candidates["XGB_Pareto"] = (
        xgb_p
    )

    candidates["RSM_Pareto"] = (
        rsm_p
    )

    candidates["Physical_WinWin"] = (
        physical_win
    )

    candidates["XGB_WinWin"] = (
        xgb_win
    )

    candidates["RSM_WinWin"] = (
        rsm_win
    )

    candidates[
        candidates["Physical_Pareto"]
    ].to_csv(
        OUT_DIR
        / f"seed_{seed}_physical_pareto.csv",
        index=False,
    )

    print(
        f"  Physical Pareto: "
        f"{physical_p.sum()}"
    )

    print(
        f"  XGB recall: "
        f"{xgb_p_metrics[4]:.4f} "
        f"({xgb_p_metrics[0]}/"
        f"{physical_p.sum()})"
    )

    print(
        f"  RSM recall: "
        f"{rsm_p_metrics[4]:.4f} "
        f"({rsm_p_metrics[0]}/"
        f"{physical_p.sum()})"
    )

    return rows


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)

    print(
        "OptiBF - ROBUST ML vs RSM "
        "GLOBAL PARETO"
    )

    print("=" * 70)

    print(
        "\nTraining models once from "
        "the 1000-point datasets..."
    )

    bf_df = pd.read_csv(
        BF_DATASET
    )

    delta_df = pd.read_csv(
        DELTA_DATASET
    )

    xgb_bf = train_xgb(
        bf_df,
        "BF",
    )

    xgb_delta = train_xgb(
        delta_df,
        "Delta_BF_percent",
    )

    rsm_bf = train_rsm(
        bf_df,
        "BF",
    )

    rsm_delta = train_rsm(
        delta_df,
        "Delta_BF_percent",
    )

    print(
        "Models trained."
    )

    all_rows = []

    for seed in SEEDS:

        all_rows.extend(
            analyze_seed(
                seed,
                xgb_bf,
                xgb_delta,
                rsm_bf,
                rsm_delta,
            )
        )

    results = pd.DataFrame(
        all_rows
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary_rows = []

    for method in [
        "XGBoost",
        "RSM",
    ]:

        sub = results[
            results["Method"] == method
        ]

        row = {
            "Method": method
        }

        metrics = [
            "Physical_Pareto",
            "Surrogate_Pareto",
            "Pareto_TP",
            "Pareto_Precision",
            "Pareto_Recall",
            "Pareto_F1",
            "Physical_WinWin",
            "Surrogate_WinWin",
            "WinWin_TP",
            "WinWin_Precision",
            "WinWin_Recall",
            "WinWin_F1",
        ]

        for metric in metrics:

            row[
                metric + "_mean"
            ] = sub[metric].mean()

            row[
                metric + "_std"
            ] = sub[metric].std(
                ddof=1
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    results.to_csv(
        OUT_DIR
        / "robust_seed_results.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR
        / "robust_seed_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS BY SEED")
    print("=" * 70)

    print(
        results[
            [
                "Seed",
                "Method",
                "Physical_Pareto",
                "Surrogate_Pareto",
                "Pareto_TP",
                "Pareto_Recall",
                "Pareto_Precision",
                "Pareto_F1",
                "Physical_WinWin",
                "Surrogate_WinWin",
                "WinWin_TP",
                "WinWin_Recall",
                "WinWin_Precision",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("MEAN ± STD OVER 5 SEEDS")
    print("=" * 70)

    for _, row in summary.iterrows():

        print(
            f"\n{row['Method']}"
        )

        for metric in [
            "Physical_Pareto",
            "Surrogate_Pareto",
            "Pareto_TP",
            "Pareto_Recall",
            "Pareto_Precision",
            "Pareto_F1",
            "Physical_WinWin",
            "Surrogate_WinWin",
            "WinWin_TP",
            "WinWin_Recall",
            "WinWin_Precision",
            "WinWin_F1",
        ]:

            print(
                f"  {metric:24s}: "
                f"{row[metric+'_mean']:.4f} ± "
                f"{row[metric+'_std']:.4f}"
            )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        OUT_DIR
        / "robust_seed_results.csv"
    )

    print(
        OUT_DIR
        / "robust_seed_summary.csv"
    )

    print("\n" + "=" * 70)
    print(
        "ROBUST SEED ANALYSIS COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()