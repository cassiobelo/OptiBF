import math
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from core.brake_model import RD, calculate_bf, calculate_bf_asymmetry


# =========================================================
# OptiBF - FAIR ML vs RSM MULTI-OBJECTIVE COMPARISON
# =========================================================
#
# Methodology:
# 1. Train XGBoost and RSM independently.
# 2. Generate ONE common set of 30,000 candidates (seed=42).
# 3. Each method predicts BF and Delta BF on the same candidates.
# 4. Build a separate Pareto front for each surrogate.
# 5. Define the same WIN-WIN region for both methods:
#       BF_pred > BF_reference
#       DeltaBF_pred < DeltaBF_reference
# 6. Among WIN-WIN Pareto candidates, calculate the same
#    normalized compromise score:
#
#       G_BF = (BF - BF_ref) / (BF_max - BF_ref)
#       G_D  = (Delta_ref - Delta) /
#              (Delta_ref - Delta_min)
#
#       Score = (G_BF + G_D) / 2
#
#    The candidate with maximum Score is the Best Win-Win.
# 7. Only AFTER selection, validate the selected solutions
#    with the physical model.
#
# Important:
# - CL/CT is reported only.
# - It is NOT a constraint.
# - The physical model is NOT used to select the surrogate
#   optimum; it is used to validate the selected solution.
# =========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "ml_rsm_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_CANDIDATES = 30000
MU = 0.30
ASYMMETRY_DELTA = 0.10

BOUNDS = {
    "L1_mm": (35.0, 45.0),
    "L3_mm": (138.15, 143.15),
    "L4_mm": (152.0, 162.0),
    "theta1_deg": (18.0, 38.0),
    "theta2_deg": (135.0, 155.0),
}

FEATURES = [
    "L1_mm", "L3_mm", "L4_mm",
    "theta1_deg", "theta2_deg", "mu"
]

REFERENCE = {
    "L1_mm": 40.0,
    "L3_mm": 143.15,
    "L4_mm": 157.0,
    "theta1_deg": 28.0,
    "theta2_deg": 145.0,
    "mu": MU,
}

BF_DATASET = DATA_DIR / "optibf_dataset_lhs_1000.csv"
DELTA_DATASET = DATA_DIR / "optibf_dataset_lhs_1000_robustness.csv"

XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.0609418819,
    "n_estimators": 800,
    "subsample": 0.6030189277,
    "colsample_bytree": 0.9852181122,
}


def generate_common_candidates():
    rng = np.random.default_rng(RANDOM_SEED)
    data = {}

    for name, (lo, hi) in BOUNDS.items():
        data[name] = rng.uniform(lo, hi, N_CANDIDATES)

    data["mu"] = np.full(N_CANDIDATES, MU)

    df = pd.DataFrame(data)
    df.insert(0, "candidate_id", np.arange(len(df)))
    return df


def train_xgb(df, target):
    model = XGBRegressor(
        **XGB_PARAMS,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(df[FEATURES], df[target])
    return model


def train_rsm(df, target):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df[FEATURES])

    model = LinearRegression()
    model.fit(X_poly, df[target])

    return poly, model


def rsm_predict(poly, model, X):
    return model.predict(poly.transform(X))


def pareto_filter(df, bf_col, delta_col):
    values = df[[bf_col, delta_col]].to_numpy()
    n = len(values)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        bf_i, delta_i = values[i]

        dominates = (
            (values[:, 0] >= bf_i)
            & (values[:, 1] <= delta_i)
            & (
                (values[:, 0] > bf_i)
                | (values[:, 1] < delta_i)
            )
        )
        dominates[i] = False

        if np.any(dominates):
            is_pareto[i] = False

    return df.loc[is_pareto].copy().reset_index(drop=True)


def physical_values(row):
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

    return {
        "BF_physical": bf["BF"],
        "Delta_BF_physical": delta["delta_BF_percent"],
        "CL_physical": bf["CL"],
        "CT_physical": bf["CT"],
        "CL_CT_physical": bf["CL_CT"],
    }


def reference_physical():
    return physical_values(pd.Series(REFERENCE))


def select_best_win_win(pareto, bf_col, delta_col, reference):
    """
    Same criterion for XGBoost and RSM.

    First filter the surrogate Pareto front to the region
    that improves BOTH objectives relative to the physical
    reference.

    Then normalize the gains using the range available
    within the WIN-WIN subset and maximize the equal-weight
    compromise score.
    """
    win = pareto[
        (pareto[bf_col] > reference["BF"])
        & (pareto[delta_col] < reference["Delta_BF"])
    ].copy()

    if len(win) == 0:
        return None, win

    bf_gain_den = win[bf_col].max() - reference["BF"]
    delta_gain_den = reference["Delta_BF"] - win[delta_col].min()

    if bf_gain_den <= 0 or delta_gain_den <= 0:
        return None, win

    win["BF_gain_norm"] = (
        (win[bf_col] - reference["BF"]) / bf_gain_den
    )

    win["Delta_gain_norm"] = (
        (reference["Delta_BF"] - win[delta_col])
        / delta_gain_den
    )

    win["Compromise_Score"] = (
        win["BF_gain_norm"] + win["Delta_gain_norm"]
    ) / 2.0

    best_idx = win["Compromise_Score"].idxmax()
    best = win.loc[best_idx].copy()

    return best, win


def add_physical_validation(row):
    result = physical_values(row)

    for key, value in result.items():
        row[key] = value

    row["BF_error_percent"] = (
        abs(row["BF_pred"] - row["BF_physical"])
        / abs(row["BF_physical"])
    ) * 100.0

    row["Delta_BF_error_percent"] = (
        abs(row["Delta_BF_pred"] - row["Delta_BF_physical"])
        / abs(row["Delta_BF_physical"])
    ) * 100.0

    row["BF_change_percent"] = (
        (row["BF_physical"] - REF["BF"])
        / REF["BF"]
    ) * 100.0

    row["Delta_BF_change_pp"] = (
        row["Delta_BF_physical"] - REF["Delta_BF"]
    )

    return row


def physical_pareto(df):
    return pareto_filter(
        df,
        "BF_physical",
        "Delta_BF_physical",
    )


def main():
    print("\n========================================")
    print("OptiBF - FAIR ML vs RSM COMPARISON")
    print("========================================")

    print("\nLoading datasets...")
    bf_df = pd.read_csv(BF_DATASET)
    delta_df = pd.read_csv(DELTA_DATASET)

    print(f"BF samples: {len(bf_df)}")
    print(f"Delta BF samples: {len(delta_df)}")

    # -----------------------------------------------------
    # REFERENCE
    # -----------------------------------------------------
    global REF
    REF_PHYSICAL = reference_physical()

    REF = {
        "BF": REF_PHYSICAL["BF_physical"],
        "Delta_BF": REF_PHYSICAL["Delta_BF_physical"],
        "CL_CT": REF_PHYSICAL["CL_CT_physical"],
    }

    print("\nREFERENCE PHYSICAL MODEL")
    print(f"BF       = {REF['BF']:.8f}")
    print(f"Delta BF = {REF['Delta_BF']:.8f}%")
    print(f"CL/CT    = {REF['CL_CT']:.8f}")

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    print("\nTraining XGBoost...")
    xgb_bf = train_xgb(bf_df, "BF")
    xgb_delta = train_xgb(delta_df, "Delta_BF_percent")

    print("XGBoost models trained.")

    print("\nTraining RSM...")
    rsm_bf_poly, rsm_bf = train_rsm(bf_df, "BF")
    rsm_delta_poly, rsm_delta = train_rsm(
        delta_df, "Delta_BF_percent"
    )

    print("RSM models trained.")

    # -----------------------------------------------------
    # COMMON CANDIDATES
    # -----------------------------------------------------
    candidates = generate_common_candidates()

    print("\n========================================")
    print("COMMON CANDIDATE SPACE")
    print("========================================")
    print(f"Candidates: {len(candidates)}")
    print(f"Seed: {RANDOM_SEED}")
    print("Same candidates used by XGBoost and RSM.")

    X = candidates[FEATURES]

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------
    candidates["BF_XGB"] = xgb_bf.predict(X)
    candidates["Delta_BF_XGB"] = xgb_delta.predict(X)

    candidates["BF_RSM"] = rsm_predict(
        rsm_bf_poly, rsm_bf, X
    )
    candidates["Delta_BF_RSM"] = rsm_predict(
        rsm_delta_poly, rsm_delta, X
    )

    # -----------------------------------------------------
    # PARETO FRONTS
    # -----------------------------------------------------
    pareto_xgb = pareto_filter(
        candidates,
        "BF_XGB",
        "Delta_BF_XGB",
    )

    pareto_rsm = pareto_filter(
        candidates,
        "BF_RSM",
        "Delta_BF_RSM",
    )

    print("\n========================================")
    print("PARETO FRONTS")
    print("========================================")
    print(f"XGBoost Pareto: {len(pareto_xgb)}")
    print(f"RSM Pareto:     {len(pareto_rsm)}")

    # -----------------------------------------------------
    # WIN-WIN SELECTION
    # -----------------------------------------------------
    best_xgb, win_xgb = select_best_win_win(
        pareto_xgb,
        "BF_XGB",
        "Delta_BF_XGB",
        REF,
    )

    best_rsm, win_rsm = select_best_win_win(
        pareto_rsm,
        "BF_RSM",
        "Delta_BF_RSM",
        REF,
    )

    print("\n========================================")
    print("WIN-WIN ANALYSIS")
    print("========================================")
    print(f"XGBoost win-win Pareto points: {len(win_xgb)}")
    print(f"RSM win-win Pareto points:     {len(win_rsm)}")

    # -----------------------------------------------------
    # PHYSICAL VALIDATION OF SELECTED POINTS
    # -----------------------------------------------------
    selected_rows = []

    for method, best, bf_col, delta_col in [
        ("XGBoost", best_xgb, "BF_XGB", "Delta_BF_XGB"),
        ("RSM", best_rsm, "BF_RSM", "Delta_BF_RSM"),
    ]:
        if best is None:
            print(f"\nNo win-win solution found for {method}.")
            continue

        row = best.copy()
        row["Method"] = method
        row["BF_pred"] = row[bf_col]
        row["Delta_BF_pred"] = row[delta_col]

        row = add_physical_validation(row)
        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)

    # -----------------------------------------------------
    # PHYSICAL PARETO DIAGNOSTIC
    # -----------------------------------------------------
    physical_sets = []

    for method, pareto in [
        ("XGBoost", pareto_xgb),
        ("RSM", pareto_rsm),
    ]:
        rows = []
        for _, row in pareto.iterrows():
            vals = physical_values(row)
            r = row.copy()
            r["Method"] = method
            for k, v in vals.items():
                r[k] = v
            rows.append(r)

        physical_df = pd.DataFrame(rows)

        if len(physical_df):
            physical_df = physical_pareto(physical_df)

        physical_sets.append(physical_df)

    physical_pareto_all = pd.concat(
        physical_sets, ignore_index=True
    )

    # -----------------------------------------------------
    # CONVERGENCE CHECK
    # -----------------------------------------------------
    convergence = pd.DataFrame()

    if len(selected) == 2:
        x = selected[selected["Method"] == "XGBoost"].iloc[0]
        r = selected[selected["Method"] == "RSM"].iloc[0]

        params = [
            "L1_mm", "L3_mm", "L4_mm",
            "theta1_deg", "theta2_deg"
        ]

        convergence = pd.DataFrame({
            "Parameter": params,
            "XGBoost": [x[p] for p in params],
            "RSM": [r[p] for p in params],
        })

        convergence["Absolute_Difference"] = (
            convergence["XGBoost"] - convergence["RSM"]
        ).abs()

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    candidates.to_csv(
        OUTPUT_DIR / "common_candidates_30000.csv",
        index=False,
    )

    pareto_xgb.to_csv(
        OUTPUT_DIR / "xgb_pareto.csv",
        index=False,
    )

    pareto_rsm.to_csv(
        OUTPUT_DIR / "rsm_pareto.csv",
        index=False,
    )

    win_xgb.to_csv(
        OUTPUT_DIR / "xgb_winwin.csv",
        index=False,
    )

    win_rsm.to_csv(
        OUTPUT_DIR / "rsm_winwin.csv",
        index=False,
    )

    selected.to_csv(
        OUTPUT_DIR / "selected_best_winwin.csv",
        index=False,
    )

    physical_pareto_all.to_csv(
        OUTPUT_DIR / "physical_pareto_diagnostic.csv",
        index=False,
    )

    convergence.to_csv(
        OUTPUT_DIR / "convergence_check.csv",
        index=False,
    )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------
    print("\n========================================")
    print("BEST WIN-WIN RESULTS")
    print("========================================")

    if len(selected):
        cols = [
            "Method",
            "candidate_id",
            "L1_mm", "L3_mm", "L4_mm",
            "theta1_deg", "theta2_deg",
            "BF_pred", "Delta_BF_pred",
            "Compromise_Score",
            "BF_physical", "Delta_BF_physical",
            "CL_CT_physical",
            "BF_change_percent",
            "Delta_BF_change_pp",
            "BF_error_percent",
            "Delta_BF_error_percent",
        ]

        print(
            selected[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.8f}"
            )
        )

    if len(convergence):
        print("\n========================================")
        print("ML vs RSM CONVERGENCE CHECK")
        print("========================================")
        print(
            convergence.to_string(
                index=False,
                float_format=lambda x: f"{x:.10f}"
            )
        )

        max_diff = convergence["Absolute_Difference"].max()
        print(f"\nMaximum parameter difference: {max_diff:.12f}")

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")
    print(OUTPUT_DIR / "common_candidates_30000.csv")
    print(OUTPUT_DIR / "xgb_pareto.csv")
    print(OUTPUT_DIR / "rsm_pareto.csv")
    print(OUTPUT_DIR / "xgb_winwin.csv")
    print(OUTPUT_DIR / "rsm_winwin.csv")
    print(OUTPUT_DIR / "selected_best_winwin.csv")
    print(OUTPUT_DIR / "physical_pareto_diagnostic.csv")
    print(OUTPUT_DIR / "convergence_check.csv")

    print("\n========================================")
    print("FAIR ML vs RSM COMPARISON COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()