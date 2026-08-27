# =========================================================
# OptiBF - LOCAL RANKING INVERSION ANALYSIS
# XGBoost vs RSM
# =========================================================
#
# Tests whether small local prediction errors produce
# candidate-order inversions near the physical Pareto front.
#
# For each pair of candidates within a local objective-space
# neighborhood, compare the physical ordering with XGBoost
# and RSM ordering.
#
# Outputs:
#   - pair_level_local_inversions.csv
#   - local_inversion_summary.csv
#   - pareto_neighborhood_inversions.csv
#
# =========================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml_rsm_comparison"
GLOBAL = DATA / "global_physical_pareto"
OUT = GLOBAL / "ranking_analysis"
OUT.mkdir(parents=True, exist_ok=True)

def sign_cmp(a, b, tol=1e-12):
    d = a - b
    if d > tol:
        return 1
    if d < -tol:
        return -1
    return 0

def pair_inversion(actual_a, actual_b, pred_a, pred_b):
    actual = sign_cmp(actual_a, actual_b)
    pred = sign_cmp(pred_a, pred_b)
    return actual != 0 and pred != 0 and actual != pred

def main():
    print("\n" + "=" * 70)
    print("OptiBF - LOCAL RANKING INVERSION ANALYSIS")
    print("=" * 70)

    physical = pd.read_csv(
        GLOBAL / "all_30000_physical_evaluated.csv"
    )
    common = pd.read_csv(
        DATA / "common_candidates_30000.csv"
    )

    df = common.merge(
        physical[
            ["candidate_id", "BF_physical", "Delta_BF_physical"]
        ],
        on="candidate_id",
        how="inner",
    )

    print(f"Candidates: {len(df)}")

    # Normalize objective space.
    bf_min, bf_max = df.BF_physical.min(), df.BF_physical.max()
    d_min, d_max = df.Delta_BF_physical.min(), df.Delta_BF_physical.max()

    df["BF_n"] = (df.BF_physical - bf_min) / max(bf_max - bf_min, 1e-12)
    df["D_n"] = (df.Delta_BF_physical - d_min) / max(d_max - d_min, 1e-12)

    # Distance to physical Pareto.
    pareto = pd.read_csv(GLOBAL / "global_physical_pareto.csv")
    p_bf = pareto.BF_physical.to_numpy()
    p_d = pareto.Delta_BF_physical.to_numpy()

    distances = []
    for _, r in df.iterrows():
        distances.append(
            np.sqrt(
                (r.BF_n - (p_bf-bf_min)/max(bf_max-bf_min,1e-12))**2
                +
                (r.D_n - (p_d-d_min)/max(d_max-d_min,1e-12))**2
            ).min()
        )
    df["distance_to_pareto"] = distances

    # Candidate neighborhoods.
    # 1%, 5%, 10% closest to physical Pareto.
    neighborhood_defs = {
        "closest_1pct": max(300, int(len(df)*0.01)),
        "closest_5pct": max(1500, int(len(df)*0.05)),
        "closest_10pct": max(3000, int(len(df)*0.10)),
    }

    # For each neighborhood, compare all pairs within a local
    # objective-space radius. To keep computation manageable,
    # sort by normalized BF and compare nearby ranks.
    rows = []

    for neighborhood, n in neighborhood_defs.items():
        sub = df.nsmallest(n, "distance_to_pareto").copy()
        sub = sub.sort_values("BF_n").reset_index(drop=True)

        # Compare a local window of 10 neighbors on each side.
        window = 10

        print(f"\nAnalyzing {neighborhood}: N={len(sub)}")

        for i in range(len(sub)):
            a = sub.iloc[i]

            for j in range(i+1, min(i+1+window, len(sub))):
                b = sub.iloc[j]

                # Physical ordering for each objective.
                inv_bf_xgb = pair_inversion(
                    a.BF_physical, b.BF_physical,
                    a.BF_XGB, b.BF_XGB
                )
                inv_bf_rsm = pair_inversion(
                    a.BF_physical, b.BF_physical,
                    a.BF_RSM, b.BF_RSM
                )

                inv_d_xgb = pair_inversion(
                    a.Delta_BF_physical, b.Delta_BF_physical,
                    a.Delta_BF_XGB, b.Delta_BF_XGB
                )
                inv_d_rsm = pair_inversion(
                    a.Delta_BF_physical, b.Delta_BF_physical,
                    a.Delta_BF_RSM, b.Delta_BF_RSM
                )

                rows.append({
                    "Neighborhood": neighborhood,
                    "candidate_A": int(a.candidate_id),
                    "candidate_B": int(b.candidate_id),
                    "BF_physical_A": a.BF_physical,
                    "BF_physical_B": b.BF_physical,
                    "Delta_physical_A": a.Delta_BF_physical,
                    "Delta_physical_B": b.Delta_BF_physical,
                    "BF_gap": abs(a.BF_physical-b.BF_physical),
                    "Delta_gap": abs(a.Delta_BF_physical-b.Delta_BF_physical),
                    "BF_inversion_XGB": inv_bf_xgb,
                    "BF_inversion_RSM": inv_bf_rsm,
                    "Delta_inversion_XGB": inv_d_xgb,
                    "Delta_inversion_RSM": inv_d_rsm,
                })

    pairs = pd.DataFrame(rows)

    # Summary by neighborhood.
    summaries = []

    for neighborhood, g in pairs.groupby("Neighborhood"):

        total = len(g)

        for method in ["XGB", "RSM"]:
            bf_col = f"BF_inversion_{method}"
            d_col = f"Delta_inversion_{method}"

            bf_inv = int(g[bf_col].sum())
            d_inv = int(g[d_col].sum())

            summaries.append({
                "Neighborhood": neighborhood,
                "Method": "XGBoost" if method == "XGB" else "RSM",
                "Pairs": total,
                "BF_inversions": bf_inv,
                "BF_inversion_rate_percent": 100*bf_inv/total,
                "Delta_BF_inversions": d_inv,
                "Delta_BF_inversion_rate_percent": 100*d_inv/total,
                "Any_objective_inversion": int(
                    (g[bf_col] | g[d_col]).sum()
                ),
                "Any_inversion_rate_percent": 100*int(
                    (g[bf_col] | g[d_col]).sum()
                )/total,
            })

    summary = pd.DataFrame(summaries)

    # Focus only on pairs where at least one candidate is a physical
    # Pareto point.
    pareto_ids = set(pareto.candidate_id.astype(int))
    pairs["Pareto_involved"] = (
        pairs.candidate_A.astype(int).isin(pareto_ids)
        | pairs.candidate_B.astype(int).isin(pareto_ids)
    )

    pareto_pairs = pairs[pairs.Pareto_involved].copy()

    pareto_summary = []
    for neighborhood, g in pareto_pairs.groupby("Neighborhood"):
        total = len(g)
        if total == 0:
            continue

        for method in ["XGB", "RSM"]:
            bf_col = f"BF_inversion_{method}"
            d_col = f"Delta_inversion_{method}"
            any_col = g[bf_col] | g[d_col]

            pareto_summary.append({
                "Neighborhood": neighborhood,
                "Method": "XGBoost" if method == "XGB" else "RSM",
                "Pairs": total,
                "BF_inversions": int(g[bf_col].sum()),
                "BF_inversion_rate_percent": 100*g[bf_col].mean(),
                "Delta_BF_inversions": int(g[d_col].sum()),
                "Delta_BF_inversion_rate_percent": 100*g[d_col].mean(),
                "Any_inversion_rate_percent": 100*any_col.mean(),
            })

    pareto_summary = pd.DataFrame(pareto_summary)

    pairs.to_csv(
        OUT / "pair_level_local_inversions.csv",
        index=False,
    )

    summary.to_csv(
        OUT / "local_inversion_summary.csv",
        index=False,
    )

    pareto_summary.to_csv(
        OUT / "pareto_neighborhood_inversions.csv",
        index=False,
    )

    print("\n========================================")
    print("ALL LOCAL PAIRS")
    print("========================================")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n========================================")
    print("PAIRS INVOLVING PHYSICAL PARETO POINTS")
    print("========================================")

    if len(pareto_summary):
        print(
            pareto_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
    else:
        print("No Pareto-involved pairs found.")

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(OUT / "pair_level_local_inversions.csv")
    print(OUT / "local_inversion_summary.csv")
    print(OUT / "pareto_neighborhood_inversions.csv")

    print("\n========================================")
    print("COMPLETED")
    print("========================================")

if __name__ == "__main__":
    main()
