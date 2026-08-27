# OptiBF - One-at-a-Time (OAT) Parameter Trend Analysis
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brake_model import calculate_bf, calculate_bf_asymmetry, RD

RESULTS_DIR = PROJECT_ROOT / "sensitivity_results"
TREND_DIR = RESULTS_DIR / "parameter_trends"
CSV_DIR = TREND_DIR / "csv"
PLOT_DIR = TREND_DIR / "plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_MU = 0.30
ROBUSTNESS_DELTA = 0.10
N_POINTS = 101

PARAMETERS = {
    "L1_mm": (35.0, 45.0, "L1 (mm)"),
    "L3_mm": (138.15, 143.15, "L3 (mm)"),
    "L4_mm": (152.0, 162.0, "L4 (mm)"),
    "theta1_deg": (18.0, 38.0, r"$\theta_1$ (deg)"),
    "theta2_deg": (135.0, 155.0, r"$\theta_2$ (deg)"),
}

CENTRAL_VALUES = {k: (v[0] + v[1]) / 2 for k, v in PARAMETERS.items()}

def evaluate(geometry):
    L1 = geometry["L1_mm"] / 1000
    L3 = geometry["L3_mm"] / 1000
    L4 = geometry["L4_mm"] / 1000
    t1 = np.deg2rad(geometry["theta1_deg"])
    t2 = np.deg2rad(geometry["theta2_deg"])

    r1 = calculate_bf(L1=L1, L3=L3, L4=L4, theta1=t1,
                      theta2=t2, mu=FIXED_MU, RD=RD)
    r2 = calculate_bf_asymmetry(L1=L1, L3=L3, L4=L4, theta1=t1,
                                theta2=t2, mu=FIXED_MU,
                                delta=ROBUSTNESS_DELTA, RD=RD)
    bf = np.nan if r1 is None else r1["BF"]
    dbf = np.nan if r2 is None else r2["delta_BF_percent"]
    return bf, dbf

def trend_label(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return "Insufficient data"
    d = np.diff(y)
    tol = max(np.max(np.abs(y)) * 1e-10, 1e-10)
    if np.all(d >= -tol) and np.any(d > tol):
        return "Monotonic increasing"
    if np.all(d <= tol) and np.any(d < -tol):
        return "Monotonic decreasing"
    if np.all(np.abs(d) <= tol):
        return "Approximately constant"
    return "Non-monotonic"

def main():
    print("=" * 60)
    print("OptiBF - One-at-a-Time Parameter Trend Analysis")
    print("=" * 60)
    print(f"\nProject directory:\n{PROJECT_ROOT}")
    print(f"\nOutput directory:\n{TREND_DIR}")
    print(f"\nCSV directory:\n{CSV_DIR}")
    print(f"\nPlot directory:\n{PLOT_DIR}")
    print(f"\nBrake drum radius RD: {RD * 1000:.2f} mm")
    print(f"Fixed friction coefficient: mu = {FIXED_MU:.2f}")
    print(f"Friction asymmetry: {ROBUSTNESS_DELTA * 100:.1f}%")
    print("\nCentral geometry:")
    for k, v in CENTRAL_VALUES.items():
        print(f"  {k}: {v}")

    bf_rows, dbf_rows, summary = [], [], []

    for name, (vmin, vmax, xlabel) in PARAMETERS.items():
        values = np.linspace(vmin, vmax, N_POINTS)
        bfs, dbfs = [], []

        for value in values:
            g = CENTRAL_VALUES.copy()
            g[name] = value
            bf, dbf = evaluate(g)
            bfs.append(bf)
            dbfs.append(dbf)
            bf_rows.append({"Parameter": name, "Parameter_Value": value, "BF": bf})
            dbf_rows.append({"Parameter": name, "Parameter_Value": value,
                             "Delta_BF_percent": dbf})

        x = np.asarray(values)
        y1 = np.asarray(bfs, dtype=float)
        y2 = np.asarray(dbfs, dtype=float)

        def slope_corr(y):
            mask = np.isfinite(y)
            if mask.sum() < 2:
                return np.nan, np.nan
            slope = np.polyfit(x[mask], y[mask], 1)[0]
            corr = np.corrcoef(x[mask], y[mask])[0, 1]
            return slope, corr

        s1, c1 = slope_corr(y1)
        s2, c2 = slope_corr(y2)

        summary.append({
            "Parameter": name, "Min": vmin, "Max": vmax,
            "Central": CENTRAL_VALUES[name],
            "BF_at_min": y1[0], "BF_at_max": y1[-1],
            "BF_change": y1[-1] - y1[0],
            "BF_slope": s1, "BF_correlation": c1,
            "BF_trend": trend_label(y1),
            "DeltaBF_at_min": y2[0], "DeltaBF_at_max": y2[-1],
            "DeltaBF_change": y2[-1] - y2[0],
            "DeltaBF_slope": s2, "DeltaBF_correlation": c2,
            "DeltaBF_trend": trend_label(y2)
        })

        for data, y, ylabel, prefix in [
            (bf_rows, y1, "Brake Factor (BF)", "BF"),
            (dbf_rows, y2, "Delta BF (%)", "DeltaBF")
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(values, y, linewidth=2)
            ax.axvline(CENTRAL_VALUES[name], linestyle="--", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs {xlabel}\n"
                         f"mu = {FIXED_MU:.2f} | Other geometry parameters fixed at central values")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            path = PLOT_DIR / f"{prefix}_vs_{name}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] PNG: {path}")

    df_bf = pd.DataFrame(bf_rows)
    df_dbf = pd.DataFrame(dbf_rows)
    df_summary = pd.DataFrame(summary)

    paths = [
        (df_bf, CSV_DIR / "parameter_trends_bf.csv"),
        (df_dbf, CSV_DIR / "parameter_trends_delta_bf.csv"),
        (df_summary, CSV_DIR / "parameter_trends_summary.csv")
    ]
    for df, path in paths:
        df.to_csv(path, index=False)
        print(f"[OK] CSV: {path}")

    print("\n" + "=" * 60)
    print("TREND SUMMARY")
    print("=" * 60)
    print(df_summary[["Parameter", "BF_change", "BF_trend",
                      "DeltaBF_change", "DeltaBF_trend"]].to_string(index=False))
    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 60)
        print("EXECUTION ERROR")
        print("=" * 60)
        print(f"\nType: {type(error).__name__}")
        print(f"\nMessage:\n{error}")
        print("\nFull traceback:")
        traceback.print_exc()
    finally:
        input("\n\nPress ENTER to close...")
