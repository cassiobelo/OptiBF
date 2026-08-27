import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = (
    "data/optibf_dataset_lhs_1000_robustness.csv"
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    df = pd.read_csv(DATA_FILE)

    print("========================================")
    print("OptiBF - Dataset Analysis")
    print("========================================")

    print(
        f"\nDataset loaded: "
        f"{len(df)} samples"
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    return df


# =========================================================
# BASIC STATISTICS
# =========================================================

def report_statistics(df):

    print("\n========================================")
    print("BASIC STATISTICS")
    print("========================================")

    print("\nBF:")

    print(
        df["BF"].describe()
    )

    print("\nDelta BF (%):")

    print(
        df["Delta_BF_percent"].describe()
    )


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def report_correlations(df):

    variables = [
        "L1_mm",
        "L3_mm",
        "L4_mm",
        "theta1_deg",
        "theta2_deg",
        "mu"
    ]

    print("\n========================================")
    print("CORRELATION ANALYSIS")
    print("========================================")

    correlation_bf = (
        df[variables + ["BF"]]
        .corr()["BF"]
        .drop("BF")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    correlation_delta_bf = (
        df[variables + ["Delta_BF_percent"]]
        .corr()["Delta_BF_percent"]
        .drop("Delta_BF_percent")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    print("\nCorrelation with BF:")

    print(
        correlation_bf
    )

    print("\nCorrelation with Delta BF:")

    print(
        correlation_delta_bf
    )


# =========================================================
# DELTA BF CONDITIONED ON MU
# =========================================================

def report_delta_bf_by_mu(df):

    print("\n========================================")
    print("DELTA BF CONDITIONED ON MU")
    print("========================================")

    bins = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40
    ]

    labels = [
        "0.05–0.10",
        "0.10–0.15",
        "0.15–0.20",
        "0.20–0.25",
        "0.25–0.30",
        "0.30–0.35",
        "0.35–0.40"
    ]

    df_analysis = df.copy()

    df_analysis["mu_range"] = pd.cut(
        df_analysis["mu"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    summary = (
        df_analysis
        .groupby(
            "mu_range",
            observed=True
        )["Delta_BF_percent"]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ])
    )

    print(summary)

    return summary


# =========================================================
# DELTA BF BY MU RANGE - BOX PLOT
# =========================================================

def plot_delta_bf_by_mu(df):

    bins = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40
    ]

    labels = [
        "0.05–0.10",
        "0.10–0.15",
        "0.15–0.20",
        "0.20–0.25",
        "0.25–0.30",
        "0.30–0.35",
        "0.35–0.40"
    ]

    df_analysis = df.copy()

    df_analysis["mu_range"] = pd.cut(
        df_analysis["mu"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    groups = []

    for label in labels:

        values = df_analysis.loc[
            df_analysis["mu_range"] == label,
            "Delta_BF_percent"
        ]

        groups.append(values)

    plt.figure()

    plt.boxplot(
        groups,
        labels=labels
    )

    plt.xlabel(
        "Friction coefficient range (μ)"
    )

    plt.ylabel(
        "ΔBF (%)"
    )

    plt.title(
        "Brake Factor Variation by Friction Range"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# BF DISTRIBUTION
# =========================================================

def plot_bf_distribution(df):

    plt.figure()

    plt.hist(
        df["BF"],
        bins=30
    )

    plt.xlabel(
        "Brake Factor (BF)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Distribution of Brake Factor"
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# DELTA BF DISTRIBUTION
# =========================================================

def plot_delta_bf_distribution(df):

    plt.figure()

    plt.hist(
        df["Delta_BF_percent"],
        bins=30
    )

    plt.xlabel(
        "ΔBF (%)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Distribution of Brake Factor Variation"
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# VARIABLES VS BF
# =========================================================

def plot_variables_vs_bf(df):

    variables = [
        ("L1_mm", "L1 (mm)"),
        ("L3_mm", "L3 (mm)"),
        ("L4_mm", "L4 (mm)"),
        ("theta1_deg", "θ1 (deg)"),
        ("theta2_deg", "θ2 (deg)"),
        ("mu", "Friction coefficient (μ)")
    ]

    for column, label in variables:

        plt.figure()

        plt.scatter(
            df[column],
            df["BF"],
            alpha=0.5
        )

        plt.xlabel(
            label
        )

        plt.ylabel(
            "Brake Factor (BF)"
        )

        plt.title(
            f"{label} vs Brake Factor"
        )

        plt.tight_layout()

        plt.show()


# =========================================================
# VARIABLES VS DELTA BF
# =========================================================

def plot_variables_vs_delta_bf(df):

    variables = [
        ("L1_mm", "L1 (mm)"),
        ("L3_mm", "L3 (mm)"),
        ("L4_mm", "L4 (mm)"),
        ("theta1_deg", "θ1 (deg)"),
        ("theta2_deg", "θ2 (deg)"),
        ("mu", "Friction coefficient (μ)")
    ]

    for column, label in variables:

        plt.figure()

        plt.scatter(
            df[column],
            df["Delta_BF_percent"],
            alpha=0.5
        )

        plt.xlabel(
            label
        )

        plt.ylabel(
            "ΔBF (%)"
        )

        plt.title(
            f"{label} vs ΔBF"
        )

        plt.tight_layout()

        plt.show()


# =========================================================
# PERFORMANCE VS ROBUSTNESS
# =========================================================

def plot_bf_vs_delta_bf(df):

    plt.figure()

    plt.scatter(
        df["BF"],
        df["Delta_BF_percent"],
        alpha=0.5
    )

    plt.xlabel(
        "Brake Factor (BF)"
    )

    plt.ylabel(
        "ΔBF (%)"
    )

    plt.title(
        "Performance vs Robustness"
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_dataset()

    report_statistics(df)

    report_correlations(df)

    report_delta_bf_by_mu(df)

    plot_bf_distribution(df)

    plot_delta_bf_distribution(df)

    plot_variables_vs_bf(df)

    plot_variables_vs_delta_bf(df)

    plot_delta_bf_by_mu(df)

    plot_bf_vs_delta_bf(df)


if __name__ == "__main__":
    main()