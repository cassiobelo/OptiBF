import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "sensitivity_plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NOMES DAS VARIÁVEIS
# ============================================================

VARIABLE_LABELS = {
    "L1": "L1",
    "L1_mm": "L1",

    "L3": "L3",
    "L3_mm": "L3",

    "L4": "L4",
    "L4_mm": "L4",

    "theta1": r"$\theta_1$",
    "theta1_deg": r"$\theta_1$",
    "theta1_deg_raw": r"$\theta_1$",

    "theta2": r"$\theta_2$",
    "theta2_deg": r"$\theta_2$",
    "theta2_deg_raw": r"$\theta_2$",

    "mu": r"$\mu$",
    "MU": r"$\mu$",
}


def variable_label(value):
    return VARIABLE_LABELS.get(str(value), str(value))


# ============================================================
# IDENTIFICAÇÃO AUTOMÁTICA DAS COLUNAS
# ============================================================

def find_column(df, candidates):

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    return None


def find_variable_column(df):

    return find_column(
        df,
        [
            "Variable",
            "variable",
            "Parameter",
            "parameter",
            "Name",
            "name",
            "Feature",
            "feature",
        ]
    )


def find_mustar_column(df):

    return find_column(
        df,
        [
            "mu_star",
            "mu*",
            "mu_star_mean",
            "Mu_star",
            "Mu*",
            "mean",
        ]
    )


def find_sigma_column(df):

    return find_column(
        df,
        [
            "sigma",
            "Sigma",
        ]
    )


def find_s1_column(df):

    return find_column(
        df,
        [
            "S1",
            "s1",
            "First Order",
            "first_order",
        ]
    )


def find_st_column(df):

    return find_column(
        df,
        [
            "ST",
            "st",
            "Total Order",
            "total_order",
        ]
    )


# ============================================================
# MORRIS
# ============================================================

def generate_morris(csv_file, output_name, title):

    path = os.path.join(DATA_DIR, csv_file)

    if not os.path.exists(path):

        print(f"[AVISO] Arquivo não encontrado: {path}")
        return None

    df = pd.read_csv(path)

    print(f"\nMorris: {path}")
    print("Colunas:", list(df.columns))

    var_col = find_variable_column(df)
    mu_star_col = find_mustar_column(df)
    sigma_col = find_sigma_column(df)

    if var_col is None:
        print("[ERRO] Coluna de variável não encontrada.")
        return None

    if mu_star_col is None:
        print("[ERRO] Coluna mu* não encontrada.")
        return None

    if sigma_col is None:
        print("[ERRO] Coluna sigma não encontrada.")
        return None

    df[mu_star_col] = pd.to_numeric(
        df[mu_star_col],
        errors="coerce"
    )

    df[sigma_col] = pd.to_numeric(
        df[sigma_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            mu_star_col,
            sigma_col
        ]
    )

    # --------------------------------------------------------
    # GRÁFICO MORRIS μ* × σ
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        df[mu_star_col],
        df[sigma_col],
        s=80
    )

    for _, row in df.iterrows():

        ax.annotate(
            variable_label(row[var_col]),
            (
                row[mu_star_col],
                row[sigma_col]
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=11
        )

    ax.set_xlabel(
        r"$\mu^*$",
        fontsize=12
    )

    ax.set_ylabel(
        r"$\sigma$",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=13
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Gráfico salvo: {output_path}"
    )

    # --------------------------------------------------------
    # RANKING μ*
    # --------------------------------------------------------

    ranking = df.sort_values(
        mu_star_col,
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    labels = [
        variable_label(x)
        for x in ranking[var_col]
    ]

    ax.barh(
        labels,
        ranking[mu_star_col]
    )

    ax.set_xlabel(
        r"$\mu^*$",
        fontsize=12
    )

    ax.set_title(
        f"{title} — Variable Ranking",
        fontsize=13
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    ranking_path = os.path.join(
        OUTPUT_DIR,
        output_name.replace(
            ".png",
            "_ranking.png"
        )
    )

    fig.savefig(
        ranking_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Ranking salvo: {ranking_path}"
    )

    return df


# ============================================================
# SOBOL
# ============================================================

def generate_sobol(csv_file, output_name, title):

    path = os.path.join(DATA_DIR, csv_file)

    if not os.path.exists(path):

        print(f"[AVISO] Arquivo não encontrado: {path}")
        return None

    df = pd.read_csv(path)

    print(f"\nSobol: {path}")
    print("Colunas:", list(df.columns))

    var_col = find_variable_column(df)
    s1_col = find_s1_column(df)
    st_col = find_st_column(df)

    if var_col is None:
        print("[ERRO] Coluna de variável não encontrada.")
        return None

    if s1_col is None:
        print("[ERRO] Coluna S1 não encontrada.")
        return None

    if st_col is None:
        print("[ERRO] Coluna ST não encontrada.")
        return None

    df[s1_col] = pd.to_numeric(
        df[s1_col],
        errors="coerce"
    )

    df[st_col] = pd.to_numeric(
        df[st_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            s1_col,
            st_col
        ]
    )

    # --------------------------------------------------------
    # SOBOL S1 × ST
    # --------------------------------------------------------

    x = np.arange(
        len(df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        x - width / 2,
        df[s1_col],
        width,
        label=r"$S_1$"
    )

    ax.bar(
        x + width / 2,
        df[st_col],
        width,
        label=r"$S_T$"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            variable_label(x)
            for x in df[var_col]
        ]
    )

    ax.set_ylabel(
        "Sensitivity Index",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=13
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Gráfico salvo: {output_path}"
    )

    return df


# ============================================================
# BF × ΔBF
# ============================================================

def generate_comparison(
    bf_df,
    delta_df
):

    if bf_df is None or delta_df is None:

        print(
            "\n[AVISO] Não foi possível gerar "
            "a comparação BF × ΔBF."
        )

        return

    var_bf = find_variable_column(
        bf_df
    )

    var_delta = find_variable_column(
        delta_df
    )

    mu_bf = find_mustar_column(
        bf_df
    )

    mu_delta = find_mustar_column(
        delta_df
    )

    if None in [
        var_bf,
        var_delta,
        mu_bf,
        mu_delta
    ]:

        print(
            "[ERRO] Colunas necessárias "
            "para comparação não encontradas."
        )

        return

    bf = bf_df[
        [
            var_bf,
            mu_bf
        ]
    ].copy()

    delta = delta_df[
        [
            var_delta,
            mu_delta
        ]
    ].copy()

    bf.columns = [
        "variable",
        "BF"
    ]

    delta.columns = [
        "variable",
        "DeltaBF"
    ]

    data = pd.merge(
        bf,
        delta,
        on="variable",
        how="inner"
    )

    data["BF"] = pd.to_numeric(
        data["BF"],
        errors="coerce"
    )

    data["DeltaBF"] = pd.to_numeric(
        data["DeltaBF"],
        errors="coerce"
    )

    data = data.dropna()

    # Normalização
    data["BF_norm"] = (
        data["BF"]
        / data["BF"].abs().max()
    )

    data["DeltaBF_norm"] = (
        data["DeltaBF"]
        / data["DeltaBF"].abs().max()
    )

    # Ordenar pela importância no BF
    data = data.sort_values(
        "BF_norm",
        ascending=True
    )

    y = np.arange(
        len(data)
    )

    height = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        y - height / 2,
        data["BF_norm"],
        height,
        label="BF"
    )

    ax.barh(
        y + height / 2,
        data["DeltaBF_norm"],
        height,
        label=r"$\Delta BF$"
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        [
            variable_label(x)
            for x in data["variable"]
        ]
    )

    ax.set_xlabel(
        "Normalized importance",
        fontsize=12
    )

    ax.set_title(
        "Variable Importance — BF vs. ΔBF",
        fontsize=13
    )

    ax.set_xlim(
        0,
        1.1
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "bf_vs_delta_bf.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Comparação salva: {output_path}"
    )

    # Também salva os dados usados
    data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "bf_vs_delta_bf_values.csv"
        ),
        index=False
    )



# ============================================================
# SS-ANOVA
# ============================================================

def generate_ss_anova(
    csv_file,
    output_name,
    title,
    top_n=None
):
    """
    Gera gráfico de contribuição dos efeitos a partir
    dos resultados reais de SS-Anova.
    """

    path = os.path.join(
        DATA_DIR,
        csv_file
    )

    if not os.path.exists(path):

        print(
            f"[AVISO] Arquivo não encontrado: {path}"
        )

        return None

    df = pd.read_csv(path)

    print(f"\nSS-Anova: {path}")
    print("Colunas:", list(df.columns))

    effect_col = find_column(
        df,
        [
            "Effect",
            "effect",
            "Variable",
            "variable",
            "Term",
            "term"
        ]
    )

    contribution_col = find_column(
        df,
        [
            "Contribution_percent",
            "contribution_percent",
            "Contribution (%)",
            "Contribution",
            "contribution"
        ]
    )

    if effect_col is None:

        print(
            "[ERRO] Coluna de efeito não encontrada."
        )

        return None

    if contribution_col is None:

        print(
            "[ERRO] Coluna de contribuição "
            "não encontrada."
        )

        return None

    df[contribution_col] = pd.to_numeric(
        df[contribution_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[contribution_col]
    )

    # Ordena pela contribuição real.
    df = df.sort_values(
        contribution_col,
        ascending=False
    )

    # Para manter o gráfico legível, limitar aos
    # principais efeitos quando solicitado.
    if top_n is not None:

        plot_df = df.head(top_n).copy()

    else:

        plot_df = df.copy()

    plot_df = plot_df.sort_values(
        contribution_col,
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df[effect_col].astype(str),
        plot_df[contribution_col]
    )

    ax.set_xlabel(
        "Contribution to variance (%)",
        fontsize=12
    )

    ax.set_ylabel(
        "Effect",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=13
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    # Mostrar o valor percentual ao final da barra.
    max_value = plot_df[contribution_col].max()

    for i, value in enumerate(
        plot_df[contribution_col]
    ):

        offset = (
            max_value * 0.01
            if max_value > 0
            else 0.01
        )

        ax.text(
            value + offset,
            i,
            f"{value:.2f}%",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Gráfico SS-Anova salvo: "
        f"{output_path}"
    )

    # Salvar também uma cópia dos resultados
    # utilizados no gráfico.
    values_path = os.path.join(
        OUTPUT_DIR,
        output_name.replace(
            ".png",
            "_values.csv"
        )
    )

    df.to_csv(
        values_path,
        index=False
    )

    print(
        f"[OK] Dados SS-Anova salvos: "
        f"{values_path}"
    )

    return df


# ============================================================
# EXECUÇÃO
# ============================================================

print("=" * 60)
print("OptiBF - Sensitivity Analysis Plots")
print("=" * 60)

print(
    f"\nPasta de saída:\n{OUTPUT_DIR}"
)


# Morris
morris_bf = generate_morris(
    "morris_bf.csv",
    "morris_bf.png",
    "Morris Sensitivity Analysis — BF"
)

morris_delta = generate_morris(
    "morris_delta_bf.csv",
    "morris_delta_bf.png",
    r"Morris Sensitivity Analysis — $\Delta BF$"
)


# Sobol
sobol_bf = generate_sobol(
    "sobol_bf.csv",
    "sobol_bf.png",
    "Sobol Sensitivity Analysis — BF"
)

sobol_delta = generate_sobol(
    "sobol_delta_bf.csv",
    "sobol_delta_bf.png",
    r"Sobol Sensitivity Analysis — $\Delta BF$"
)


# Comparação
generate_comparison(
    morris_bf,
    morris_delta
)


# ============================================================
# SS-ANOVA
# ============================================================

ss_anova_bf = generate_ss_anova(
    "ss_anova_bf.csv",
    "ss_anova_bf.png",
    "SS-Anova Sensitivity — BF",
    top_n=15
)

ss_anova_delta = generate_ss_anova(
    "ss_anova_delta_bf.csv",
    "ss_anova_delta_bf.png",
    r"SS-Anova Sensitivity — $\Delta BF$",
    top_n=15
)


print("\n" + "=" * 60)
print("FINALIZADO")
print("=" * 60)

print(
    f"\nOs gráficos estão em:\n{OUTPUT_DIR}"
)