import os
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Diretório do próprio arquivo .py
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Pasta data do projeto
DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

# Pasta específica para os gráficos com μ fixado
OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "sensitivity_plots_mu_fixado"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARÂMETROS DA ANÁLISE
# ============================================================

FIXED_MU = 0.30


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

    return VARIABLE_LABELS.get(
        str(value),
        str(value)
    )


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def find_column(
    df,
    candidates
):

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


def find_effect_column(df):

    return find_column(
        df,
        [
            "Effect",
            "effect",
            "Variable",
            "variable",
            "Term",
            "term",
        ]
    )


def find_contribution_column(df):

    return find_column(
        df,
        [
            "Contribution_percent",
            "contribution_percent",
            "Contribution (%)",
            "Contribution",
            "contribution",
        ]
    )


def csv_path(filename):

    return os.path.join(
        DATA_DIR,
        filename
    )


def output_path(filename):

    return os.path.join(
        OUTPUT_DIR,
        filename
    )


# ============================================================
# MORRIS — GEOMETRIA
# ============================================================

def generate_geometry_morris(
    csv_file,
    output_name,
    title
):

    path = csv_path(csv_file)

    if not os.path.exists(path):

        print(
            f"[AVISO] Arquivo não encontrado:"
            f"\n{path}"
        )

        return None

    df = pd.read_csv(path)

    print(
        f"\nMorris geométrico:"
        f"\n{path}"
    )

    print(
        "Colunas:",
        list(df.columns)
    )

    var_col = find_variable_column(df)

    mu_star_col = find_mustar_column(df)

    sigma_col = find_sigma_column(df)

    if var_col is None:

        print(
            "[ERRO] Coluna de variável "
            "não encontrada."
        )

        return None

    if mu_star_col is None:

        print(
            "[ERRO] Coluna mu* "
            "não encontrada."
        )

        return None

    if sigma_col is None:

        print(
            "[ERRO] Coluna sigma "
            "não encontrada."
        )

        return None

    # --------------------------------------------------------
    # CONVERSÃO NUMÉRICA
    # --------------------------------------------------------

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

    if df.empty:

        print(
            "[ERRO] Nenhum dado válido "
            "encontrado no arquivo."
        )

        return None

    # --------------------------------------------------------
    # MORRIS: μ* × σ
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        df[mu_star_col],
        df[sigma_col],
        s=90
    )

    for _, row in df.iterrows():

        label = variable_label(
            row[var_col]
        )

        ax.annotate(
            label,
            (
                row[mu_star_col],
                row[sigma_col]
            ),
            xytext=(8, 8),
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
        fontsize=14
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_path = output_path(
        output_name
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Gráfico salvo:"
        f"\n{save_path}"
    )

    # --------------------------------------------------------
    # RANKING MORRIS
    # --------------------------------------------------------

    ranking = df.sort_values(
        mu_star_col,
        ascending=True
    )

    labels = [
        variable_label(value)
        for value in ranking[var_col]
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

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

    ranking_filename = (
        output_name.replace(
            ".png",
            "_ranking.png"
        )
    )

    ranking_path = output_path(
        ranking_filename
    )

    fig.savefig(
        ranking_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Ranking salvo:"
        f"\n{ranking_path}"
    )

    # --------------------------------------------------------
    # SALVAR CSV UTILIZADO
    # --------------------------------------------------------

    csv_output = output_path(
        output_name.replace(
            ".png",
            "_values.csv"
        )
    )

    df.to_csv(
        csv_output,
        index=False
    )

    print(
        f"[OK] Dados salvos:"
        f"\n{csv_output}"
    )

    return df


# ============================================================
# SOBOL — GEOMETRIA
# ============================================================

def generate_geometry_sobol(
    csv_file,
    output_name,
    title
):

    path = csv_path(csv_file)

    if not os.path.exists(path):

        print(
            f"[AVISO] Arquivo não encontrado:"
            f"\n{path}"
        )

        return None

    df = pd.read_csv(path)

    print(
        f"\nSobol geométrico:"
        f"\n{path}"
    )

    print(
        "Colunas:",
        list(df.columns)
    )

    var_col = find_variable_column(df)

    s1_col = find_s1_column(df)

    st_col = find_st_column(df)

    if var_col is None:

        print(
            "[ERRO] Coluna de variável "
            "não encontrada."
        )

        return None

    if s1_col is None:

        print(
            "[ERRO] Coluna S1 "
            "não encontrada."
        )

        return None

    if st_col is None:

        print(
            "[ERRO] Coluna ST "
            "não encontrada."
        )

        return None

    # --------------------------------------------------------
    # CONVERSÃO NUMÉRICA
    # --------------------------------------------------------

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

    if df.empty:

        print(
            "[ERRO] Nenhum dado válido "
            "encontrado no arquivo."
        )

        return None

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

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            variable_label(value)
            for value in df[var_col]
        ]
    )

    ax.set_ylabel(
        "Sensitivity Index",
        fontsize=12
    )

    ax.set_xlabel(
        "Geometry variable",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=14
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    save_path = output_path(
        output_name
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Gráfico salvo:"
        f"\n{save_path}"
    )

    # --------------------------------------------------------
    # SALVAR CSV UTILIZADO
    # --------------------------------------------------------

    csv_output = output_path(
        output_name.replace(
            ".png",
            "_values.csv"
        )
    )

    df.to_csv(
        csv_output,
        index=False
    )

    print(
        f"[OK] Dados salvos:"
        f"\n{csv_output}"
    )

    return df


# ============================================================
# SS-ANOVA — GEOMETRIA
# ============================================================

def generate_geometry_ss_anova(
    csv_file,
    output_name,
    title,
    top_n=15
):

    path = csv_path(csv_file)

    if not os.path.exists(path):

        print(
            f"[AVISO] Arquivo não encontrado:"
            f"\n{path}"
        )

        return None

    df = pd.read_csv(path)

    print(
        f"\nSS-Anova geométrica:"
        f"\n{path}"
    )

    print(
        "Colunas:",
        list(df.columns)
    )

    effect_col = find_effect_column(df)

    contribution_col = find_contribution_column(df)

    if effect_col is None:

        print(
            "[ERRO] Coluna de efeito "
            "não encontrada."
        )

        return None

    if contribution_col is None:

        print(
            "[ERRO] Coluna de contribuição "
            "não encontrada."
        )

        return None

    # --------------------------------------------------------
    # CONVERSÃO NUMÉRICA
    # --------------------------------------------------------

    df[contribution_col] = pd.to_numeric(
        df[contribution_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            contribution_col
        ]
    )

    if df.empty:

        print(
            "[ERRO] Nenhum dado válido "
            "encontrado no arquivo."
        )

        return None

    # --------------------------------------------------------
    # ORDENAR POR CONTRIBUIÇÃO
    # --------------------------------------------------------

    df = df.sort_values(
        contribution_col,
        ascending=False
    )

    if top_n is not None:

        plot_df = df.head(
            top_n
        ).copy()

    else:

        plot_df = df.copy()

    plot_df = plot_df.sort_values(
        contribution_col,
        ascending=True
    )

    # --------------------------------------------------------
    # FORMATAR NOMES DOS EFEITOS
    # --------------------------------------------------------

    def format_effect(effect):

        text = str(effect)

        # Trata multiplicações escritas com ×
        if "×" in text:

            parts = text.split("×")

            labels = [
                variable_label(
                    part.strip()
                )
                for part in parts
            ]

            return " × ".join(
                labels
            )

        # Trata também possíveis interações
        # escritas com *
        if " * " in text:

            parts = text.split(
                " * "
            )

            labels = [
                variable_label(
                    part.strip()
                )
                for part in parts
            ]

            return " × ".join(
                labels
            )

        return variable_label(
            text
        )

    labels = [
        format_effect(effect)
        for effect in plot_df[effect_col]
    ]

    # --------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        labels,
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
        fontsize=14
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    max_value = plot_df[
        contribution_col
    ].max()

    for i, value in enumerate(
        plot_df[contribution_col]
    ):

        offset = max(
            max_value * 0.01,
            0.01
        )

        ax.text(
            value + offset,
            i,
            f"{value:.2f}%",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    save_path = output_path(
        output_name
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Gráfico SS-Anova salvo:"
        f"\n{save_path}"
    )

    # --------------------------------------------------------
    # SALVAR CSV UTILIZADO
    # --------------------------------------------------------

    csv_output = output_path(
        output_name.replace(
            ".png",
            "_values.csv"
        )
    )

    df.to_csv(
        csv_output,
        index=False
    )

    print(
        f"[OK] Dados SS-Anova salvos:"
        f"\n{csv_output}"
    )

    return df


# ============================================================
# COMPARAÇÃO MORRIS — BF × ΔBF
# ============================================================

def generate_morris_comparison(
    bf_df,
    delta_df
):

    if bf_df is None:

        print(
            "[AVISO] Morris BF não disponível."
        )

        return

    if delta_df is None:

        print(
            "[AVISO] Morris ΔBF não disponível."
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
            "[ERRO] Não foi possível "
            "gerar comparação Morris."
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

    if data.empty:

        print(
            "[ERRO] Nenhum dado comum "
            "entre BF e ΔBF."
        )

        return

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    bf_max = data["BF"].abs().max()

    delta_max = data[
        "DeltaBF"
    ].abs().max()

    if bf_max > 0:

        data["BF_norm"] = (
            data["BF"]
            / bf_max
        )

    else:

        data["BF_norm"] = 0.0

    if delta_max > 0:

        data["DeltaBF_norm"] = (
            data["DeltaBF"]
            / delta_max
        )

    else:

        data["DeltaBF_norm"] = 0.0

    data = data.sort_values(
        "BF_norm",
        ascending=True
    )

    y = np.arange(
        len(data)
    )

    height = 0.35

    # --------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------

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

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        [
            variable_label(value)
            for value in data["variable"]
        ]
    )

    ax.set_xlabel(
        "Normalized importance",
        fontsize=12
    )

    ax.set_title(
        rf"Variable Importance — Geometry Only — "
        rf"BF vs. $\Delta BF$ ($\mu = {FIXED_MU:.2f}$)",
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

    save_path = output_path(
        "geometry_bf_vs_delta_bf.png"
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] Comparação salva:"
        f"\n{save_path}"
    )

    # --------------------------------------------------------
    # SALVAR DADOS
    # --------------------------------------------------------

    csv_output = output_path(
        "geometry_bf_vs_delta_bf_values.csv"
    )

    data.to_csv(
        csv_output,
        index=False
    )

    print(
        f"[OK] Dados da comparação salvos:"
        f"\n{csv_output}"
    )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("=" * 60)

    print(
        "OptiBF - Geometry Sensitivity Analysis Plots"
    )

    print("=" * 60)

    print(
        f"\nDiretório do script:"
        f"\n{BASE_DIR}"
    )

    print(
        f"\nDiretório dos dados:"
        f"\n{DATA_DIR}"
    )

    print(
        f"\nPasta de saída:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nCoeficiente de atrito fixado:"
        f" μ = {FIXED_MU:.2f}"
    )

    print(
        "\nNesta análise o μ NÃO é uma variável."
    )

    print(
        "A sensibilidade é calculada "
        "exclusivamente entre as variáveis geométricas."
    )

    print("\nCSV de entrada esperados:")
    expected_csv = [
        "morris_geometry_bf.csv",
        "morris_geometry_delta_bf.csv",
        "sobol_geometry_bf.csv",
        "sobol_geometry_delta_bf.csv",
        "ss_anova_geometry_bf.csv",
        "ss_anova_geometry_delta_bf.csv",
    ]

    for filename in expected_csv:
        p = csv_path(filename)
        if os.path.exists(p):
            print(f"  [OK] {p}")
        else:
            print(f"  [FALTA] {p}")

    # ========================================================
    # MORRIS
    # ========================================================

    print("\n" + "=" * 60)
    print("MORRIS — GEOMETRY ONLY")
    print("=" * 60)

    morris_bf = generate_geometry_morris(
        "morris_geometry_bf.csv",
        "morris_geometry_bf.png",
        rf"Morris Sensitivity Analysis — "
        rf"Geometry Only — BF "
        rf"($\mu = {FIXED_MU:.2f}$)"
    )

    morris_delta = generate_geometry_morris(
        "morris_geometry_delta_bf.csv",
        "morris_geometry_delta_bf.png",
        rf"Morris Sensitivity Analysis — "
        rf"Geometry Only — $\Delta BF$ "
        rf"($\mu = {FIXED_MU:.2f}$)"
    )

    # ========================================================
    # COMPARAÇÃO MORRIS
    # ========================================================

    print("\n" + "=" * 60)
    print("MORRIS — BF vs. ΔBF")
    print("=" * 60)

    generate_morris_comparison(
        morris_bf,
        morris_delta
    )

    # ========================================================
    # SOBOL
    # ========================================================

    print("\n" + "=" * 60)
    print("SOBOL — GEOMETRY ONLY")
    print("=" * 60)

    generate_geometry_sobol(
        "sobol_geometry_bf.csv",
        "sobol_geometry_bf.png",
        rf"Sobol Sensitivity Analysis — "
        rf"Geometry Only — BF "
        rf"($\mu = {FIXED_MU:.2f}$)"
    )

    generate_geometry_sobol(
        "sobol_geometry_delta_bf.csv",
        "sobol_geometry_delta_bf.png",
        rf"Sobol Sensitivity Analysis — "
        rf"Geometry Only — $\Delta BF$ "
        rf"($\mu = {FIXED_MU:.2f}$)"
    )

    # ========================================================
    # SS-ANOVA
    # ========================================================

    print("\n" + "=" * 60)
    print("SS-ANOVA — GEOMETRY ONLY")
    print("=" * 60)

    generate_geometry_ss_anova(
        "ss_anova_geometry_bf.csv",
        "ss_anova_geometry_bf.png",
        rf"SS-Anova Sensitivity — "
        rf"Geometry Only — BF "
        rf"($\mu = {FIXED_MU:.2f}$)",
        top_n=15
    )

    generate_geometry_ss_anova(
        "ss_anova_geometry_delta_bf.csv",
        "ss_anova_geometry_delta_bf.png",
        rf"SS-Anova Sensitivity — "
        rf"Geometry Only — $\Delta BF$ "
        rf"($\mu = {FIXED_MU:.2f}$)",
        top_n=15
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("FINALIZADO COM SUCESSO")
    print("=" * 60)

    print(
        f"\nOs gráficos estão em:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "\nArquivos esperados:"
    )

    expected_files = [

        "morris_geometry_bf.png",
        "morris_geometry_bf_ranking.png",

        "morris_geometry_delta_bf.png",
        "morris_geometry_delta_bf_ranking.png",

        "geometry_bf_vs_delta_bf.png",

        "sobol_geometry_bf.png",
        "sobol_geometry_delta_bf.png",

        "ss_anova_geometry_bf.png",
        "ss_anova_geometry_delta_bf.png",
    ]

    for filename in expected_files:

        full_path = output_path(
            filename
        )

        if os.path.exists(
            full_path
        ):

            print(
                f"  [OK] {filename}"
            )

        else:

            print(
                f"  [--] {filename}"
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n" + "=" * 60)
        print("ERRO DURANTE A EXECUÇÃO")
        print("=" * 60)

        print(
            f"\nTipo:"
            f" {type(error).__name__}"
        )

        print(
            f"\nMensagem:"
            f"\n{error}"
        )

        print(
            "\nTraceback completo:"
        )

        traceback.print_exc()

    finally:

        print(
            "\n" + "=" * 60
        )

        input(
            "Pressione ENTER para fechar..."
        )