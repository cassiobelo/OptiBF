import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from pathlib import Path

from sklearn.model_selection import (
    KFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor
)

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from core.brake_model import (
    RD,
    BOUNDS,
    calculate_bf
)


# =========================================================
# TUNING CONFIGURATION
# =========================================================

TUNING_STAGE = 5

# Number of Optuna trials for the global search.
N_TRIALS = 50


# =========================================================
# CURRENT BEST XGBOOST CONFIGURATION
# =========================================================
#
# Best configuration validated independently so far:
# Tuning 3.
#
# Independent validation RMSE:
# 0.012993
# =========================================================

BEST_XGB_PARAMS = {

    "max_depth":
        3,

    "learning_rate":
        0.0609418819,

    "n_estimators":
        800,

    "subsample":
        0.6030189277,

    "colsample_bytree":
        0.9852181122
}


# =========================================================
# ORIGINAL XGBOOST BASELINE
# =========================================================

BASELINE_XGB_PARAMS = {

    "max_depth":
        4,

    "learning_rate":
        0.05,

    "n_estimators":
        300,

    "subsample":
        0.90,

    "colsample_bytree":
        0.90
}


# =========================================================
# GENERAL CONFIGURATION
# =========================================================

DATA_FILE = (
    "data/optibf_dataset_lhs_1000_robustness.csv"
)

OUTPUT_DIR = Path("data")

PLOTS_DIR = (
    OUTPUT_DIR /
    "ml_plots"
)

RANDOM_SEED = 42

N_SPLITS = 5


# =========================================================
# INDEPENDENT VALIDATION CONFIGURATION
# =========================================================

VALIDATION_SAMPLES = 500

VALIDATION_SEED = 2026


# =========================================================
# ROBUSTNESS VALIDATION CONFIGURATION
# =========================================================

ROBUSTNESS_SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030
]

ROBUSTNESS_SAMPLES = 500


# =========================================================
# FEATURES AND TARGET
# =========================================================

FEATURES = [

    "L1_mm",
    "L3_mm",
    "L4_mm",
    "theta1_deg",
    "theta2_deg",
    "mu"

]

TARGET = "BF"


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "========================================"
    )

    print(
        "OptiBF - Machine Learning Analysis"
    )

    print(
        "========================================"
    )

    print(
        f"\nDataset loaded: "
        f"{len(df)} samples"
    )

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET
    ].copy()

    return X, y


# =========================================================
# GENERATE INDEPENDENT VALIDATION SET
# =========================================================

def generate_validation_set(
    seed=None,
    samples=None
):

    if seed is None:
        seed = VALIDATION_SEED

    if samples is None:
        samples = VALIDATION_SAMPLES

    rng = np.random.default_rng(seed)

    data = {}

    for feature in FEATURES:

        lower, upper = BOUNDS[
            feature
        ]

        data[feature] = rng.uniform(
            lower,
            upper,
            samples
        )

    return pd.DataFrame(data)


# =========================================================
# EVALUATE PHYSICAL MODEL
# =========================================================

def evaluate_physical_model(X):

    results = []

    for _, row in X.iterrows():

        result = calculate_bf(

            L1=(
                row["L1_mm"]
                / 1000.0
            ),

            L3=(
                row["L3_mm"]
                / 1000.0
            ),

            L4=(
                row["L4_mm"]
                / 1000.0
            ),

            theta1=np.radians(
                row["theta1_deg"]
            ),

            theta2=np.radians(
                row["theta2_deg"]
            ),

            mu=row["mu"],

            RD=RD

        )

        if result is None:

            results.append(
                np.nan
            )

        else:

            results.append(
                result["BF"]
            )

    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# DEFINE MODELS
# =========================================================

def define_models():

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    random_forest = RandomForestRegressor(

        n_estimators=300,

        random_state=RANDOM_SEED,

        n_jobs=1
    )


    # -----------------------------------------------------
    # XGBOOST BASELINE
    # -----------------------------------------------------

    xgboost = XGBRegressor(

        **BASELINE_XGB_PARAMS,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )


    # -----------------------------------------------------
    # MLP
    # -----------------------------------------------------

    mlp = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",

            MLPRegressor(

                hidden_layer_sizes=(
                    64,
                    64
                ),

                activation="relu",

                solver="adam",

                learning_rate_init=0.001,

                max_iter=2000,

                early_stopping=True,

                validation_fraction=0.15,

                n_iter_no_change=30,

                random_state=RANDOM_SEED
            )
        )
    ])


    # -----------------------------------------------------
    # SVR
    # -----------------------------------------------------

    svr = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",

            SVR(

                kernel="rbf",

                C=100.0,

                gamma="scale",

                epsilon=0.01
            )
        )
    ])


    # -----------------------------------------------------
    # ENSEMBLE
    # -----------------------------------------------------

    ensemble = VotingRegressor(

        estimators=[

            (
                "RF",
                random_forest
            ),

            (
                "XGB",
                xgboost
            ),

            (
                "MLP",
                mlp
            )

        ],

        weights=[

            1,
            1,
            1

        ],

        n_jobs=1
    )


    # -----------------------------------------------------
    # MODEL DICTIONARY
    # -----------------------------------------------------

    models = {

        "Random Forest":
            random_forest,

        "XGBoost":
            xgboost,

        "MLP":
            mlp,

        "SVR":
            svr,

        "Ensemble":
            ensemble
    }

    return models


# =========================================================
# CROSS VALIDATION
# =========================================================

def evaluate_models(
    X,
    y,
    models
):

    print(
        "\n========================================"
    )

    print(
        "MODEL EVALUATION"
    )

    print(
        "========================================"
    )

    cv = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_SEED
    )

    results = []

    for name, model in models.items():

        print(
            f"\nEvaluating: {name}"
        )

        scores = cross_validate(

            model,

            X,

            y,

            cv=cv,

            scoring={

                "r2":
                    "r2",

                "rmse":
                    "neg_root_mean_squared_error",

                "mae":
                    "neg_mean_absolute_error"

            },

            n_jobs=-1
        )

        results.append({

            "Model":
                name,

            "R2_mean":
                scores[
                    "test_r2"
                ].mean(),

            "R2_std":
                scores[
                    "test_r2"
                ].std(),

            "RMSE_mean":
                -scores[
                    "test_rmse"
                ].mean(),

            "RMSE_std":
                scores[
                    "test_rmse"
                ].std(),

            "MAE_mean":
                -scores[
                    "test_mae"
                ].mean(),

            "MAE_std":
                scores[
                    "test_mae"
                ].std()
        })

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df
        .sort_values(
            "RMSE_mean"
        )
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(results):

    print(
        "\n========================================"
    )

    print(
        "CROSS-VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        results.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (

        OUTPUT_DIR /
        "ml_model_comparison_bf.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


# =========================================================
# XGBOOST FEATURE IMPORTANCE
# =========================================================

def save_feature_importance(
    X,
    y,
    models
):

    model = models[
        "XGBoost"
    ]

    model.fit(
        X,
        y
    )

    importance = pd.DataFrame({

        "Feature":
            FEATURES,

        "Importance":
            model.feature_importances_

    })

    importance = (

        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    output_file = (

        OUTPUT_DIR /
        "ml_feature_importance_bf.csv"
    )

    importance.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )

    return importance


# =========================================================
# MODEL COMPARISON PLOT
# =========================================================

def plot_model_comparison(
    results
):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = [

        ("R2_mean", "R²"),

        ("RMSE_mean", "RMSE"),

        ("MAE_mean", "MAE")

    ]

    for column, label in metrics:

        plt.figure()

        plt.bar(
            results["Model"],
            results[column]
        )

        plt.ylabel(label)

        plt.xlabel("Model")

        plt.title(
            f"Machine Learning Model Comparison - {label}"
        )

        plt.xticks(
            rotation=20,
            ha="right"
        )

        plt.tight_layout()

        output_file = (

            PLOTS_DIR /
            f"ml_model_comparison_{column.lower()}_bf.png"
        )

        plt.savefig(

            output_file,

            dpi=300,

            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )


# =========================================================
# FEATURE IMPORTANCE PLOT
# =========================================================

def plot_feature_importance(
    importance
):

    plt.figure()

    plt.barh(

        importance["Feature"][::-1],

        importance["Importance"][::-1]
    )

    plt.xlabel(
        "XGBoost Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "XGBoost Feature Importance - BF"
    )

    plt.tight_layout()

    output_file = (

        PLOTS_DIR /
        "ml_feature_importance_bf.png"
    )

    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# =========================================================
# ACTUAL VS PREDICTED PLOT
# =========================================================

def plot_actual_vs_predicted(
    validation_results,
    model_column="BF_xgboost"
):

    if model_column not in validation_results.columns:

        print(
            f"Warning: {model_column} not found. "
            "Actual-vs-predicted plot skipped."
        )

        return

    y_true = validation_results[
        "BF_physical"
    ]

    y_pred = validation_results[
        model_column
    ]

    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )

    plt.figure()

    plt.scatter(

        y_true,

        y_pred,

        alpha=0.6
    )

    plt.plot(

        [min_value, max_value],

        [min_value, max_value],

        linestyle="--"
    )

    plt.xlabel(
        "BF - Physical Model"
    )

    plt.ylabel(
        "BF - XGBoost"
    )

    plt.title(
        "Independent Validation - XGBoost"
    )

    plt.tight_layout()

    output_file = (

        PLOTS_DIR /
        "ml_actual_vs_predicted_xgboost_bf.png"
    )

    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# =========================================================
# RESIDUAL PLOT
# =========================================================

def plot_residuals(
    validation_results,
    model_column="BF_xgboost"
):

    if model_column not in validation_results.columns:

        print(
            f"Warning: {model_column} not found. "
            "Residual plot skipped."
        )

        return

    y_true = validation_results[
        "BF_physical"
    ]

    y_pred = validation_results[
        model_column
    ]

    residuals = (
        y_pred - y_true
    )

    plt.figure()

    plt.scatter(

        y_true,

        residuals,

        alpha=0.6
    )

    plt.axhline(

        0,

        linestyle="--"
    )

    plt.xlabel(
        "BF - Physical Model"
    )

    plt.ylabel(
        "Residual (ML - Physical)"
    )

    plt.title(
        "Independent Validation - XGBoost Residuals"
    )

    plt.tight_layout()

    output_file = (

        PLOTS_DIR /
        "ml_residuals_xgboost_bf.png"
    )

    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# =========================================================
# INDEPENDENT VALIDATION - ALL MODELS
# =========================================================

def validate_models(
    models,
    X_train,
    y_train
):

    print(
        "\n========================================"
    )

    print(
        "INDEPENDENT SURROGATE VALIDATION"
    )

    print(
        "========================================"
    )

    print(
        f"Generating "
        f"{VALIDATION_SAMPLES} "
        f"independent samples..."
    )

    X_validation = (
        generate_validation_set()
    )

    print(
        "\nEvaluating physical model..."
    )

    y_physical = (
        evaluate_physical_model(
            X_validation
        )
    )

    valid = np.isfinite(
        y_physical
    )

    print(
        f"Valid physical evaluations: "
        f"{np.sum(valid)} / "
        f"{len(y_physical)}"
    )

    if not np.all(valid):

        X_validation = (

            X_validation.loc[
                valid
            ]

            .reset_index(
                drop=True
            )
        )

        y_physical = (
            y_physical[
                valid
            ]
        )

    validation_summary = []

    validation_predictions = {}

    for name, model in models.items():

        print(
            f"\nValidating: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_validation
        )

        r2 = r2_score(
            y_physical,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_physical,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_physical,
            y_pred
        )

        threshold = np.percentile(
            y_physical,
            90
        )

        high_bf = (
            y_physical >= threshold
        )

        r2_high = r2_score(
            y_physical[high_bf],
            y_pred[high_bf]
        )

        rmse_high = np.sqrt(
            mean_squared_error(
                y_physical[high_bf],
                y_pred[high_bf]
            )
        )

        mae_high = mean_absolute_error(
            y_physical[high_bf],
            y_pred[high_bf]
        )

        validation_summary.append({

            "Model":
                name,

            "R2":
                r2,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "Top10_R2":
                r2_high,

            "Top10_RMSE":
                rmse_high,

            "Top10_MAE":
                mae_high
        })

        validation_predictions[
            name
        ] = y_pred

    summary_df = pd.DataFrame(
        validation_summary
    )

    summary_df = (
        summary_df
        .sort_values(
            "RMSE"
        )
    )

    print(
        "\n========================================"
    )

    print(
        "INDEPENDENT VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        summary_df.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    validation_results = (
        X_validation.copy()
    )

    validation_results[
        "BF_physical"
    ] = y_physical

    for name, y_pred in (
        validation_predictions.items()
    ):

        safe_name = (

            name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        validation_results[
            f"BF_{safe_name}"
        ] = y_pred

    return (

        summary_df,

        validation_results
    )


# =========================================================
# VALIDATION PLOTS
# =========================================================

def plot_validation_results(
    validation_results
):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    y_true = validation_results[
        "BF_physical"
    ]

    model_columns = [

        column

        for column in validation_results.columns

        if column.startswith("BF_")

        and column != "BF_physical"
    ]

    for column in model_columns:

        plt.figure()

        plt.scatter(

            y_true,

            validation_results[column],

            alpha=0.6
        )

        min_value = min(

            y_true.min(),

            validation_results[
                column
            ].min()
        )

        max_value = max(

            y_true.max(),

            validation_results[
                column
            ].max()
        )

        plt.plot(

            [min_value, max_value],

            [min_value, max_value],

            linestyle="--"
        )

        model_name = (

            column
            .replace(
                "BF_",
                ""
            )
            .replace(
                "_",
                " "
            )
            .title()
        )

        plt.xlabel(
            "BF - Physical Model"
        )

        plt.ylabel(
            f"BF - {model_name}"
        )

        plt.title(
            f"Independent Validation - {model_name}"
        )

        plt.tight_layout()

        output_file = (

            PLOTS_DIR /

            f"ml_actual_vs_predicted_"
            f"{column.replace('BF_', '')}_bf.png"
        )

        plt.savefig(

            output_file,

            dpi=300,

            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )


# =========================================================
# TUNING 3 - ROBUSTNESS VALIDATION
# =========================================================

def validate_tuning3_robustness(
    X_train,
    y_train
):

    print("\n========================================")
    print("TUNING 3 - ROBUSTNESS VALIDATION")
    print("========================================")

    print(
        f"Generating {len(ROBUSTNESS_SEEDS)} independent "
        f"validation sets with {ROBUSTNESS_SAMPLES} samples each..."
    )

    model = XGBRegressor(
        **BEST_XGB_PARAMS,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    rows = []

    for i, seed in enumerate(ROBUSTNESS_SEEDS, start=1):

        print(
            f"\nValidating set {i}/{len(ROBUSTNESS_SEEDS)} "
            f"(seed={seed})..."
        )

        X_validation = generate_validation_set(
            seed=seed,
            samples=ROBUSTNESS_SAMPLES
        )

        y_physical = evaluate_physical_model(
            X_validation
        )

        valid = np.isfinite(y_physical)

        X_validation = (
            X_validation.loc[valid]
            .reset_index(drop=True)
        )

        y_physical = y_physical[valid]

        y_pred = model.predict(X_validation)

        r2 = r2_score(y_physical, y_pred)
        rmse = np.sqrt(
            mean_squared_error(y_physical, y_pred)
        )
        mae = mean_absolute_error(
            y_physical, y_pred
        )

        threshold = np.percentile(
            y_physical, 90
        )
        high_bf = y_physical >= threshold

        rmse_high = np.sqrt(
            mean_squared_error(
                y_physical[high_bf],
                y_pred[high_bf]
            )
        )

        mae_high = mean_absolute_error(
            y_physical[high_bf],
            y_pred[high_bf]
        )

        high_mu = X_validation["mu"] > 0.35

        rmse_mu = np.sqrt(
            mean_squared_error(
                y_physical[high_mu],
                y_pred[high_mu]
            )
        )

        mae_mu = mean_absolute_error(
            y_physical[high_mu],
            y_pred[high_mu]
        )

        rows.append({
            "Seed": seed,
            "Valid_Samples": len(y_physical),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Top10_RMSE": rmse_high,
            "Top10_MAE": mae_high,
            "MuHigh_RMSE": rmse_mu,
            "MuHigh_MAE": mae_mu
        })

    results_df = pd.DataFrame(rows)

    mean_row = {
        "Seed": "Mean",
        "Valid_Samples": results_df["Valid_Samples"].mean(),
        "R2": results_df["R2"].mean(),
        "RMSE": results_df["RMSE"].mean(),
        "MAE": results_df["MAE"].mean(),
        "Top10_RMSE": results_df["Top10_RMSE"].mean(),
        "Top10_MAE": results_df["Top10_MAE"].mean(),
        "MuHigh_RMSE": results_df["MuHigh_RMSE"].mean(),
        "MuHigh_MAE": results_df["MuHigh_MAE"].mean()
    }

    std_row = {
        "Seed": "Std",
        "Valid_Samples": results_df["Valid_Samples"].std(ddof=1),
        "R2": results_df["R2"].std(ddof=1),
        "RMSE": results_df["RMSE"].std(ddof=1),
        "MAE": results_df["MAE"].std(ddof=1),
        "Top10_RMSE": results_df["Top10_RMSE"].std(ddof=1),
        "Top10_MAE": results_df["Top10_MAE"].std(ddof=1),
        "MuHigh_RMSE": results_df["MuHigh_RMSE"].std(ddof=1),
        "MuHigh_MAE": results_df["MuHigh_MAE"].std(ddof=1)
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame([mean_row, std_row])
        ],
        ignore_index=True
    )

    print("\n========================================")
    print("TUNING 3 ROBUSTNESS RESULTS")
    print("========================================")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    output_file = (
        OUTPUT_DIR /
        "ml_tuning3_robustness_bf.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )

    return results_df


# =========================================================
# TUNING 1
# max_depth
# =========================================================

def objective_max_depth(
    trial,
    X,
    y
):

    max_depth = trial.suggest_int(

        "max_depth",

        2,

        8
    )

    model = XGBRegressor(

        n_estimators=300,

        max_depth=max_depth,

        learning_rate=0.05,

        subsample=0.90,

        colsample_bytree=0.90,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    cv = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_SEED
    )

    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1
    )

    rmse = -scores[
        "test_score"
    ].mean()

    print(

        f"max_depth={max_depth} "
        f"→ RMSE={rmse:.6f}"
    )

    return rmse


# =========================================================
# TUNING 2
# learning_rate + n_estimators
# =========================================================

def objective_xgboost(
    trial,
    X,
    y
):

    learning_rate = trial.suggest_float(

        "learning_rate",

        0.01,

        0.10,

        log=True
    )

    n_estimators = trial.suggest_int(

        "n_estimators",

        100,

        800,

        step=50
    )

    model = XGBRegressor(

        n_estimators=n_estimators,

        max_depth=3,

        learning_rate=learning_rate,

        subsample=0.90,

        colsample_bytree=0.90,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    cv = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_SEED
    )

    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1
    )

    rmse = -scores[
        "test_score"
    ].mean()

    print(

        f"learning_rate={learning_rate:.5f} | "

        f"n_estimators={n_estimators} | "

        f"RMSE={rmse:.6f}"
    )

    return rmse


# =========================================================
# TUNING 3
# subsample + colsample_bytree
# =========================================================

def objective_xgboost_sampling(
    trial,
    X,
    y
):

    subsample = trial.suggest_float(

        "subsample",

        0.60,

        1.00
    )

    colsample_bytree = trial.suggest_float(

        "colsample_bytree",

        0.60,

        1.00
    )

    model = XGBRegressor(

        n_estimators=800,

        max_depth=3,

        learning_rate=0.0609418819,

        subsample=subsample,

        colsample_bytree=colsample_bytree,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    cv = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_SEED
    )

    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1
    )

    rmse = -scores[
        "test_score"
    ].mean()

    print(

        f"subsample={subsample:.4f} | "

        f"colsample_bytree={colsample_bytree:.4f} | "

        f"RMSE={rmse:.6f}"
    )

    return rmse


# =========================================================
# TUNING 4
# min_child_weight + gamma
# =========================================================

def objective_xgboost_regularization(
    trial,
    X,
    y
):

    min_child_weight = trial.suggest_int(

        "min_child_weight",

        1,

        10
    )

    gamma = trial.suggest_float(

        "gamma",

        0.0,

        1.0
    )

    model = XGBRegressor(

        n_estimators=800,

        max_depth=3,

        learning_rate=0.0609418819,

        subsample=0.6030189277,

        colsample_bytree=0.9852181122,

        min_child_weight=min_child_weight,

        gamma=gamma,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    cv = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_SEED
    )

    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1
    )

    rmse = -scores[
        "test_score"
    ].mean()

    print(

        f"min_child_weight="
        f"{min_child_weight} | "

        f"gamma="
        f"{gamma:.5f} | "

        f"RMSE="
        f"{rmse:.6f}"
    )

    return rmse


# =========================================================
# TUNING 5
# GLOBAL XGBOOST HYPERPARAMETER OPTIMIZATION
# =========================================================

def objective_xgboost_global(
    trial,
    X,
    y
):
    """
    Global Optuna search over the main XGBoost hyperparameters.

    This stage is intentionally independent from Tunings 1-4.
    The objective uses only 5-fold cross-validation on the
    training dataset. The independent validation set is kept
    completely outside the Optuna optimization.
    """

    max_depth = trial.suggest_int(
        "max_depth",
        2,
        5
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        0.02,
        0.10,
        log=True
    )

    n_estimators = trial.suggest_int(
        "n_estimators",
        300,
        1000,
        step=50
    )

    subsample = trial.suggest_float(
        "subsample",
        0.60,
        1.00
    )

    colsample_bytree = trial.suggest_float(
        "colsample_bytree",
        0.80,
        1.00
    )

    min_child_weight = trial.suggest_int(
        "min_child_weight",
        1,
        10
    )

    gamma = trial.suggest_float(
        "gamma",
        0.0,
        0.10
    )

    reg_alpha = trial.suggest_float(
        "reg_alpha",
        0.0,
        0.10
    )

    reg_lambda = trial.suggest_float(
        "reg_lambda",
        0.50,
        2.00
    )

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    rmse = -scores["test_score"].mean()

    print(
        f"depth={max_depth} | "
        f"lr={learning_rate:.5f} | "
        f"trees={n_estimators} | "
        f"sub={subsample:.4f} | "
        f"col={colsample_bytree:.4f} | "
        f"mcw={min_child_weight} | "
        f"gamma={gamma:.5f} | "
        f"alpha={reg_alpha:.5f} | "
        f"lambda={reg_lambda:.5f} | "
        f"RMSE={rmse:.6f}"
    )

    return rmse


# =========================================================
# COMPARE BASELINE VS CURRENT BEST
# =========================================================

def compare_xgboost_optimized(
    X_train,
    y_train,
    validation_results,
    optimized_params
):

    print(
        "\n========================================"
    )

    print(
        "XGBOOST BASELINE VS CURRENT BEST"
    )

    print(
        "========================================"
    )

    y_true = validation_results[
        "BF_physical"
    ]

    X_validation = validation_results[
        FEATURES
    ]

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------

    baseline = XGBRegressor(

        **BASELINE_XGB_PARAMS,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    baseline.fit(

        X_train,

        y_train
    )

    pred_baseline = baseline.predict(
        X_validation
    )

    # -----------------------------------------------------
    # CURRENT BEST / CANDIDATE
    # -----------------------------------------------------

    optimized = XGBRegressor(

        **optimized_params,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    optimized.fit(

        X_train,

        y_train
    )

    pred_optimized = optimized.predict(
        X_validation
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    results = []

    threshold = np.percentile(

        y_true,

        90
    )

    high_bf = (

        y_true >= threshold
    )

    high_mu = (

        validation_results[
            "mu"
        ] > 0.35
    )

    for name, prediction in [

        ("Original", pred_baseline),

        ("Candidate", pred_optimized)

    ]:

        r2 = r2_score(

            y_true,

            prediction
        )

        rmse = np.sqrt(

            mean_squared_error(

                y_true,

                prediction
            )
        )

        mae = mean_absolute_error(

            y_true,

            prediction
        )

        r2_high = r2_score(

            y_true[high_bf],

            prediction[high_bf]
        )

        rmse_high = np.sqrt(

            mean_squared_error(

                y_true[high_bf],

                prediction[high_bf]
            )
        )

        mae_high = mean_absolute_error(

            y_true[high_bf],

            prediction[high_bf]
        )

        rmse_mu = np.sqrt(

            mean_squared_error(

                y_true[high_mu],

                prediction[high_mu]
            )
        )

        mae_mu = mean_absolute_error(

            y_true[high_mu],

            prediction[high_mu]
        )

        results.append({

            "Model":
                name,

            "R2":
                r2,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "Top10_R2":
                r2_high,

            "Top10_RMSE":
                rmse_high,

            "Top10_MAE":
                mae_high,

            "MuHigh_RMSE":
                rmse_mu,

            "MuHigh_MAE":
                mae_mu
        })

    comparison = pd.DataFrame(
        results
    )

    print(

        comparison.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    return comparison



# =========================================================
# ROBUST COMPARISON
# TUNING 2 vs TUNING 3 vs TUNING 5
# =========================================================

TUNING_2_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.0609418819,
    "n_estimators": 800,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0
}


TUNING_3_PARAMS = {
    **BEST_XGB_PARAMS,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0
}


# Parameters obtained in the completed Stage 5 global search.
# These are kept separately so that the robustness experiment
# does not depend on re-running Optuna.
TUNING_5_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05950,
    "n_estimators": 950,
    "subsample": 0.70360,
    "colsample_bytree": 0.89572,
    "min_child_weight": 3,
    "gamma": 0.00456,
    "reg_alpha": 0.02811,
    "reg_lambda": 1.71280
}


def compare_tuning_robustness(
    X_train,
    y_train
):
    """
    Compare Tuning 2, Tuning 3 and Tuning 5 on exactly the
    same independent validation sets.

    No validation set is used during training or Optuna.
    Each model is fitted once on the complete 1000-sample
    training dataset and then evaluated on five independent
    sets of 500 samples.
    """

    print("\n========================================")
    print("ROBUST COMPARISON")
    print("TUNING 2 vs TUNING 3 vs TUNING 5")
    print("========================================")

    models_params = {
        "Tuning 2": TUNING_2_PARAMS,
        "Tuning 3": TUNING_3_PARAMS,
        "Tuning 5": TUNING_5_PARAMS
    }

    detailed_rows = []

    # -----------------------------------------------------
    # Generate the independent datasets ONCE.
    # This guarantees that every model is evaluated on
    # exactly the same points.
    # -----------------------------------------------------

    validation_sets = {}

    print(
        f"\nGenerating {len(ROBUSTNESS_SEEDS)} common "
        f"validation sets with "
        f"{ROBUSTNESS_SAMPLES} samples each..."
    )

    for seed in ROBUSTNESS_SEEDS:

        print(
            f"  Generating seed={seed}"
        )

        X_validation = generate_validation_set(
            seed=seed,
            samples=ROBUSTNESS_SAMPLES
        )

        y_physical = evaluate_physical_model(
            X_validation
        )

        valid = np.isfinite(
            y_physical
        )

        X_validation = (
            X_validation.loc[valid]
            .reset_index(drop=True)
        )

        y_physical = y_physical[valid]

        validation_sets[seed] = (
            X_validation,
            y_physical
        )

    # -----------------------------------------------------
    # Evaluate every model on every common dataset.
    # -----------------------------------------------------

    for model_name, params in models_params.items():

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Evaluating: {model_name}"
        )

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1
        )

        model.fit(
            X_train,
            y_train
        )

        for seed in ROBUSTNESS_SEEDS:

            X_validation, y_physical = (
                validation_sets[seed]
            )

            y_pred = model.predict(
                X_validation
            )

            # ---------------------------------------------
            # GLOBAL METRICS
            # ---------------------------------------------

            r2 = r2_score(
                y_physical,
                y_pred
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_physical,
                    y_pred
                )
            )

            mae = mean_absolute_error(
                y_physical,
                y_pred
            )

            # ---------------------------------------------
            # TOP 10% BF
            # ---------------------------------------------

            threshold = np.percentile(
                y_physical,
                90
            )

            high_bf = (
                y_physical >= threshold
            )

            r2_high = r2_score(
                y_physical[high_bf],
                y_pred[high_bf]
            )

            rmse_high = np.sqrt(
                mean_squared_error(
                    y_physical[high_bf],
                    y_pred[high_bf]
                )
            )

            mae_high = mean_absolute_error(
                y_physical[high_bf],
                y_pred[high_bf]
            )

            # ---------------------------------------------
            # HIGH MU
            # ---------------------------------------------

            high_mu = (
                X_validation["mu"] > 0.35
            )

            r2_mu = r2_score(
                y_physical[high_mu],
                y_pred[high_mu]
            )

            rmse_mu = np.sqrt(
                mean_squared_error(
                    y_physical[high_mu],
                    y_pred[high_mu]
                )
            )

            mae_mu = mean_absolute_error(
                y_physical[high_mu],
                y_pred[high_mu]
            )

            detailed_rows.append({

                "Model":
                    model_name,

                "Seed":
                    seed,

                "Valid_Samples":
                    len(y_physical),

                "R2":
                    r2,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "Top10_R2":
                    r2_high,

                "Top10_RMSE":
                    rmse_high,

                "Top10_MAE":
                    mae_high,

                "MuHigh_R2":
                    r2_mu,

                "MuHigh_RMSE":
                    rmse_mu,

                "MuHigh_MAE":
                    mae_mu
            })

    detailed_df = pd.DataFrame(
        detailed_rows
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary_df = (
        detailed_df
        .groupby("Model")
        .agg(

            R2_mean=(
                "R2",
                "mean"
            ),

            R2_std=(
                "R2",
                "std"
            ),

            RMSE_mean=(
                "RMSE",
                "mean"
            ),

            RMSE_std=(
                "RMSE",
                "std"
            ),

            MAE_mean=(
                "MAE",
                "mean"
            ),

            MAE_std=(
                "MAE",
                "std"
            ),

            Top10_R2_mean=(
                "Top10_R2",
                "mean"
            ),

            Top10_RMSE_mean=(
                "Top10_RMSE",
                "mean"
            ),

            Top10_RMSE_std=(
                "Top10_RMSE",
                "std"
            ),

            Top10_MAE_mean=(
                "Top10_MAE",
                "mean"
            ),

            MuHigh_R2_mean=(
                "MuHigh_R2",
                "mean"
            ),

            MuHigh_RMSE_mean=(
                "MuHigh_RMSE",
                "mean"
            ),

            MuHigh_RMSE_std=(
                "MuHigh_RMSE",
                "std"
            ),

            MuHigh_MAE_mean=(
                "MuHigh_MAE",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "RMSE_mean"
        )
    )

    # =====================================================
    # PRINT
    # =====================================================

    print("\n========================================")
    print("ROBUST COMPARISON RESULTS")
    print("========================================")

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # =====================================================
    # SAVE
    # =====================================================

    detailed_file = (
        OUTPUT_DIR /
        "ml_tuning_robust_comparison_bf.csv"
    )

    summary_file = (
        OUTPUT_DIR /
        "ml_tuning_robust_comparison_summary_bf.csv"
    )

    detailed_df.to_csv(
        detailed_file,
        index=False
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        f"\nSaved: {detailed_file}"
    )

    print(
        f"Saved: {summary_file}"
    )

    return (
        detailed_df,
        summary_df
    )


# =========================================================
# SELECT TUNING OBJECTIVE
# =========================================================

def get_tuning_objective():

    if TUNING_STAGE == 1:

        return objective_max_depth

    elif TUNING_STAGE == 2:

        return objective_xgboost

    elif TUNING_STAGE == 3:

        return objective_xgboost_sampling

    elif TUNING_STAGE == 4:

        return objective_xgboost_regularization

    elif TUNING_STAGE == 5:

        return objective_xgboost_global

    else:

        raise ValueError(
            f"Unsupported TUNING_STAGE: "
            f"{TUNING_STAGE}"
        )


# =========================================================
# TUNING MAIN
# =========================================================

def tuning_main():

    df = load_dataset()

    X, y = prepare_data(
        df
    )

    print(
        "\n========================================"
    )

    print(
        f"XGBOOST HYPERPARAMETER TUNING "
        f"- STAGE {TUNING_STAGE}"
    )

    print(
        "========================================"
    )

    objective = (
        get_tuning_objective()
    )

    # =====================================================
    # OPTUNA
    # =====================================================

    study = optuna.create_study(

        direction="minimize",

        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_SEED
        )
    )

    study.optimize(

        lambda trial:
            objective(
                trial,
                X,
                y
            ),

        n_trials=N_TRIALS
    )

    print(
        "\n========================================"
    )

    print(
        "OPTUNA RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Stage: {TUNING_STAGE}"
    )

    print(
        f"Best parameters: "
        f"{study.best_params}"
    )

    print(
        f"Best RMSE: "
        f"{study.best_value:.6f}"
    )

    # =====================================================
    # INDEPENDENT VALIDATION SET
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "INDEPENDENT VALIDATION"
    )

    print(
        "========================================"
    )

    X_validation = (
        generate_validation_set()
    )

    y_physical = (
        evaluate_physical_model(
            X_validation
        )
    )

    valid = np.isfinite(
        y_physical
    )

    X_validation = (

        X_validation.loc[
            valid
        ]

        .reset_index(
            drop=True
        )
    )

    y_physical = (
        y_physical[
            valid
        ]
    )

    validation_results = (
        X_validation.copy()
    )

    validation_results[
        "BF_physical"
    ] = y_physical

    # =====================================================
    # BUILD CANDIDATE PARAMETERS
    # =====================================================

    if TUNING_STAGE == 1:

        candidate_params = {

            **BASELINE_XGB_PARAMS,

            **study.best_params
        }

    elif TUNING_STAGE == 2:

        candidate_params = {

            **BEST_XGB_PARAMS,

            **study.best_params
        }

    elif TUNING_STAGE == 3:

        candidate_params = {

            **BEST_XGB_PARAMS,

            **study.best_params
        }

    elif TUNING_STAGE == 4:

        candidate_params = {

            **BEST_XGB_PARAMS,

            **study.best_params
        }

    elif TUNING_STAGE == 5:

        # Global search is independent: the candidate is built
        # entirely from Optuna's best global parameter set.
        candidate_params = {

            **study.best_params
        }

    else:

        raise ValueError(
            f"Unsupported TUNING_STAGE: "
            f"{TUNING_STAGE}"
        )

    print(
        "\nCandidate XGBoost parameters:"
    )

    print(
        candidate_params
    )

    # =====================================================
    # INDEPENDENT VALIDATION
    # =====================================================

    comparison = (
        compare_xgboost_optimized(

            X,

            y,

            validation_results,

            candidate_params
        )
    )

    # =====================================================
    # SAVE TUNING RESULT
    # =====================================================

    # Extract the candidate metrics from the comparison table
    # so the global tuning result remains traceable.
    candidate_row = comparison[
        comparison["Model"] == "Candidate"
    ].iloc[0]

    tuning_result = {

        "Stage":
            TUNING_STAGE,

        "Best_CV_RMSE":
            study.best_value,

        "Independent_R2":
            candidate_row["R2"],

        "Independent_RMSE":
            candidate_row["RMSE"],

        "Independent_MAE":
            candidate_row["MAE"],

        "Top10_R2":
            candidate_row["Top10_R2"],

        "Top10_RMSE":
            candidate_row["Top10_RMSE"],

        "Top10_MAE":
            candidate_row["Top10_MAE"],

        "MuHigh_RMSE":
            candidate_row["MuHigh_RMSE"],

        "MuHigh_MAE":
            candidate_row["MuHigh_MAE"],

        **study.best_params

    }

    tuning_file = (

        OUTPUT_DIR /
        f"xgboost_tuning_stage_"
        f"{TUNING_STAGE}.csv"
    )

    pd.DataFrame([
        tuning_result
    ]).to_csv(

        tuning_file,

        index=False
    )

    print(
        f"\nSaved tuning result: "
        f"{tuning_file}"
    )

    # =====================================================
    # ROBUSTNESS VALIDATION OF TUNING 3
    # =====================================================

    validate_tuning3_robustness(
        X,
        y
    )

    # =====================================================
    # ROBUST COMPARISON OF FINAL CANDIDATES
    # =====================================================

    compare_tuning_robustness(
        X,
        y
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    tuning_main()