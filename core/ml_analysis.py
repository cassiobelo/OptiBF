import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# CONFIGURATION
# =========================================================

DATA_FILE = (
    "data/optibf_dataset_lhs_1000_robustness.csv"
)

OUTPUT_DIR = Path("data")
PLOTS_DIR = OUTPUT_DIR / "ml_plots"

RANDOM_SEED = 42

N_SPLITS = 5


# =========================================================
# INDEPENDENT VALIDATION CONFIGURATION
# =========================================================

VALIDATION_SAMPLES = 500

VALIDATION_SEED = 2026


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

    print("========================================")
    print("OptiBF - Machine Learning Analysis")
    print("========================================")

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

def generate_validation_set():

    rng = np.random.default_rng(
        VALIDATION_SEED
    )

    data = {}

    for feature in FEATURES:

        lower, upper = BOUNDS[
            feature
        ]

        data[feature] = rng.uniform(
            lower,
            upper,
            VALIDATION_SAMPLES
        )

    return pd.DataFrame(
        data
    )


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
    # XGBOOST
    # -----------------------------------------------------

    xgboost = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
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
                hidden_layer_sizes=(64, 64),
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
    # SUPPORT VECTOR REGRESSION (SVM)
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

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

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

        r2_mean = (
            scores[
                "test_r2"
            ].mean()
        )

        r2_std = (
            scores[
                "test_r2"
            ].std()
        )

        rmse_mean = (
            -scores[
                "test_rmse"
            ].mean()
        )

        rmse_std = (
            scores[
                "test_rmse"
            ].std()
        )

        mae_mean = (
            -scores[
                "test_mae"
            ].mean()
        )

        mae_std = (
            scores[
                "test_mae"
            ].std()
        )

        results.append({

            "Model":
                name,

            "R2_mean":
                r2_mean,

            "R2_std":
                r2_std,

            "RMSE_mean":
                rmse_mean,

            "RMSE_std":
                rmse_std,

            "MAE_mean":
                mae_mean,

            "MAE_std":
                mae_std
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "RMSE_mean"
        )
    )

    return results_df


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    results
):

    print("\n========================================")
    print("CROSS-VALIDATION RESULTS")
    print("========================================")

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
    """
    Save cross-validation model comparison.
    """
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
    """
    Train XGBoost on the complete dataset and save
    feature importance values.
    """

    model = models["XGBoost"]

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
    """
    Generate one figure with the three main CV metrics.
    """

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
# RESIDUAL PLOT
# =========================================================

def plot_residuals(
    validation_results,
    model_column="BF_xgboost"
):
    """
    Plot prediction residuals for XGBoost.
    """

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

    residuals = y_pred - y_true

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
# FEATURE IMPORTANCE PLOT
# =========================================================

def plot_feature_importance(
    importance
):
    """
    Plot XGBoost feature importance.
    """

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
    """
    Plot physical BF against XGBoost prediction.
    """

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
# INDEPENDENT SURROGATE VALIDATION
# =========================================================

def validate_models(
    models,
    X_train,
    y_train
):

    print("\n========================================")
    print("INDEPENDENT SURROGATE VALIDATION")
    print("========================================")

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

    # =====================================================
    # EVALUATE EACH MODEL
    # =====================================================

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

        # -------------------------------------------------
        # GLOBAL METRICS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # TOP 10% BF
        # -------------------------------------------------

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

    # =====================================================
    # SUMMARY
    # =====================================================

    summary_df = pd.DataFrame(
        validation_summary
    )

    summary_df = (
        summary_df
        .sort_values(
            "RMSE"
        )
    )

    print("\n========================================")
    print("INDEPENDENT VALIDATION RESULTS")
    print("========================================")

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

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
    """
    Generate and save physical-vs-predicted plots for all models.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    y_true = (
        validation_results[
            "BF_physical"
        ]
    )

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
            validation_results[column].min()
        )

        max_value = max(
            y_true.max(),
            validation_results[column].max()
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
            f"ml_actual_vs_predicted_{column.replace('BF_', '')}_bf.png"
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
# MAIN
# =========================================================

def main():

    df = load_dataset()

    X, y = prepare_data(
        df
    )

    print("\nFeatures:")

    print(
        FEATURES
    )

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        f"Number of samples: "
        f"{len(X)}"
    )

    models = define_models()

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    results = evaluate_models(
        X,
        y,
        models
    )

    print_results(
        results
    )

    save_results(
        results
    )

    # =====================================================
    # INDEPENDENT VALIDATION
    # =====================================================

    validation_summary, validation_results = (
        validate_models(
            models,
            X,
            y
        )
    )

    validation_summary_file = (
        OUTPUT_DIR /
        "ml_independent_validation_summary_bf.csv"
    )

    validation_summary.to_csv(
        validation_summary_file,
        index=False
    )

    print(
        f"\nSaved: "
        f"{validation_summary_file}"
    )

    validation_file = (
        OUTPUT_DIR /
        "ml_independent_validation_bf.csv"
    )

    validation_results.to_csv(
        validation_file,
        index=False
    )

    print(
        f"\nSaved: "
        f"{validation_file}"
    )

    plot_validation_results(
        validation_results
    )

    # =====================================================
    # ADDITIONAL RESULTS FOR REPORTING
    # =====================================================

    importance = save_feature_importance(
        X,
        y,
        models
    )

    plot_model_comparison(
        results
    )

    plot_feature_importance(
        importance
    )

    plot_actual_vs_predicted(
        validation_results
    )

    plot_residuals(
        validation_results
    )

    print("\n========================================")
    print("ML RESULTS EXPORT COMPLETED")
    print("========================================")
    print(
        f"CSV results: {OUTPUT_DIR}"
    )
    print(
        f"Figures: {PLOTS_DIR}"
    )


if __name__ == "__main__":

    main()