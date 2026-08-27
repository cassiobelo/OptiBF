import math
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from core.brake_model import (
    RD,
    calculate_bf_asymmetry
)


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    PROJECT_ROOT /
    "data"
)

# Dataset containing the robustness calculation
DATA_FILE = (
    DATA_DIR /
    "optibf_dataset_lhs_1000_robustness.csv"
)

OUTPUT_DIR = (
    DATA_DIR /
    "ml_optimization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# GENERAL CONFIGURATION
# =========================================================

RANDOM_SEED = 42

N_SPLITS = 5

VALIDATION_SAMPLES = 500

VALIDATION_SEED = 2026

# Friction asymmetry used in the project
ASYMMETRY_DELTA = 0.10

# Fixed friction coefficient for the first
# geometry-focused surrogate
FIXED_MU = 0.30


# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    "L1_mm",

    "L3_mm",

    "L4_mm",

    "theta1_deg",

    "theta2_deg",

    "mu"
]


TARGET = (
    "Delta_BF_percent"
)


# =========================================================
# TUNING 3 PARAMETERS
# =========================================================
#
# These parameters were selected during the BF
# optimization process.
#
# IMPORTANT:
# They are NOT assumed to be optimal for Delta BF.
#
# This is only the first surrogate candidate.
# =========================================================

TUNING3_PARAMS = {

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
# LOAD DATASET
# =========================================================

def load_dataset():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - DELTA BF SURROGATE"
    )

    print(
        "========================================"
    )

    print(
        f"\nDataset:"
    )

    print(
        DATA_FILE
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nSamples loaded: "
        f"{len(df)}"
    )

    print(
        "\nAvailable columns:"
    )

    print(
        df.columns.tolist()
    )

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    required_columns = (
        FEATURES +
        [TARGET]
    )

    missing = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            +
            "\n".join(missing)
        )


    data = df[
        required_columns
    ].copy()


    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    data = data.dropna()


    print(
        f"\nValid samples for Delta BF: "
        f"{len(data)}"
    )


    return data


# =========================================================
# TRAIN SURROGATE
# =========================================================

def train_surrogate(
    X,
    y
):

    print(
        "\n========================================"
    )

    print(
        "TRAINING DELTA BF SURROGATE"
    )

    print(
        "========================================"
    )


    model = XGBRegressor(

        **TUNING3_PARAMS,

        objective=
            "reg:squarederror",

        random_state=
            RANDOM_SEED,

        n_jobs=1
    )


    model.fit(
        X,
        y
    )


    print(
        "\nSurrogate trained."
    )


    return model


# =========================================================
# CROSS VALIDATION
# =========================================================

def cross_validation(
    X,
    y
):

    from sklearn.model_selection import (
        KFold,
        cross_validate
    )

    print(
        "\n========================================"
    )

    print(
        "5-FOLD CROSS VALIDATION"
    )

    print(
        "========================================"
    )


    model = XGBRegressor(

        **TUNING3_PARAMS,

        objective=
            "reg:squarederror",

        random_state=
            RANDOM_SEED,

        n_jobs=1
    )


    kfold = KFold(

        n_splits=
            N_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_SEED
    )


    scores = cross_validate(

        model,

        X,

        y,

        cv=kfold,

        scoring={

            "r2":
                "r2",

            "rmse":
                "neg_root_mean_squared_error",

            "mae":
                "neg_mean_absolute_error"
        },

        n_jobs=1
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


    result = pd.DataFrame({

        "Metric": [

            "R2",

            "RMSE",

            "MAE"
        ],

        "Mean": [

            r2_mean,

            rmse_mean,

            mae_mean
        ],

        "Std": [

            r2_std,

            rmse_std,

            mae_std
        ]
    })


    print(
        result.to_string(
            index=False,
            float_format=
                lambda x:
                f"{x:.6f}"
        )
    )


    return result


# =========================================================
# GENERATE INDEPENDENT VALIDATION SET
# =========================================================

def generate_validation_set():

    rng = np.random.default_rng(
        VALIDATION_SEED
    )


    data = {}


    # -----------------------------------------------------
    # Geometry
    # -----------------------------------------------------

    data["L1_mm"] = (
        rng.uniform(
            35.0,
            45.0,
            VALIDATION_SAMPLES
        )
    )


    data["L3_mm"] = (
        rng.uniform(
            138.15,
            143.15,
            VALIDATION_SAMPLES
        )
    )


    data["L4_mm"] = (
        rng.uniform(
            152.0,
            162.0,
            VALIDATION_SAMPLES
        )
    )


    data["theta1_deg"] = (
        rng.uniform(
            18.0,
            38.0,
            VALIDATION_SAMPLES
        )
    )


    data["theta2_deg"] = (
        rng.uniform(
            135.0,
            155.0,
            VALIDATION_SAMPLES
        )
    )


    # -----------------------------------------------------
    # Friction
    # -----------------------------------------------------

    data["mu"] = np.full(

        VALIDATION_SAMPLES,

        FIXED_MU
    )


    return pd.DataFrame(
        data
    )


# =========================================================
# EVALUATE PHYSICAL DELTA BF
# =========================================================

def evaluate_physical_delta_bf(
    X
):

    results = []


    for _, row in X.iterrows():

        result = (
            calculate_bf_asymmetry(

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

                theta1=math.radians(
                    row["theta1_deg"]
                ),

                theta2=math.radians(
                    row["theta2_deg"]
                ),

                mu=row["mu"],

                delta=
                    ASYMMETRY_DELTA,

                RD=RD
            )
        )


        if result is None:

            results.append(
                np.nan
            )

        else:

            results.append(

                result[
                    "delta_BF_percent"
                ]
            )


    return np.asarray(
        results,
        dtype=float
    )


# =========================================================
# INDEPENDENT VALIDATION
# =========================================================

def independent_validation(
    model
):

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


    y_true = (
        evaluate_physical_delta_bf(
            X_validation
        )
    )


    valid = np.isfinite(
        y_true
    )


    X_validation = (
        X_validation.loc[
            valid
        ]
        .reset_index(drop=True)
    )


    y_true = y_true[
        valid
    ]


    y_pred = model.predict(
        X_validation[
            FEATURES
        ]
    )


    r2 = r2_score(
        y_true,
        y_pred
    )


    rmse = np.sqrt(

        mean_squared_error(

            y_true,

            y_pred
        )
    )


    mae = mean_absolute_error(

        y_true,

        y_pred
    )


    # -----------------------------------------------------
    # Prediction error
    # -----------------------------------------------------

    absolute_error = (
        np.abs(
            y_pred -
            y_true
        )
    )


    relative_error = (

        absolute_error /

        np.abs(y_true)

    ) * 100.0


    result = pd.DataFrame({

        "Metric": [

            "R2",

            "RMSE",

            "MAE",

            "Mean_Absolute_Error_Percent",

            "Max_Absolute_Error_Percent"
        ],

        "Value": [

            r2,

            rmse,

            mae,

            relative_error.mean(),

            relative_error.max()
        ]
    })


    print(
        f"\nValid validation samples: "
        f"{len(y_true)}"
    )


    print(
        result.to_string(
            index=False,
            float_format=
                lambda x:
                f"{x:.6f}"
        )
    )


    # -----------------------------------------------------
    # Save point-by-point validation
    # -----------------------------------------------------

    validation_results = (
        X_validation.copy()
    )


    validation_results[
        "Delta_BF_physical"
    ] = y_true


    validation_results[
        "Delta_BF_ML"
    ] = y_pred


    validation_results[
        "Absolute_Error"
    ] = absolute_error


    validation_results[
        "Relative_Error_percent"
    ] = relative_error


    output_file = (

        OUTPUT_DIR /

        "ml_delta_bf_independent_validation.csv"
    )


    validation_results.to_csv(

        output_file,

        index=False
    )


    print(
        f"\nSaved validation:"
    )

    print(
        output_file
    )


    return result


# =========================================================
# SAVE SUMMARY
# =========================================================

def save_summary(
    cv_results,
    validation_results
):

    summary = pd.DataFrame({

        "Stage": [

            "Tuning 3 parameters applied "
            "to Delta BF"
        ],

        "max_depth": [

            TUNING3_PARAMS[
                "max_depth"
            ]
        ],

        "learning_rate": [

            TUNING3_PARAMS[
                "learning_rate"
            ]
        ],

        "n_estimators": [

            TUNING3_PARAMS[
                "n_estimators"
            ]
        ],

        "subsample": [

            TUNING3_PARAMS[
                "subsample"
            ]
        ],

        "colsample_bytree": [

            TUNING3_PARAMS[
                "colsample_bytree"
            ]
        ],

        "CV_R2": [

            cv_results.loc[
                cv_results["Metric"] == "R2",
                "Mean"
            ].iloc[0]
        ],

        "CV_RMSE": [

            cv_results.loc[
                cv_results["Metric"] == "RMSE",
                "Mean"
            ].iloc[0]
        ],

        "CV_MAE": [

            cv_results.loc[
                cv_results["Metric"] == "MAE",
                "Mean"
            ].iloc[0]
        ],

        "Independent_R2": [

            validation_results.loc[
                validation_results["Metric"] == "R2",
                "Value"
            ].iloc[0]
        ],

        "Independent_RMSE": [

            validation_results.loc[
                validation_results["Metric"] == "RMSE",
                "Value"
            ].iloc[0]
        ],

        "Independent_MAE": [

            validation_results.loc[
                validation_results["Metric"] == "MAE",
                "Value"
            ].iloc[0]
        ]
    })


    output_file = (

        OUTPUT_DIR /

        "ml_delta_bf_surrogate_summary.csv"
    )


    summary.to_csv(

        output_file,

        index=False
    )


    print(
        f"\nSaved summary:"
    )

    print(
        output_file
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - DELTA BF SURROGATE ANALYSIS"
    )

    print(
        "========================================"
    )


    print(
        f"\nFriction asymmetry:"
        f" ±{ASYMMETRY_DELTA * 100:.1f}%"
    )


    print(
        f"Fixed mu:"
        f" {FIXED_MU:.2f}"
    )


    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    df = load_dataset()


    # -----------------------------------------------------
    # PREPARE
    # -----------------------------------------------------

    data = prepare_data(
        df
    )


    X = data[
        FEATURES
    ]

    y = data[
        TARGET
    ]


    print(
        "\nDelta BF statistics:"
    )

    print(
        y.describe()
    )


    # -----------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------

    cv_results = cross_validation(
        X,
        y
    )


    # -----------------------------------------------------
    # TRAIN FINAL SURROGATE
    # -----------------------------------------------------

    model = train_surrogate(
        X,
        y
    )


    # -----------------------------------------------------
    # INDEPENDENT VALIDATION
    # -----------------------------------------------------

    validation_results = (
        independent_validation(
            model
        )
    )


    # -----------------------------------------------------
    # SAVE SUMMARY
    # -----------------------------------------------------

    save_summary(

        cv_results,

        validation_results
    )


    print(
        "\n========================================"
    )

    print(
        "DELTA BF SURROGATE COMPLETED"
    )

    print(
        "========================================"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()