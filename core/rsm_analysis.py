import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from core.brake_model import (
    RD,
    calculate_bf,
    calculate_bf_asymmetry
)


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = DATA_DIR / "rsm"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DATASETS
# =========================================================

BF_DATASET = (
    DATA_DIR /
    "optibf_dataset_lhs_1000.csv"
)

DELTA_DATASET = (
    DATA_DIR /
    "optibf_dataset_lhs_1000_robustness.csv"
)


# =========================================================
# VARIABLES
# =========================================================

FEATURES = [

    "L1_mm",

    "L3_mm",

    "L4_mm",

    "theta1_deg",

    "theta2_deg",

    "mu"
]


# =========================================================
# REFERENCE
# =========================================================

REFERENCE = {

    "L1_mm": 40.0,

    "L3_mm": 143.15,

    "L4_mm": 157.0,

    "theta1_deg": 28.0,

    "theta2_deg": 145.0,

    "mu": 0.30
}


# =========================================================
# RSM MODEL
# =========================================================

def create_rsm():

    return (

        PolynomialFeatures(

            degree=2,

            include_bias=False
        ),

        LinearRegression()
    )


# =========================================================
# TRAIN RSM
# =========================================================

def train_rsm(
    df,
    target
):

    X = df[
        FEATURES
    ]

    y = df[
        target
    ]

    polynomial, model = create_rsm()

    X_poly = polynomial.fit_transform(
        X
    )

    model.fit(
        X_poly,
        y
    )

    return (
        polynomial,
        model
    )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "R2":
            r2_score(
                y_true,
                y_pred
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred
            )
    }


# =========================================================
# CROSS VALIDATION
# =========================================================

def cross_validate_rsm(
    df,
    target,
    n_splits=5
):

    from sklearn.model_selection import KFold

    X = df[
        FEATURES
    ]

    y = df[
        target
    ]

    kfold = KFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=42
    )

    results = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        kfold.split(X),
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        polynomial, model = (
            create_rsm()
        )

        X_train_poly = (
            polynomial.fit_transform(
                X_train
            )
        )

        X_test_poly = (
            polynomial.transform(
                X_test
            )
        )

        model.fit(
            X_train_poly,
            y_train
        )

        prediction = (
            model.predict(
                X_test_poly
            )
        )

        metrics = (
            calculate_metrics(
                y_test,
                prediction
            )
        )

        metrics[
            "Fold"
        ] = fold

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    summary = pd.DataFrame({

        "Metric": [

            "R2",

            "RMSE",

            "MAE"
        ],

        "Mean": [

            results_df[
                "R2"
            ].mean(),

            results_df[
                "RMSE"
            ].mean(),

            results_df[
                "MAE"
            ].mean()
        ],

        "Std": [

            results_df[
                "R2"
            ].std(),

            results_df[
                "RMSE"
            ].std(),

            results_df[
                "MAE"
            ].std()
        ]
    })

    return (
        results_df,
        summary
    )


# =========================================================
# INDEPENDENT VALIDATION
# =========================================================

def independent_validation(
    model,
    polynomial,
    validation_df,
    target
):

    X = validation_df[
        FEATURES
    ]

    y = validation_df[
        target
    ]

    X_poly = (
        polynomial.transform(
            X
        )
    )

    prediction = (
        model.predict(
            X_poly
        )
    )

    metrics = calculate_metrics(
        y,
        prediction
    )

    # Relative error
    relative_error = (

        np.abs(
            prediction - y
        )

        /

        np.abs(y)

    ) * 100.0

    metrics[
        "Mean_Absolute_Error_Percent"
    ] = relative_error.mean()

    metrics[
        "Max_Absolute_Error_Percent"
    ] = relative_error.max()

    result = validation_df.copy()

    result[
        "RSM_Predicted"
    ] = prediction

    result[
        "Absolute_Error"
    ] = np.abs(
        prediction - y
    )

    result[
        "Absolute_Error_Percent"
    ] = relative_error

    return (
        metrics,
        result
    )


# =========================================================
# PHYSICAL REFERENCE
# =========================================================

def evaluate_reference():

    r = REFERENCE

    bf = calculate_bf(

        L1=r["L1_mm"] / 1000,

        L3=r["L3_mm"] / 1000,

        L4=r["L4_mm"] / 1000,

        theta1=math.radians(
            r["theta1_deg"]
        ),

        theta2=math.radians(
            r["theta2_deg"]
        ),

        mu=r["mu"],

        RD=RD
    )

    delta = (
        calculate_bf_asymmetry(

            L1=r["L1_mm"] / 1000,

            L3=r["L3_mm"] / 1000,

            L4=r["L4_mm"] / 1000,

            theta1=math.radians(
                r["theta1_deg"]
            ),

            theta2=math.radians(
                r["theta2_deg"]
            ),

            mu=r["mu"],

            delta=0.10,

            RD=RD
        )
    )

    return {

        "BF":
            bf["BF"],

        "Delta_BF":
            delta[
                "delta_BF_percent"
            ],

        "CL":
            bf["CL"],

        "CT":
            bf["CT"],

        "CL_CT":
            bf["CL_CT"]
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - RESPONSE SURFACE METHODOLOGY"
    )

    print(
        "========================================"
    )


    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    bf_df = pd.read_csv(
        BF_DATASET
    )

    delta_df = pd.read_csv(
        DELTA_DATASET
    )

    print(
        f"BF samples: "
        f"{len(bf_df)}"
    )

    print(
        f"Delta BF samples: "
        f"{len(delta_df)}"
    )


    # =====================================================
    # BF RSM
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "RSM — BRAKE FACTOR"
    )

    print(
        "========================================"
    )

    bf_folds, bf_cv = (
        cross_validate_rsm(

            bf_df,

            "BF"
        )
    )

    print(
        "\n5-Fold Cross Validation"
    )

    print(
        bf_cv.to_string(
            index=False
        )
    )


    bf_poly, bf_model = (
        train_rsm(

            bf_df,

            "BF"
        )
    )


    # =====================================================
    # DELTA BF RSM
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "RSM — DELTA BF"
    )

    print(
        "========================================"
    )

    delta_folds, delta_cv = (
        cross_validate_rsm(

            delta_df,

            "Delta_BF_percent"
        )
    )

    print(
        "\n5-Fold Cross Validation"
    )

    print(
        delta_cv.to_string(
            index=False
        )
    )


    delta_poly, delta_model = (
        train_rsm(

            delta_df,

            "Delta_BF_percent"
        )
    )


    # =====================================================
    # SAVE MODELS COEFFICIENTS
    # =====================================================

    bf_coefficients = pd.DataFrame({

        "term":
            bf_poly.get_feature_names_out(
                FEATURES
            ),

        "coefficient":
            bf_model.coef_
    })

    bf_coefficients.loc[
        len(bf_coefficients)
    ] = [
        "intercept",
        bf_model.intercept_
    ]

    bf_coefficients.to_csv(

        OUTPUT_DIR /
        "rsm_bf_coefficients.csv",

        index=False
    )


    delta_coefficients = pd.DataFrame({

        "term":
            delta_poly.get_feature_names_out(
                FEATURES
            ),

        "coefficient":
            delta_model.coef_
    })

    delta_coefficients.loc[
        len(delta_coefficients)
    ] = [
        "intercept",
        delta_model.intercept_
    ]

    delta_coefficients.to_csv(

        OUTPUT_DIR /
        "rsm_delta_bf_coefficients.csv",

        index=False
    )


    # =====================================================
    # REFERENCE
    # =====================================================

    reference = (
        evaluate_reference()
    )

    print(
        "\n========================================"
    )

    print(
        "REFERENCE PHYSICAL MODEL"
    )

    print(
        "========================================"
    )

    for key, value in reference.items():

        print(
            f"{key:10s} = "
            f"{value:.8f}"
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    summary_rows = []

    for metric in [
        "R2",
        "RMSE",
        "MAE"
    ]:

        row = {

            "Model":
                "RSM",

            "Target":
                "BF",

            "Metric":
                metric,

            "CV_Mean":
                bf_cv.loc[
                    bf_cv[
                        "Metric"
                    ] == metric,
                    "Mean"
                ].iloc[0],

            "CV_Std":
                bf_cv.loc[
                    bf_cv[
                        "Metric"
                    ] == metric,
                    "Std"
                ].iloc[0]
        }

        summary_rows.append(
            row
        )


    for metric in [
        "R2",
        "RMSE",
        "MAE"
    ]:

        row = {

            "Model":
                "RSM",

            "Target":
                "Delta_BF",

            "Metric":
                metric,

            "CV_Mean":
                delta_cv.loc[
                    delta_cv[
                        "Metric"
                    ] == metric,
                    "Mean"
                ].iloc[0],

            "CV_Std":
                delta_cv.loc[
                    delta_cv[
                        "Metric"
                    ] == metric,
                    "Std"
                ].iloc[0]
        }

        summary_rows.append(
            row
        )


    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(

        OUTPUT_DIR /
        "rsm_cv_summary.csv",

        index=False
    )


    print(
        "\n========================================"
    )

    print(
        "RSM MODELING COMPLETED"
    )

    print(
        "========================================"
    )

    print(
        "\nFiles saved:"
    )

    print(
        OUTPUT_DIR /
        "rsm_bf_coefficients.csv"
    )

    print(
        OUTPUT_DIR /
        "rsm_delta_bf_coefficients.csv"
    )

    print(
        OUTPUT_DIR /
        "rsm_cv_summary.csv"
    )


if __name__ == "__main__":

    main()