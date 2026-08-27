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

RSM_DIR = DATA_DIR / "rsm"

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

# This is the 500-point independent validation set
# already used for the Delta BF XGBoost surrogate.
VALIDATION_FILE = (
    DATA_DIR /
    "ml_optimization" /
    "ml_delta_bf_independent_validation.csv"
)


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


# =========================================================
# RSM
# =========================================================

def create_rsm():

    polynomial = PolynomialFeatures(

        degree=2,

        include_bias=False
    )

    model = LinearRegression()

    return (
        polynomial,
        model
    )


# =========================================================
# TRAIN MODEL
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

    polynomial, model = (
        create_rsm()
    )

    X_poly = (
        polynomial.fit_transform(
            X
        )
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

    absolute_error = np.abs(
        y_pred -
        y_true
    )

    relative_error = (

        absolute_error /

        np.abs(y_true)

    ) * 100.0


    return {

        "R2":
            r2,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "Mean_Absolute_Error_Percent":
            relative_error.mean(),

        "Max_Absolute_Error_Percent":
            relative_error.max()
    }


# =========================================================
# PHYSICAL BF
# =========================================================

def calculate_physical_bf(
    validation
):

    values = []

    cl_values = []

    ct_values = []

    clct_values = []


    for _, row in validation.iterrows():

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

            theta1=math.radians(
                row["theta1_deg"]
            ),

            theta2=math.radians(
                row["theta2_deg"]
            ),

            mu=row["mu"],

            RD=RD
        )


        if result is None:

            values.append(
                np.nan
            )

            cl_values.append(
                np.nan
            )

            ct_values.append(
                np.nan
            )

            clct_values.append(
                np.nan
            )

        else:

            values.append(
                result["BF"]
            )

            cl_values.append(
                result["CL"]
            )

            ct_values.append(
                result["CT"]
            )

            clct_values.append(
                result["CL_CT"]
            )


    return (

        np.asarray(
            values,
            dtype=float
        ),

        np.asarray(
            cl_values,
            dtype=float
        ),

        np.asarray(
            ct_values,
            dtype=float
        ),

        np.asarray(
            clct_values,
            dtype=float
        )
    )


# =========================================================
# PHYSICAL DELTA BF
# =========================================================

def calculate_physical_delta_bf(
    validation
):

    values = []


    for _, row in validation.iterrows():

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

                delta=0.10,

                RD=RD
            )
        )


        if result is None:

            values.append(
                np.nan
            )

        else:

            values.append(
                result[
                    "delta_BF_percent"
                ]
            )


    return np.asarray(
        values,
        dtype=float
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    title,
    metrics
):

    print(
        "\n----------------------------------------"
    )

    print(
        title
    )

    print(
        "----------------------------------------"
    )

    for key, value in metrics.items():

        print(
            f"{key:35s}"
            f" = {value:.6f}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - RSM INDEPENDENT VALIDATION"
    )

    print(
        "========================================"
    )


    # =====================================================
    # CHECK FILES
    # =====================================================

    print(
        "\nChecking datasets..."
    )


    if not BF_DATASET.exists():

        raise FileNotFoundError(
            f"\nBF dataset not found:\n"
            f"{BF_DATASET}"
        )


    if not DELTA_DATASET.exists():

        raise FileNotFoundError(
            f"\nDelta BF dataset not found:\n"
            f"{DELTA_DATASET}"
        )


    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(

            "\nIndependent validation file "
            "not found:\n"

            f"{VALIDATION_FILE}\n\n"

            "Run the Delta BF surrogate "
            "validation first."
        )


    # =====================================================
    # LOAD DATASETS
    # =====================================================

    bf_df = pd.read_csv(
        BF_DATASET
    )

    delta_df = pd.read_csv(
        DELTA_DATASET
    )

    validation = pd.read_csv(
        VALIDATION_FILE
    )


    print(
        f"\nTraining BF samples:"
        f" {len(bf_df)}"
    )

    print(
        f"Training Delta BF samples:"
        f" {len(delta_df)}"
    )

    print(
        f"Independent validation samples:"
        f" {len(validation)}"
    )


    # =====================================================
    # CHECK VALIDATION FEATURES
    # =====================================================

    missing = [

        feature

        for feature in FEATURES

        if feature not in validation.columns
    ]


    if missing:

        raise ValueError(

            "\nValidation file is missing:"
            "\n"

            +
            "\n".join(missing)
        )


    if len(validation) != 500:

        print(
            "\nWARNING:"
        )

        print(
            "Validation set does not contain "
            "exactly 500 samples."
        )


    # =====================================================
    # TRAIN RSM — BF
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "TRAINING RSM — BF"
    )

    print(
        "========================================"
    )


    bf_poly, bf_model = (
        train_rsm(

            bf_df,

            "BF"
        )
    )


    print(
        "BF RSM trained."
    )


    # =====================================================
    # TRAIN RSM — DELTA BF
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "TRAINING RSM — DELTA BF"
    )

    print(
        "========================================"
    )


    delta_poly, delta_model = (
        train_rsm(

            delta_df,

            "Delta_BF_percent"
        )
    )


    print(
        "Delta BF RSM trained."
    )


    # =====================================================
    # PHYSICAL REFERENCE VALUES
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "CALCULATING PHYSICAL VALIDATION VALUES"
    )

    print(
        "========================================"
    )


    physical_bf, cl, ct, clct = (
        calculate_physical_bf(
            validation
        )
    )


    physical_delta = (
        calculate_physical_delta_bf(
            validation
        )
    )


    # =====================================================
    # REMOVE INVALID POINTS
    # =====================================================

    valid = (

        np.isfinite(
            physical_bf
        )

        &

        np.isfinite(
            physical_delta
        )
    )


    validation = (
        validation.loc[
            valid
        ]
        .reset_index(drop=True)
    )


    physical_bf = (
        physical_bf[valid]
    )

    physical_delta = (
        physical_delta[valid]
    )

    cl = cl[valid]

    ct = ct[valid]

    clct = clct[valid]


    # =====================================================
    # RSM PREDICTIONS
    # =====================================================

    X_validation = validation[
        FEATURES
    ]


    bf_prediction = (
        bf_model.predict(

            bf_poly.transform(
                X_validation
            )
        )
    )


    delta_prediction = (
        delta_model.predict(

            delta_poly.transform(
                X_validation
            )
        )
    )


    # =====================================================
    # METRICS — BF
    # =====================================================

    bf_metrics = calculate_metrics(

        physical_bf,

        bf_prediction
    )


    # =====================================================
    # METRICS — DELTA BF
    # =====================================================

    delta_metrics = calculate_metrics(

        physical_delta,

        delta_prediction
    )


    print_results(

        "RSM — BF — INDEPENDENT VALIDATION",

        bf_metrics
    )


    print_results(

        "RSM — DELTA BF — INDEPENDENT VALIDATION",

        delta_metrics
    )


    # =====================================================
    # SAVE POINT-BY-POINT RESULTS
    # =====================================================

    results = validation.copy()


    results[
        "BF_physical"
    ] = physical_bf


    results[
        "BF_RSM"
    ] = bf_prediction


    results[
        "BF_absolute_error"
    ] = np.abs(

        bf_prediction -

        physical_bf
    )


    results[
        "BF_relative_error_percent"
    ] = (

        results[
            "BF_absolute_error"
        ]

        /

        np.abs(
            physical_bf
        )

    ) * 100.0


    results[
        "Delta_BF_physical"
    ] = physical_delta


    results[
        "Delta_BF_RSM"
    ] = delta_prediction


    results[
        "Delta_BF_absolute_error"
    ] = np.abs(

        delta_prediction -

        physical_delta
    )


    results[
        "Delta_BF_relative_error_percent"
    ] = (

        results[
            "Delta_BF_absolute_error"
        ]

        /

        np.abs(
            physical_delta
        )

    ) * 100.0


    results[
        "CL"
    ] = cl


    results[
        "CT"
    ] = ct


    results[
        "CL_CT"
    ] = clct


    output_file = (

        OUTPUT_DIR /

        "rsm_independent_validation_500.csv"
    )


    results.to_csv(

        output_file,

        index=False
    )


    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    summary = pd.DataFrame({

        "Model": [

            "RSM",

            "RSM"
        ],

        "Target": [

            "BF",

            "Delta_BF"
        ],

        "R2": [

            bf_metrics[
                "R2"
            ],

            delta_metrics[
                "R2"
            ]
        ],

        "RMSE": [

            bf_metrics[
                "RMSE"
            ],

            delta_metrics[
                "RMSE"
            ]
        ],

        "MAE": [

            bf_metrics[
                "MAE"
            ],

            delta_metrics[
                "MAE"
            ]
        ],

        "Mean_Absolute_Error_Percent": [

            bf_metrics[
                "Mean_Absolute_Error_Percent"
            ],

            delta_metrics[
                "Mean_Absolute_Error_Percent"
            ]
        ],

        "Max_Absolute_Error_Percent": [

            bf_metrics[
                "Max_Absolute_Error_Percent"
            ],

            delta_metrics[
                "Max_Absolute_Error_Percent"
            ]
        ]
    })


    summary_file = (

        OUTPUT_DIR /

        "rsm_independent_validation_summary.csv"
    )


    summary.to_csv(

        summary_file,

        index=False
    )


    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "FILES SAVED"
    )

    print(
        "========================================"
    )

    print(
        output_file
    )

    print(
        summary_file
    )


    print(
        "\n========================================"
    )

    print(
        "RSM INDEPENDENT VALIDATION COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()