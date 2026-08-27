import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
from xgboost import XGBRegressor

from core.brake_model import (
    calculate_bf,
    RD
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT /
    "data"
)

DATA_FILE = (
    DATA_DIR /
    "optibf_dataset_lhs_1000.csv"
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

MU_OPT = 0.30

# ---------------------------------------------------------
# Geometry design space
# ---------------------------------------------------------

BOUNDS = [

    (35.0, 45.0),       # L1_mm

    (138.15, 143.15),   # L3_mm

    (152.0, 162.0),     # L4_mm

    (18.0, 38.0),       # theta1_deg

    (135.0, 155.0)      # theta2_deg
]


# =========================================================
# CURRENT / REFERENCE GEOMETRY
# =========================================================
#
# Reference geometry used previously in the project.
#
# Original configuration:
#
# L1    = 40.00 mm
# L3    = 143.15 mm
# L4    = 157.00 mm
# theta1 = 28.00 deg
# theta2 = 145.00 deg
#
# This is kept only as a reference for comparison.
# =========================================================

REFERENCE_GEOMETRY = np.array([

    40.00,

    143.15,

    157.00,

    28.00,

    145.00
])


# =========================================================
# TUNING 3 — FINAL SURROGATE
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
# LOAD DATASET
# =========================================================

def load_dataset():

    print(
        "\n========================================"
    )

    print(
        "LOADING DATASET"
    )

    print(
        "========================================"
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"Dataset: {DATA_FILE}"
    )

    print(
        f"Samples: {len(df)}"
    )

    return df


# =========================================================
# TRAIN SURROGATE
# =========================================================

def train_surrogate(df):

    print(
        "\n========================================"
    )

    print(
        "TRAINING TUNING 3 SURROGATE"
    )

    print(
        "========================================"
    )

    X = df[
        FEATURES
    ]

    y = df[
        "BF"
    ]

    model = XGBRegressor(

        **TUNING3_PARAMS,

        objective="reg:squarederror",

        random_state=RANDOM_SEED,

        n_jobs=1
    )

    model.fit(
        X,
        y
    )

    print(
        "Tuning 3 surrogate trained."
    )

    return model


# =========================================================
# SURROGATE PREDICTION
# =========================================================

def predict_bf_ml(
    model,
    x
):

    row = pd.DataFrame(

        [[

            x[0],     # L1

            x[1],     # L3

            x[2],     # L4

            x[3],     # theta1

            x[4],     # theta2

            MU_OPT

        ]],

        columns=FEATURES
    )

    prediction = (
        model.predict(row)[0]
    )

    return prediction


# =========================================================
# PHYSICAL MODEL
# =========================================================

def calculate_bf_physical(
    x
):

    L1 = (
        x[0] /
        1000.0
    )

    L3 = (
        x[1] /
        1000.0
    )

    L4 = (
        x[2] /
        1000.0
    )

    theta1 = math.radians(
        x[3]
    )

    theta2 = math.radians(
        x[4]
    )

    result = calculate_bf(

        L1=L1,

        L3=L3,

        L4=L4,

        theta1=theta1,

        theta2=theta2,

        mu=MU_OPT,

        RD=RD
    )

    if result is None:

        return None

    return result


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
#
# We maximize BF predicted by the surrogate.
#
# differential_evolution performs minimization,
# therefore the objective returns -BF.
# =========================================================

def objective_ml(
    x,
    model
):

    bf_ml = predict_bf_ml(
        model,
        x
    )

    return -bf_ml


# =========================================================
# RUN ML OPTIMIZATION
# =========================================================

def optimize_geometry(
    model
):

    print(
        "\n========================================"
    )

    print(
        "ML GEOMETRY OPTIMIZATION"
    )

    print(
        "========================================"
    )

    print(
        "\nObjective:"
    )

    print(
        "Maximize BF predicted by XGBoost"
    )

    print(
        f"\nFixed friction coefficient: "
        f"mu = {MU_OPT:.2f}"
    )

    print(
        "\nDesign space:"
    )

    for name, bounds in zip(

        FEATURES[:-1],

        BOUNDS

    ):

        print(
            f"{name:15s}: "
            f"{bounds[0]:.4f} → "
            f"{bounds[1]:.4f}"
        )


    result = differential_evolution(

        func=objective_ml,

        bounds=BOUNDS,

        args=(model,),

        strategy="best1bin",

        maxiter=200,

        popsize=20,

        tol=1e-7,

        mutation=(0.5, 1.0),

        recombination=0.7,

        polish=True,

        seed=RANDOM_SEED,

        workers=1,

        updating="immediate"
    )


    return result


# =========================================================
# EVALUATE GEOMETRY
# =========================================================

def evaluate_geometry(
    model,
    name,
    x
):

    bf_ml = predict_bf_ml(
        model,
        x
    )

    physical = calculate_bf_physical(
        x
    )

    if physical is None:

        raise RuntimeError(
            f"Physical model returned None "
            f"for {name}."
        )

    bf_physical = physical[
        "BF"
    ]

    error_abs = (
        abs(
            bf_ml -
            bf_physical
        )
    )

    error_pct = (

        error_abs /
        abs(bf_physical)
        * 100.0
    )


    print(
        "\n----------------------------------------"
    )

    print(
        f"{name}"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"L1       = {x[0]:.6f} mm"
    )

    print(
        f"L3       = {x[1]:.6f} mm"
    )

    print(
        f"L4       = {x[2]:.6f} mm"
    )

    print(
        f"Theta1   = {x[3]:.6f} deg"
    )

    print(
        f"Theta2   = {x[4]:.6f} deg"
    )

    print(
        f"mu       = {MU_OPT:.4f}"
    )

    print(
        f"\nBF ML        = "
        f"{bf_ml:.8f}"
    )

    print(
        f"BF physical  = "
        f"{bf_physical:.8f}"
    )

    print(
        f"Absolute error = "
        f"{error_abs:.8f}"
    )

    print(
        f"Relative error = "
        f"{error_pct:.6f}%"
    )

    print(
        f"CL           = "
        f"{physical['CL']:.8f}"
    )

    print(
        f"CT           = "
        f"{physical['CT']:.8f}"
    )

    print(
        f"CL/CT        = "
        f"{physical['CL_CT']:.8f}"
    )


    return {

        "Geometry":
            name,

        "L1_mm":
            x[0],

        "L3_mm":
            x[1],

        "L4_mm":
            x[2],

        "theta1_deg":
            x[3],

        "theta2_deg":
            x[4],

        "mu":
            MU_OPT,

        "BF_ML":
            bf_ml,

        "BF_physical":
            bf_physical,

        "BF_error_abs":
            error_abs,

        "BF_error_percent":
            error_pct,

        "CL":
            physical["CL"],

        "CT":
            physical["CT"],

        "CL_CT":
            physical["CL_CT"]
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - ML GEOMETRY OPTIMIZATION"
    )

    print(
        "========================================"
    )


    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    df = load_dataset()


    # -----------------------------------------------------
    # SURROGATE
    # -----------------------------------------------------

    model = train_surrogate(
        df
    )


    # -----------------------------------------------------
    # OPTIMIZATION
    # -----------------------------------------------------

    result = optimize_geometry(
        model
    )


    x_opt = result.x


    print(
        "\n========================================"
    )

    print(
        "OPTIMIZATION RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Success: "
        f"{result.success}"
    )

    print(
        f"Iterations: "
        f"{result.nit}"
    )

    print(
        f"Function value: "
        f"{result.fun:.10f}"
    )


    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    results = []


    reference_result = evaluate_geometry(

        model,

        "Reference",

        REFERENCE_GEOMETRY
    )

    results.append(
        reference_result
    )


    optimized_result = evaluate_geometry(

        model,

        "ML Optimized",

        x_opt
    )

    results.append(
        optimized_result
    )


    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    reference_bf = (
        reference_result[
            "BF_physical"
        ]
    )

    optimized_bf = (
        optimized_result[
            "BF_physical"
        ]
    )

    bf_gain = (

        (
            optimized_bf -
            reference_bf
        )

        /

        reference_bf

    ) * 100.0


    print(
        "\n========================================"
    )

    print(
        "PHYSICAL MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    print(
        f"Reference BF : "
        f"{reference_bf:.8f}"
    )

    print(
        f"Optimized BF : "
        f"{optimized_bf:.8f}"
    )

    print(
        f"BF variation : "
        f"{bf_gain:.6f}%"
    )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    output_file = (

        OUTPUT_DIR /
        "ml_geometry_optimization_mu030.csv"
    )


    results_df.to_csv(

        output_file,

        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        output_file
    )


    # -----------------------------------------------------
    # SAVE OPTIMIZATION RESULT
    # -----------------------------------------------------

    optimization_file = (

        OUTPUT_DIR /
        "ml_geometry_optimization_result.txt"
    )


    with open(

        optimization_file,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "OptiBF - ML Geometry Optimization\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"mu = {MU_OPT:.6f}\n\n"
        )

        f.write(
            "Tuning 3 parameters:\n"
        )

        for key, value in (
            TUNING3_PARAMS.items()
        ):

            f.write(
                f"{key} = {value}\n"
            )

        f.write(
            "\nOptimized geometry:\n"
        )

        names = [
            "L1_mm",
            "L3_mm",
            "L4_mm",
            "theta1_deg",
            "theta2_deg"
        ]

        for name, value in zip(
            names,
            x_opt
        ):

            f.write(
                f"{name} = {value:.10f}\n"
            )

        f.write(
            f"\nBF ML = "
            f"{optimized_result['BF_ML']:.10f}\n"
        )

        f.write(
            f"BF physical = "
            f"{optimized_result['BF_physical']:.10f}\n"
        )

        f.write(
            f"Error (%) = "
            f"{optimized_result['BF_error_percent']:.10f}\n"
        )


    print(
        optimization_file
    )


    print(
        "\n========================================"
    )

    print(
        "ML GEOMETRY OPTIMIZATION COMPLETED"
    )

    print(
        "========================================"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()