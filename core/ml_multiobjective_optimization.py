import math
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from core.brake_model import (
    RD,
    calculate_bf,
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

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = (
    DATA_DIR /
    "ml_optimization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_SEED = 42

N_CANDIDATES = 30000

MU_VALUE = 0.30

ASYMMETRY_DELTA = 0.10



# =========================================================
# DESIGN SPACE
# =========================================================

BOUNDS = {

    "L1_mm":
        (35.0, 45.0),

    "L3_mm":
        (138.15, 143.15),

    "L4_mm":
        (152.0, 162.0),

    "theta1_deg":
        (18.0, 38.0),

    "theta2_deg":
        (135.0, 155.0)
}


FEATURES = [

    "L1_mm",
    "L3_mm",
    "L4_mm",
    "theta1_deg",
    "theta2_deg",
    "mu"
]


# =========================================================
# REFERENCE GEOMETRY
# =========================================================

REFERENCE = {

    "L1_mm": 40.00,

    "L3_mm": 143.15,

    "L4_mm": 157.00,

    "theta1_deg": 28.00,

    "theta2_deg": 145.00,

    "mu": MU_VALUE
}


# =========================================================
# TUNING 3 — BF SURROGATE
# =========================================================

BF_PARAMS = {

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
# DELTA BF SURROGATE
# =========================================================
#
# Same configuration used in the first Delta BF
# surrogate evaluation.
#
# This configuration was NOT separately tuned for Delta BF.
# =========================================================

DELTA_PARAMS = {

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
# LOAD DATA
# =========================================================

def load_datasets():

    print(
        "\n========================================"
    )

    print(
        "LOADING SURROGATE DATASETS"
    )

    print(
        "========================================"
    )

    bf_df = pd.read_csv(
        BF_DATASET
    )

    delta_df = pd.read_csv(
        DELTA_DATASET
    )

    print(
        f"BF dataset: "
        f"{len(bf_df)} samples"
    )

    print(
        f"Delta BF dataset: "
        f"{len(delta_df)} samples"
    )

    return (
        bf_df,
        delta_df
    )


# =========================================================
# TRAIN BF SURROGATE
# =========================================================

def train_bf_model(
    df
):

    X = df[
        FEATURES
    ]

    y = df[
        "BF"
    ]

    model = XGBRegressor(

        **BF_PARAMS,

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

    return model


# =========================================================
# TRAIN DELTA BF SURROGATE
# =========================================================

def train_delta_model(
    df
):

    X = df[
        FEATURES
    ]

    y = df[
        "Delta_BF_percent"
    ]

    model = XGBRegressor(

        **DELTA_PARAMS,

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

    return model


# =========================================================
# GENERATE CANDIDATES
# =========================================================

def generate_candidates():

    print(
        "\n========================================"
    )

    print(
        "GENERATING GEOMETRY CANDIDATES"
    )

    print(
        "========================================"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    data = {}

    for parameter, (
        minimum,
        maximum
    ) in BOUNDS.items():

        data[parameter] = (
            rng.uniform(
                minimum,
                maximum,
                N_CANDIDATES
            )
        )

    data["mu"] = np.full(
        N_CANDIDATES,
        MU_VALUE
    )

    candidates = pd.DataFrame(
        data
    )

    print(
        f"Generated "
        f"{len(candidates)} candidates."
    )

    return candidates


# =========================================================
# PREDICT SURROGATES
# =========================================================

def predict_surrogates(
    candidates,
    bf_model,
    delta_model
):

    X = candidates[
        FEATURES
    ]

    candidates = candidates.copy()

    candidates[
        "BF_ML"
    ] = bf_model.predict(
        X
    )

    candidates[
        "Delta_BF_ML"
    ] = delta_model.predict(
        X
    )

    return candidates


# =========================================================
# PHYSICAL CL / CT
# =========================================================

def calculate_cl_ct(
    row
):

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

        mu=MU_VALUE,

        RD=RD
    )

    if result is None:

        return np.nan

    return result["CL_CT"]


# =========================================================
# CALCULATE CL / CT AS INDICATOR
# =========================================================
#
# CL/CT is NOT used as a constraint in the final
# multi-objective optimization.
#
# It is calculated only as a reference indicator and
# reported in the final tables.
# =========================================================

def calculate_cl_ct_indicator(
    candidates
):

    print(
        "\n========================================"
    )

    print(
        "CALCULATING CL / CT INDICATOR"
    )

    print(
        "========================================"
    )

    print(
        "CL/CT is reported only — NOT a constraint."
    )

    cl_ct_values = []

    for _, row in candidates.iterrows():

        cl_ct_values.append(
            calculate_cl_ct(
                row
            )
        )

    candidates = candidates.copy()

    candidates[
        "CL_CT"
    ] = cl_ct_values

    valid_count = np.isfinite(
        candidates["CL_CT"]
    ).sum()

    print(
        f"\nCandidates: {len(candidates)}"
    )

    print(
        f"Valid CL/CT values: {valid_count}"
    )

    return candidates


# =========================================================
# PARETO FILTER
# =========================================================
#
# Objective:
#
#   maximize BF
#   minimize Delta BF
#
# A point is dominated if another point has:
#
#   BF >= current BF
#   Delta BF <= current Delta BF
#
# with at least one strict improvement.
# =========================================================

def pareto_filter(
    candidates
):

    print(
        "\n========================================"
    )

    print(
        "PARETO ANALYSIS"
    )

    print(
        "========================================"
    )

    data = candidates[
        [
            "BF_ML",
            "Delta_BF_ML"
        ]
    ].to_numpy()

    n = len(data)

    is_pareto = np.ones(
        n,
        dtype=bool
    )

    for i in range(n):

        if not is_pareto[i]:

            continue

        bf_i = data[i, 0]

        delta_i = data[i, 1]

        dominated = (

            (data[:, 0] >= bf_i)

            &

            (data[:, 1] <= delta_i)

            &

            (
                (data[:, 0] > bf_i)

                |

                (data[:, 1] < delta_i)
            )
        )

        dominated[i] = False

        if np.any(dominated):

            is_pareto[i] = False

    pareto = (
        candidates.loc[
            is_pareto
        ]
        .copy()
    )

    pareto = pareto.sort_values(
        "BF_ML"
    )

    print(
        f"Feasible candidates: "
        f"{n}"
    )

    print(
        f"Pareto solutions: "
        f"{len(pareto)}"
    )

    return pareto


# =========================================================
# PHYSICAL VALIDATION
# =========================================================

def validate_physical(
    pareto
):

    print(
        "\n========================================"
    )

    print(
        "PHYSICAL VALIDATION OF PARETO"
    )

    print(
        "========================================"
    )

    physical_bf = []

    physical_delta = []

    physical_cl = []

    physical_ct = []

    physical_clct = []


    for _, row in pareto.iterrows():

        bf_result = calculate_bf(

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

            mu=MU_VALUE,

            RD=RD
        )


        delta_result = (
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

                mu=MU_VALUE,

                delta=
                    ASYMMETRY_DELTA,

                RD=RD
            )
        )


        if bf_result is None:

            physical_bf.append(
                np.nan
            )

            physical_cl.append(
                np.nan
            )

            physical_ct.append(
                np.nan
            )

            physical_clct.append(
                np.nan
            )

        else:

            physical_bf.append(
                bf_result["BF"]
            )

            physical_cl.append(
                bf_result["CL"]
            )

            physical_ct.append(
                bf_result["CT"]
            )

            physical_clct.append(
                bf_result["CL_CT"]
            )


        if delta_result is None:

            physical_delta.append(
                np.nan
            )

        else:

            physical_delta.append(
                delta_result[
                    "delta_BF_percent"
                ]
            )


    validated = pareto.copy()

    validated[
        "BF_physical"
    ] = physical_bf

    validated[
        "Delta_BF_physical"
    ] = physical_delta

    validated[
        "CL_physical"
    ] = physical_cl

    validated[
        "CT_physical"
    ] = physical_ct

    validated[
        "CL_CT_physical"
    ] = physical_clct


    validated[
        "BF_error_percent"
    ] = (

        abs(
            validated["BF_ML"]
            -
            validated["BF_physical"]
        )

        /

        abs(
            validated["BF_physical"]
        )

    ) * 100.0


    validated[
        "Delta_BF_error_percent"
    ] = (

        abs(
            validated["Delta_BF_ML"]
            -
            validated["Delta_BF_physical"]
        )

        /

        abs(
            validated["Delta_BF_physical"]
        )

    ) * 100.0


    return validated


# =========================================================
# REFERENCE
# =========================================================

def evaluate_reference():

    row = pd.Series(
        REFERENCE
    )

    bf = calculate_bf(

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

        mu=MU_VALUE,

        RD=RD
    )

    delta = calculate_bf_asymmetry(

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

        mu=MU_VALUE,

        delta=
            ASYMMETRY_DELTA,

        RD=RD
    )

    return {

        "BF":
            bf["BF"],

        "Delta_BF":
            delta[
                "delta_BF_percent"
            ],

        "CL_CT":
            bf["CL_CT"]
    }


# =========================================================
# PRINT REPRESENTATIVE SOLUTIONS
# =========================================================

def print_representative_solutions(
    validated
):

    print(
        "\n========================================"
    )

    print(
        "REPRESENTATIVE PARETO SOLUTIONS"
    )

    print(
        "========================================"
    )


    if len(validated) == 0:

        print(
            "No Pareto solutions available."
        )

        return


    # -----------------------------------------------------
    # Maximum BF
    # -----------------------------------------------------

    max_bf = (
        validated.loc[
            validated[
                "BF_ML"
            ].idxmax()
        ]
    )


    # -----------------------------------------------------
    # Minimum Delta BF
    # -----------------------------------------------------

    min_delta = (
        validated.loc[
            validated[
                "Delta_BF_ML"
            ].idxmin()
        ]
    )


    # -----------------------------------------------------
    # Minimum distance to reference
    # -----------------------------------------------------

    normalized = validated.copy()

    normalized[
        "distance"
    ] = np.sqrt(

        (
            (
                normalized[
                    "BF_ML"
                ]
                -
                normalized[
                    "BF_ML"
                ].min()
            )

            /

            (
                normalized[
                    "BF_ML"
                ].max()
                -
                normalized[
                    "BF_ML"
                ].min()
                + 1e-12
            )
        ) ** 2

        +

        (
            (
                normalized[
                    "Delta_BF_ML"
                ]
                -
                normalized[
                    "Delta_BF_ML"
                ].min()
            )

            /

            (
                normalized[
                    "Delta_BF_ML"
                ].max()
                -
                normalized[
                    "Delta_BF_ML"
                ].min()
                + 1e-12
            )
        ) ** 2
    )


    balanced = (
        normalized.loc[
            normalized[
                "distance"
            ].idxmin()
        ]
    )


    representative = pd.DataFrame(
        [
            max_bf,
            min_delta,
            balanced
        ]
    )


    representative[
        "Solution"
    ] = [

        "Maximum BF",

        "Minimum Delta BF",

        "Balanced Pareto"
    ]


    print(

        representative[
            [
                "Solution",

                "L1_mm",

                "L3_mm",

                "L4_mm",

                "theta1_deg",

                "theta2_deg",

                "BF_ML",

                "Delta_BF_ML",

                "BF_physical",

                "Delta_BF_physical",

                "CL_CT_physical",

                "BF_error_percent",

                "Delta_BF_error_percent"
            ]
        ].to_string(

            index=False,

            float_format=
                lambda x:
                f"{x:.6f}"
        )
    )


    return representative


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - MULTI-OBJECTIVE OPTIMIZATION"
    )

    print(
        "========================================"
    )

    print(
        "\nObjectives:"
    )

    print(
        "1. MAXIMIZE BF"
    )

    print(
        "2. MINIMIZE Delta BF"
    )

    print(
        "\nConstraint: NONE"
    )

    print(
        "\nCL/CT:"
    )

    print(
        "Reported only — NOT a constraint"
    )

    print(
        f"\nFixed mu:"
        f" {MU_VALUE:.2f}"
    )


    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    bf_df, delta_df = (
        load_datasets()
    )


    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    print(
        "\nTraining BF surrogate..."
    )

    bf_model = train_bf_model(
        bf_df
    )

    print(
        "BF surrogate trained."
    )


    print(
        "\nTraining Delta BF surrogate..."
    )

    delta_model = train_delta_model(
        delta_df
    )

    print(
        "Delta BF surrogate trained."
    )


    # -----------------------------------------------------
    # CANDIDATES
    # -----------------------------------------------------

    candidates = (
        generate_candidates()
    )


    # -----------------------------------------------------
    # ML PREDICTIONS
    # -----------------------------------------------------

    candidates = (
        predict_surrogates(

            candidates,

            bf_model,

            delta_model
        )
    )


    # -----------------------------------------------------
    # CL / CT INDICATOR
    # -----------------------------------------------------

    candidates = (
        calculate_cl_ct_indicator(
            candidates
        )
    )


    # -----------------------------------------------------
    # PARETO
    # -----------------------------------------------------

    pareto = (
        pareto_filter(
            candidates
        )
    )


    # -----------------------------------------------------
    # PHYSICAL VALIDATION
    # -----------------------------------------------------

    validated = (
        validate_physical(
            pareto
        )
    )


    # -----------------------------------------------------
    # REMOVE INVALID
    # -----------------------------------------------------

    validated = validated.loc[

        np.isfinite(
            validated[
                "BF_physical"
            ]
        )

        &

        np.isfinite(
            validated[
                "Delta_BF_physical"
            ]
        )

    ].copy()


    # -----------------------------------------------------
    # REFERENCE
    # -----------------------------------------------------

    reference = (
        evaluate_reference()
    )


    print(
        "\n========================================"
    )

    print(
        "REFERENCE"
    )

    print(
        "========================================"
    )

    print(
        f"BF       = "
        f"{reference['BF']:.8f}"
    )

    print(
        f"Delta BF = "
        f"{reference['Delta_BF']:.8f}%"
    )

    print(
        f"CL/CT    = "
        f"{reference['CL_CT']:.8f}"
    )


    # -----------------------------------------------------
    # SAVE COMPLETE PARETO
    # -----------------------------------------------------

    pareto_file = (

        OUTPUT_DIR /

        "ml_multiobjective_pareto.csv"
    )


    validated.to_csv(

        pareto_file,

        index=False
    )


    print(
        "\nSaved Pareto front:"
    )

    print(
        pareto_file
    )


    # -----------------------------------------------------
    # REPRESENTATIVE SOLUTIONS
    # -----------------------------------------------------

    representative = (
        print_representative_solutions(
            validated
        )
    )


    representative_file = (

        OUTPUT_DIR /

        "ml_multiobjective_representative_solutions.csv"
    )


    if representative is not None:

        representative.to_csv(

            representative_file,

            index=False
        )

        print(
            "\nSaved representative solutions:"
        )

        print(
            representative_file
        )


    # -----------------------------------------------------
    # SAVE ALL CANDIDATES
    # -----------------------------------------------------

    candidates_file = (

        OUTPUT_DIR /

        "ml_multiobjective_candidates.csv"
    )


    candidates.to_csv(

        candidates_file,

        index=False
    )


    print(
        "\nSaved all candidates:"
    )

    print(
        candidates_file
    )


    # The previous file represented a CL/CT-constrained
    # candidate set and is obsolete for the final method.
    old_feasible_file = (
        OUTPUT_DIR /
        "ml_multiobjective_feasible_candidates.csv"
    )

    if old_feasible_file.exists():

        try:
            old_feasible_file.unlink()

            print(
                "\nRemoved obsolete file:"
            )

            print(
                old_feasible_file
            )

        except OSError:

            print(
                "\nWarning: could not remove obsolete file:"
            )

            print(
                old_feasible_file
            )


    print(
        "\n========================================"
    )

    print(
        "MULTI-OBJECTIVE OPTIMIZATION COMPLETED"
    )

    print(
        "========================================"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()