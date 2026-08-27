import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from scipy.optimize import differential_evolution

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

OUTPUT_DIR = (
    DATA_DIR /
    "rsm_optimization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DATASET
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
# DESIGN SPACE
# =========================================================

BOUNDS = [

    (35.0, 45.0),       # L1

    (138.15, 143.15),   # L3

    (152.0, 162.0),     # L4

    (18.0, 38.0),       # theta1

    (135.0, 155.0)      # theta2
]


FIXED_MU = 0.30


# =========================================================
# REFERENCE
# =========================================================

REFERENCE = {

    "L1_mm": 40.0,

    "L3_mm": 143.15,

    "L4_mm": 157.0,

    "theta1_deg": 28.0,

    "theta2_deg": 145.0,

    "mu": FIXED_MU
}


# =========================================================
# RSM MODEL
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

    polynomial = PolynomialFeatures(

        degree=2,

        include_bias=False
    )

    X_poly = (
        polynomial.fit_transform(
            X
        )
    )

    model = LinearRegression()

    model.fit(
        X_poly,
        y
    )

    return (
        polynomial,
        model
    )


# =========================================================
# RSM PREDICTION
# =========================================================

def predict_rsm(
    polynomial,
    model,
    X
):

    X_poly = (
        polynomial.transform(
            X
        )
    )

    return model.predict(
        X_poly
    )


# =========================================================
# REFERENCE PHYSICAL MODEL
# =========================================================

def physical_reference():

    r = REFERENCE

    bf = calculate_bf(

        L1=r["L1_mm"] / 1000.0,

        L3=r["L3_mm"] / 1000.0,

        L4=r["L4_mm"] / 1000.0,

        theta1=math.radians(
            r["theta1_deg"]
        ),

        theta2=math.radians(
            r["theta2_deg"]
        ),

        mu=FIXED_MU,

        RD=RD
    )

    delta = (
        calculate_bf_asymmetry(

            L1=r["L1_mm"] / 1000.0,

            L3=r["L3_mm"] / 1000.0,

            L4=r["L4_mm"] / 1000.0,

            theta1=math.radians(
                r["theta1_deg"]
            ),

            theta2=math.radians(
                r["theta2_deg"]
            ),

            mu=FIXED_MU,

            delta=0.10,

            RD=RD
        )
    )

    return {

        "BF":
            bf["BF"],

        "Delta_BF":
            delta["delta_BF_percent"],

        "CL":
            bf["CL"],

        "CT":
            bf["CT"],

        "CL_CT":
            bf["CL_CT"]
    }


# =========================================================
# GENERATE CANDIDATES
# =========================================================

def generate_candidates(
    n=30000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    data = {}

    for feature, (
        minimum,
        maximum
    ) in zip(
        [
            "L1_mm",
            "L3_mm",
            "L4_mm",
            "theta1_deg",
            "theta2_deg"
        ],
        BOUNDS
    ):

        data[feature] = (
            rng.uniform(
                minimum,
                maximum,
                n
            )
        )


    data["mu"] = np.full(
        n,
        FIXED_MU
    )

    return pd.DataFrame(
        data
    )


# =========================================================
# PARETO FILTER
# =========================================================

def pareto_filter(
    df
):

    values = df[
        [
            "BF_RSM",
            "Delta_BF_RSM"
        ]
    ].to_numpy()

    n = len(values)

    is_pareto = np.ones(
        n,
        dtype=bool
    )


    for i in range(n):

        bf_i = values[i, 0]

        delta_i = values[i, 1]


        dominates = (

            (values[:, 0] >= bf_i)

            &

            (values[:, 1] <= delta_i)

            &

            (

                (values[:, 0] > bf_i)

                |

                (values[:, 1] < delta_i)
            )
        )


        dominates[i] = False


        if np.any(
            dominates
        ):

            is_pareto[i] = False


    return (
        df.loc[
            is_pareto
        ]
        .sort_values(
            "BF_RSM"
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# PHYSICAL VALIDATION
# =========================================================

def validate_physical(
    df
):

    bf_values = []

    delta_values = []

    cl_values = []

    ct_values = []

    clct_values = []


    for _, row in df.iterrows():

        bf = calculate_bf(

            L1=row["L1_mm"] / 1000.0,

            L3=row["L3_mm"] / 1000.0,

            L4=row["L4_mm"] / 1000.0,

            theta1=math.radians(
                row["theta1_deg"]
            ),

            theta2=math.radians(
                row["theta2_deg"]
            ),

            mu=FIXED_MU,

            RD=RD
        )


        delta = (
            calculate_bf_asymmetry(

                L1=row["L1_mm"] / 1000.0,

                L3=row["L3_mm"] / 1000.0,

                L4=row["L4_mm"] / 1000.0,

                theta1=math.radians(
                    row["theta1_deg"]
                ),

                theta2=math.radians(
                    row["theta2_deg"]
                ),

                mu=FIXED_MU,

                delta=0.10,

                RD=RD
            )
        )


        bf_values.append(
            bf["BF"]
        )

        cl_values.append(
            bf["CL"]
        )

        ct_values.append(
            bf["CT"]
        )

        clct_values.append(
            bf["CL_CT"]
        )

        delta_values.append(
            delta[
                "delta_BF_percent"
            ]
        )


    result = df.copy()


    result[
        "BF_physical"
    ] = bf_values


    result[
        "Delta_BF_physical"
    ] = delta_values


    result[
        "CL_physical"
    ] = cl_values


    result[
        "CT_physical"
    ] = ct_values


    result[
        "CL_CT_physical"
    ] = clct_values


    return result


# =========================================================
# PARETO PHYSICAL
# =========================================================

def physical_pareto_filter(
    df
):

    values = df[
        [
            "BF_physical",
            "Delta_BF_physical"
        ]
    ].to_numpy()


    n = len(values)

    is_pareto = np.ones(
        n,
        dtype=bool
    )


    for i in range(n):

        bf_i = values[i, 0]

        delta_i = values[i, 1]


        dominates = (

            (values[:, 0] >= bf_i)

            &

            (values[:, 1] <= delta_i)

            &

            (

                (values[:, 0] > bf_i)

                |

                (values[:, 1] < delta_i)
            )
        )


        dominates[i] = False


        if np.any(
            dominates
        ):

            is_pareto[i] = False


    return (
        df.loc[
            is_pareto
        ]
        .sort_values(
            "BF_physical"
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "OptiBF - RSM MULTI-OBJECTIVE OPTIMIZATION"
    )

    print(
        "========================================"
    )


    # =====================================================
    # LOAD DATA
    # =====================================================

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
    # TRAIN RSM
    # =====================================================

    print(
        "\nTraining BF RSM..."
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


    print(
        "\nTraining Delta BF RSM..."
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
    # REFERENCE
    # =====================================================

    reference = (
        physical_reference()
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


    for key, value in reference.items():

        print(
            f"{key:10s} = "
            f"{value:.8f}"
        )


    # =====================================================
    # GENERATE CANDIDATES
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "GENERATING GEOMETRY CANDIDATES"
    )

    print(
        "========================================"
    )


    candidates = generate_candidates()


    print(
        f"Generated "
        f"{len(candidates)} candidates."
    )


    # =====================================================
    # RSM PREDICTIONS
    # =====================================================

    X = candidates[
        FEATURES
    ]


    candidates[
        "BF_RSM"
    ] = predict_rsm(

        bf_poly,

        bf_model,

        X
    )


    candidates[
        "Delta_BF_RSM"
    ] = predict_rsm(

        delta_poly,

        delta_model,

        X
    )


    # =====================================================
    # REMOVE INVALID PREDICTIONS
    # =====================================================

    candidates = (
        candidates[
            np.isfinite(
                candidates[
                    "BF_RSM"
                ]
            )

            &

            np.isfinite(
                candidates[
                    "Delta_BF_RSM"
                ]
            )
        ]
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # RSM PARETO
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "RSM PARETO ANALYSIS"
    )

    print(
        "========================================"
    )


    pareto = pareto_filter(
        candidates
    )


    print(
        f"Feasible candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Pareto solutions: "
        f"{len(pareto)}"
    )


    # =====================================================
    # PHYSICAL VALIDATION
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "PHYSICAL VALIDATION OF RSM PARETO"
    )

    print(
        "========================================"
    )


    physical = validate_physical(
        pareto
    )


    physical[
        "BF_error_percent"
    ] = (

        np.abs(

            physical[
                "BF_RSM"
            ]

            -

            physical[
                "BF_physical"
            ]

        )

        /

        np.abs(
            physical[
                "BF_physical"
            ]
        )

    ) * 100.0


    physical[
        "Delta_BF_error_percent"
    ] = (

        np.abs(

            physical[
                "Delta_BF_RSM"
            ]

            -

            physical[
                "Delta_BF_physical"
            ]

        )

        /

        np.abs(
            physical[
                "Delta_BF_physical"
            ]
        )

    ) * 100.0


    # =====================================================
    # PHYSICAL PARETO
    # =====================================================

    physical_pareto = (
        physical_pareto_filter(
            physical
        )
    )


    print(
        f"\nPhysical Pareto solutions: "
        f"{len(physical_pareto)}"
    )


    # =====================================================
    # REFERENCE DELTAS
    # =====================================================

    physical_pareto[
        "BF_change_percent"
    ] = (

        (

            physical_pareto[
                "BF_physical"
            ]

            -

            reference[
                "BF"
            ]

        )

        /

        reference[
            "BF"
        ]

    ) * 100.0


    physical_pareto[
        "Delta_BF_change_pp"
    ] = (

        physical_pareto[
            "Delta_BF_physical"
        ]

        -

        reference[
            "Delta_BF"
        ]
    )


    physical_pareto[
        "Delta_BF_change_percent"
    ] = (

        physical_pareto[
            "Delta_BF_change_pp"
        ]

        /

        reference[
            "Delta_BF"
        ]

    ) * 100.0


    # =====================================================
    # SAVE
    # =====================================================

    candidates_file = (

        OUTPUT_DIR /

        "rsm_multiobjective_candidates.csv"
    )


    pareto_file = (

        OUTPUT_DIR /

        "rsm_multiobjective_pareto.csv"
    )


    physical_file = (

        OUTPUT_DIR /

        "rsm_multiobjective_physical_pareto.csv"
    )


    candidates.to_csv(

        candidates_file,

        index=False
    )


    pareto.to_csv(

        pareto_file,

        index=False
    )


    physical_pareto.to_csv(

        physical_file,

        index=False
    )


    # =====================================================
    # REPRESENTATIVE SOLUTIONS
    # =====================================================

    maximum_bf = (
        physical_pareto.loc[
            physical_pareto[
                "BF_physical"
            ].idxmax()
        ]
    )


    minimum_delta = (
        physical_pareto.loc[
            physical_pareto[
                "Delta_BF_physical"
            ].idxmin()
        ]
    )


    win_win = physical_pareto[

        (

            physical_pareto[
                "BF_physical"
            ]

            >

            reference[
                "BF"
            ]

        )

        &

        (

            physical_pareto[
                "Delta_BF_physical"
            ]

            <

            reference[
                "Delta_BF"
            ]

        )
    ]


    print(
        "\n========================================"
    )

    print(
        "REPRESENTATIVE PHYSICAL SOLUTIONS"
    )

    print(
        "========================================"
    )


    columns = [

        "L1_mm",

        "L3_mm",

        "L4_mm",

        "theta1_deg",

        "theta2_deg",

        "BF_RSM",

        "Delta_BF_RSM",

        "BF_physical",

        "Delta_BF_physical",

        "CL_CT_physical",

        "BF_error_percent",

        "Delta_BF_error_percent"
    ]


    print(
        "\nMaximum BF:"
    )

    print(
        maximum_bf[
            columns
        ].to_string()
    )


    print(
        "\nMinimum Delta BF:"
    )

    print(
        minimum_delta[
            columns
        ].to_string()
    )


    if len(win_win) > 0:

        best_win_win = (
            win_win.loc[
                win_win[
                    "BF_physical"
                ].idxmax()
            ]
        )


        print(
            "\nBest win-win:"
        )

        print(
            best_win_win[
                columns
            ].to_string()
        )

    else:

        print(
            "\nNo physical win-win "
            "solution found."
        )


    # =====================================================
    # FINAL
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
        candidates_file
    )

    print(
        pareto_file
    )

    print(
        physical_file
    )


    print(
        "\n========================================"
    )

    print(
        "RSM MULTI-OBJECTIVE OPTIMIZATION COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()