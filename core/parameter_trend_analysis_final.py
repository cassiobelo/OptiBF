import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RANDOM_SEED = 42

MU_VALUE = 0.30

N_POINTS = 101


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
        0.9852181122,

    "min_child_weight":
        1,

    "gamma":
        0.0,

    "reg_alpha":
        0.0,

    "reg_lambda":
        1.0
}


# =========================================================
# GEOMETRY DOMAIN
# =========================================================

PARAM_RANGES = {

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


# =========================================================
# LOAD DATASET
# =========================================================

dataset_path = (
    DATA_DIR /
    "optibf_dataset_lhs_1000.csv"
)

df = pd.read_csv(
    dataset_path
)

print("=" * 60)
print("OptiBF - FINAL PARAMETER TREND ANALYSIS")
print("=" * 60)

print(
    f"\nDataset loaded: {len(df)} samples"
)


# =========================================================
# TARGET
# =========================================================

TARGET = "BF"

if TARGET not in df.columns:

    raise ValueError(
        f"Target '{TARGET}' not found in dataset."
    )


# =========================================================
# CHECK FEATURES
# =========================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features: "
        + str(missing_features)
    )


# =========================================================
# TRAIN TUNING 3
# =========================================================

print(
    "\nTraining Tuning 3 surrogate..."
)

model = XGBRegressor(

    objective="reg:squarederror",

    random_state=RANDOM_SEED,

    n_jobs=1,

    **TUNING3_PARAMS
)

model.fit(
    df[FEATURES],
    df[TARGET]
)

print(
    "Tuning 3 surrogate trained."
)


# =========================================================
# CENTRAL POINT
# =========================================================

central = {

    "L1_mm":
        np.mean(
            PARAM_RANGES["L1_mm"]
        ),

    "L3_mm":
        np.mean(
            PARAM_RANGES["L3_mm"]
        ),

    "L4_mm":
        np.mean(
            PARAM_RANGES["L4_mm"]
        ),

    "theta1_deg":
        np.mean(
            PARAM_RANGES["theta1_deg"]
        ),

    "theta2_deg":
        np.mean(
            PARAM_RANGES["theta2_deg"]
        ),

    "mu":
        MU_VALUE
}


print(
    "\nCentral point:"
)

for key, value in central.items():

    print(
        f"{key:15s} = {value:.6f}"
    )


# =========================================================
# OAT ANALYSIS
# =========================================================

summary_results = []

bf_results = []

delta_results = []


for parameter, (
    minimum,
    maximum
) in PARAM_RANGES.items():

    print(
        f"\nAnalyzing {parameter}..."
    )

    values = np.linspace(

        minimum,

        maximum,

        N_POINTS
    )

    predictions = []


    # -----------------------------------------------------
    # ONE-AT-A-TIME
    # -----------------------------------------------------

    for value in values:

        point = central.copy()

        point[parameter] = value

        row = pd.DataFrame(
            [point]
        )

        row = row[
            FEATURES
        ]

        prediction = (
            model.predict(
                row
            )[0]
        )

        predictions.append(
            prediction
        )


    predictions = np.asarray(
        predictions
    )


    # -----------------------------------------------------
    # CENTRAL BF
    # -----------------------------------------------------

    central_index = (
        len(predictions) // 2
    )

    bf_central = (
        predictions[
            central_index
        ]
    )


    # -----------------------------------------------------
    # MIN / MAX
    # -----------------------------------------------------

    bf_at_min = (
        predictions[0]
    )

    bf_at_max = (
        predictions[-1]
    )

    bf_min = (
        predictions.min()
    )

    bf_max = (
        predictions.max()
    )

    bf_change = (
        bf_max -
        bf_min
    )


    # -----------------------------------------------------
    # PERCENTAGE CHANGE
    # -----------------------------------------------------

    bf_change_pct = (

        bf_change /
        abs(bf_central)
        * 100
    )


    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    correlation = np.corrcoef(

        values,

        predictions

    )[0, 1]


    # -----------------------------------------------------
    # LINEAR SLOPE
    # -----------------------------------------------------

    slope = np.polyfit(

        values,

        predictions,

        1

    )[0]


    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if correlation >= 0.95:

        trend = "Increasing"

    elif correlation <= -0.95:

        trend = "Decreasing"

    else:

        trend = "Non-monotonic"


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary_results.append({

        "parameter":
            parameter,

        "min_value":
            minimum,

        "max_value":
            maximum,

        "BF_at_min":
            bf_at_min,

        "BF_at_max":
            bf_at_max,

        "BF_min":
            bf_min,

        "BF_max":
            bf_max,

        "BF_change":
            bf_change,

        "BF_change_pct":
            bf_change_pct,

        "slope":
            slope,

        "correlation":
            correlation,

        "trend":
            trend
    })


    # -----------------------------------------------------
    # BF CURVE
    # -----------------------------------------------------

    for value, prediction in zip(

        values,

        predictions

    ):

        bf_results.append({

            "parameter":
                parameter,

            "value":
                value,

            "mu":
                MU_VALUE,

            "BF":
                prediction
        })


    # -----------------------------------------------------
    # DELTA BF
    # -----------------------------------------------------

    for value, prediction in zip(

        values,

        predictions

    ):

        delta_results.append({

            "parameter":
                parameter,

            "value":
                value,

            "mu":
                MU_VALUE,

            "BF":
                prediction,

            "Delta_BF":
                prediction -
                bf_central
        })


# =========================================================
# DATAFRAMES
# =========================================================

summary_df = pd.DataFrame(
    summary_results
)

bf_df = pd.DataFrame(
    bf_results
)

delta_df = pd.DataFrame(
    delta_results
)


# =========================================================
# OUTPUT FILES
# =========================================================

summary_file = (

    DATA_DIR /
    "parameter_trends_summary_final.csv"
)

bf_file = (

    DATA_DIR /
    "parameter_trends_bf_final.csv"
)

delta_file = (

    DATA_DIR /
    "parameter_trends_delta_bf_final.csv"
)


# =========================================================
# SAVE
# =========================================================

summary_df.to_csv(

    summary_file,

    index=False
)

bf_df.to_csv(

    bf_file,

    index=False
)

delta_df.to_csv(

    delta_file,

    index=False
)


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("FINAL PARAMETER TREND RESULTS")
print("=" * 60)

print(

    summary_df[
        [
            "parameter",
            "min_value",
            "max_value",
            "BF_at_min",
            "BF_at_max",
            "BF_change",
            "BF_change_pct",
            "correlation",
            "trend"
        ]
    ].to_string(

        index=False,

        float_format=lambda x:
            f"{x:.6f}"
    )
)


# =========================================================
# FILES
# =========================================================

print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    summary_file
)

print(
    bf_file
)

print(
    delta_file
)

print("\nAnalysis completed.")