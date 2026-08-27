import math

# =========================================================
# PHYSICAL CONSTANTS
# =========================================================

RD_MM = 196.85
RD = RD_MM / 1000.0

# =========================================================
# DESIGN SPACE
# =========================================================

BOUNDS = {
    "L1_mm": (35.0, 45.0),
    "L3_mm": (138.15, 143.15),
    "L4_mm": (152.0, 162.0),
    "theta1_deg": (18.0, 38.0),
    "theta2_deg": (135.0, 155.0),
    "mu": (0.05, 0.40),
}


# =========================================================
# ROBUSTNESS
# =========================================================

ROBUSTNESS_DELTA = 0.10

# =========================================================
# BRAKE FACTOR CALCULATION
# =========================================================

def calculate_bf(
    L1,
    L3,
    L4,
    theta1,
    theta2,
    mu,
    RD
):

    a0 = theta2 - theta1

    if a0 <= 0:
        return None

    a1 = math.pi - theta2

    a3 = a0 + 2 * a1

    a_line = math.hypot(
        L1,
        L4
    )

    h = L3 + L4

    numerator = mu * h / RD

    try:

        denominator_common = (

            (a_line / RD)

            *

            (
                (
                    a0
                    -
                    (
                        math.sin(a0)
                        *
                        math.cos(a3)
                    )
                )

                /

                (
                    4
                    *
                    math.sin(a0 / 2)
                    *
                    math.sin(a3 / 2)
                )
            )
        )

    except ZeroDivisionError:

        return None

    denominator_variable = (

        mu

        *

        (
            1
            +
            (
                (a_line / RD)
                *
                math.cos(a0 / 2)
                *
                math.cos(a3 / 2)
            )
        )
    )

    den_cl = (
        denominator_common
        -
        denominator_variable
    )

    den_ct = (
        denominator_common
        +
        denominator_variable
    )

    if den_cl <= 0 or den_ct <= 0:
        return None

    cl = numerator / den_cl

    ct = numerator / den_ct

    bf = cl + ct

    return {

        "CL": cl,

        "CT": ct,

        "BF": bf,

        "CL_CT": cl / ct,

        "den_cl": den_cl,

        "den_ct": den_ct
    }

# =========================================================
# BRAKE FACTOR ROBUSTNESS TO FRICTION ASYMMETRY
# =========================================================

def calculate_bf_asymmetry(
    L1,
    L3,
    L4,
    theta1,
    theta2,
    mu,
    delta,
    RD
):

    mu_left = mu * (1.0 + delta)
    mu_right = mu * (1.0 - delta)

    result_left = calculate_bf(
        L1=L1,
        L3=L3,
        L4=L4,
        theta1=theta1,
        theta2=theta2,
        mu=mu_left,
        RD=RD
    )

    result_right = calculate_bf(
        L1=L1,
        L3=L3,
        L4=L4,
        theta1=theta1,
        theta2=theta2,
        mu=mu_right,
        RD=RD
    )

    if result_left is None or result_right is None:
        return None

    bf_left = result_left["BF"]
    bf_right = result_right["BF"]

    bf_mean = (bf_left + bf_right) / 2.0

    if bf_mean == 0:
        return None

    delta_bf = (
        abs(bf_left - bf_right)
        / bf_mean
    ) * 100.0

    return {
        "mu_left": mu_left,
        "mu_right": mu_right,
        "BF_left": bf_left,
        "BF_right": bf_right,
        "BF_mean": bf_mean,
        "delta_BF_percent": delta_bf
    }