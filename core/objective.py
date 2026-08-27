import numpy as np

from core.brake_model import calculate_bf

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

def objective(
    x,
    MU_VALUES_OPT,
    RD,
    CL_CT_limit,
    optimization_mode
):

    L1, L3, L4, theta1, theta2 = x

    bfs = []

    penalty = 0.0

    for mu in MU_VALUES_OPT:

        result = calculate_bf(

            L1=L1,

            L3=L3,

            L4=L4,

            theta1=theta1,

            theta2=theta2,

            mu=mu,

            RD=RD
        )

        if result is None:

            return 1e6

        cl = result["CL"]

        ct = result["CT"]

        bf = result["BF"]

        ratio = result["CL_CT"]

        # =================================================
        # HARD CL/CT LIMIT
        # =================================================

        if CL_CT_limit is not None:

            if ratio > CL_CT_limit:
                return 1e6

        # =================================================
        # BF LIMIT
        # =================================================

        if bf > 8:

            return 1e6

        # =================================================
        # DENOMINATOR SAFETY
        # =================================================

        if result["den_cl"] < 0.05:

            return 1e6

        bfs.append(bf)

    bf_mean = np.mean(bfs)

    bf_std = np.std(bfs)

    objective_final = (

            bf_mean

            -

            0.1 * bf_std

            -

            penalty
    )

    # ==========================================
    # OBJECTIVE
    # ==========================================

    if optimization_mode == "Maximize Brake Factor":

        return -objective_final

    else:

        return objective_final
