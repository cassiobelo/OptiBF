from scipy.optimize import differential_evolution

from core.objective import objective

# =========================================================
# RUN OPTIMIZATION
# =========================================================

def run_optimization(
    bounds,
    MU_VALUES_OPT,
    RD,
    CL_CT_limit,
    optimization_mode
):

    result = differential_evolution(

        func=objective,

        bounds=bounds,

        args=(
            MU_VALUES_OPT,
            RD,
            CL_CT_limit,
            optimization_mode
        ),

        strategy="best1bin",

        maxiter=200,

        popsize=20,

        tol=1e-6,

        mutation=(0.5, 1),

        recombination=0.7,

        polish=False,

        disp=False,

        workers=1
    )

    return result