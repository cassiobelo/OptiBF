# OptiBF - Surrogate Accuracy vs Pareto Ranking Analysis
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml_rsm_comparison"
GLOBAL = DATA / "global_physical_pareto"
OUT = GLOBAL / "ranking_analysis"
OUT.mkdir(parents=True, exist_ok=True)

REF_BF = 1.55877649
REF_D = 28.12134928

def pareto_mask(bf, d):
    a = np.column_stack([bf, d])
    keep = np.ones(len(a), dtype=bool)
    for i in range(len(a)):
        dominated = ((a[:,0] >= a[i,0]) & (a[:,1] <= a[i,1]) &
                     ((a[:,0] > a[i,0]) | (a[:,1] < a[i,1])))
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep

def cls(actual, pred):
    tp = int((actual & pred).sum())
    fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    tn = int((~actual & ~pred).sum())
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Precision=precision, Recall=recall, F1=f1)

def main():
    common = pd.read_csv(DATA / "common_candidates_30000.csv")
    physical = pd.read_csv(GLOBAL / "all_30000_physical_evaluated.csv")
    df = common.merge(
        physical[["candidate_id","BF_physical","Delta_BF_physical"]],
        on="candidate_id", how="inner"
    )

    df["Physical_Pareto"] = pareto_mask(df.BF_physical.values, df.Delta_BF_physical.values)
    df["XGB_Pareto"] = pareto_mask(df.BF_XGB.values, df.Delta_BF_XGB.values)
    df["RSM_Pareto"] = pareto_mask(df.BF_RSM.values, df.Delta_BF_RSM.values)

    rows = []
    for method, bf, dd in [
        ("XGBoost","BF_XGB","Delta_BF_XGB"),
        ("RSM","BF_RSM","Delta_BF_RSM")]:
        sb, pb = spearmanr(df[bf], df.BF_physical)
        kb, _ = kendalltau(df[bf], df.BF_physical)
        sd, pdv = spearmanr(df[dd], df.Delta_BF_physical)
        kd, _ = kendalltau(df[dd], df.Delta_BF_physical)
        rows.append([method,sb,kb,sd,kd,pb,pdv])
    rank = pd.DataFrame(rows, columns=["Method","Spearman_BF","Kendall_BF",
                                        "Spearman_Delta_BF","Kendall_Delta_BF",
                                        "p_BF","p_Delta_BF"])

    err = []
    for method,bf,dd in [("XGBoost","BF_XGB","Delta_BF_XGB"),("RSM","BF_RSM","Delta_BF_RSM")]:
        eb = df[bf]-df.BF_physical
        ed = df[dd]-df.Delta_BF_physical
        err.append([method, np.mean(abs(eb)), np.sqrt(np.mean(eb**2)),
                    np.mean(abs(ed)), np.sqrt(np.mean(ed**2))])
    errors = pd.DataFrame(err, columns=["Method","BF_MAE","BF_RMSE","Delta_BF_MAE","Delta_BF_RMSE"])

    physical_p = df.Physical_Pareto.values
    pareto = pd.DataFrame(
        [dict(Method=m, **cls(physical_p, df[col].values))
         for m,col in [("XGBoost","XGB_Pareto"),("RSM","RSM_Pareto")]]
    )

    physical_w = ((df.BF_physical > REF_BF) & (df.Delta_BF_physical < REF_D)).values
    xgb_w = ((df.BF_XGB > REF_BF) & (df.Delta_BF_XGB < REF_D)).values
    rsm_w = ((df.BF_RSM > REF_BF) & (df.Delta_BF_RSM < REF_D)).values
    winwin = pd.DataFrame(
        [dict(Method=m, **cls(physical_w, p))
         for m,p in [("XGBoost",xgb_w),("RSM",rsm_w)]]
    )

    # Distance to the physical Pareto, using normalized objectives.
    pp = df[df.Physical_Pareto]
    br = max(df.BF_physical.max()-df.BF_physical.min(),1e-12)
    dr = max(df.Delta_BF_physical.max()-df.Delta_BF_physical.min(),1e-12)
    dists=[]
    for _,r in df.iterrows():
        d=np.sqrt(((r.BF_physical-pp.BF_physical.values)/br)**2 +
                  ((r.Delta_BF_physical-pp.Delta_BF_physical.values)/dr)**2)
        dists.append(d.min())
    df["Distance_to_Physical_Pareto"]=dists

    near=[]
    for pct in [1,5,10]:
        n=max(10,int(len(df)*pct/100))
        z=df.nsmallest(n,"Distance_to_Physical_Pareto")
        for method,bf,dd in [("XGBoost","BF_XGB","Delta_BF_XGB"),("RSM","BF_RSM","Delta_BF_RSM")]:
            near.append([f"Closest {pct}%",n,method,
                          spearmanr(z[bf],z.BF_physical).statistic,
                          spearmanr(z[dd],z.Delta_BF_physical).statistic])
    near=pd.DataFrame(near,columns=["Neighborhood","N","Method","Spearman_BF","Spearman_Delta_BF"])

    df.to_csv(OUT/"point_level_ranking_diagnostic.csv",index=False)
    rank.to_csv(OUT/"rank_correlation_global.csv",index=False)
    errors.to_csv(OUT/"prediction_error_comparison.csv",index=False)
    pareto.to_csv(OUT/"pareto_classification.csv",index=False)
    winwin.to_csv(OUT/"winwin_classification.csv",index=False)
    near.to_csv(OUT/"near_pareto_ranking.csv",index=False)

    print("\nGLOBAL RANK CORRELATION")
    print(rank.to_string(index=False, float_format=lambda x:f"{x:.8f}"))
    print("\nPREDICTION ERROR")
    print(errors.to_string(index=False, float_format=lambda x:f"{x:.8f}"))
    print("\nPHYSICAL PARETO CLASSIFICATION")
    print(pareto.to_string(index=False, float_format=lambda x:f"{x:.6f}"))
    print("\nPHYSICAL WIN-WIN CLASSIFICATION")
    print(winwin.to_string(index=False, float_format=lambda x:f"{x:.6f}"))
    print("\nRANKING NEAR PHYSICAL PARETO")
    print(near.to_string(index=False, float_format=lambda x:f"{x:.8f}"))
    print("\nFiles saved in:", OUT)

if __name__=="__main__":
    main()
