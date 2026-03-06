"""Utilities extracted and cleaned from the paper notebook.

This module was generated from the submission notebook and organized for
version control / GitHub upload.
"""

from __future__ import annotations

import math
import os
import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_X_y, check_array
from statsmodels.stats.contingency_tables import mcnemar

try:
    import neurokit2 as nk
except Exception:  # pragma: no cover
    nk = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    import wfdb
except Exception:  # pragma: no cover
    wfdb = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover
    keras = None
    layers = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from .preprocessing import *



def bootstrap_ci_stat(y_true, scores, stat_fn, B=2000, rng=rng):
    """Generic bootstrap CI for a scalar stat like AUROC or accuracy."""
    vals=[]
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    for _ in range(B):
        idx = rng.randint(0, n, n)
        vals.append(stat_fn(y_true[idx], scores[idx]))
    return np.percentile(vals, [2.5, 97.5])



def to_df_print(df_, title=None):
    print("\n" + (title or ""))
    display(df_.style.format({c:"{:.3f}" for c in df_.select_dtypes('number').columns}))



def auc_ci_delong(y_true, y_scores):
    # Minimal DeLong via https://github.com/yandexdataschool/roc_comparison (drop-in util recommended)
    # If not available, bootstrap 2000x as fallback for CIs and pairwise p-values.
    pass  # For speed: do bootstrap CIs; report paired difference p-values by permutation.



def bootstrap_auc(Xs, ys, B=2000):
    aucs=[]
    for _ in range(B):
        idx = rng.randint(0,len(ys),len(ys))
        try:
            aucs.append(roc_auc_score(ys[idx], Xs[idx]))
        except Exception:
            continue
    low, high = np.percentile(aucs,[2.5,97.5])
    return (np.mean(aucs), low, high)



def net_benefit(y_true, y_prob, thresholds):
    """
    Net Benefit:
      NB(t) = TP/N - FP/N * (t / (1 - t))
    for decision rule: treat if p >= t
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    N = len(y)
    prev = y.mean() if N > 0 else np.nan

    nb_model = []
    nb_all = []
    nb_none = []

    for t in thresholds:
        if t <= 0 or t >= 1:
            nb_model.append(np.nan); nb_all.append(np.nan); nb_none.append(0.0)
            continue

        pred_pos = p >= t
        TP = np.sum((pred_pos) & (y == 1))
        FP = np.sum((pred_pos) & (y == 0))

        w = t / (1 - t)

        nb = (TP / N) - (FP / N) * w
        nb_model.append(nb)

        # Treat-all: everyone treated
        TP_all = np.sum(y == 1)
        FP_all = np.sum(y == 0)
        nbA = (TP_all / N) - (FP_all / N) * w
        nb_all.append(nbA)

        # Treat-none: NB = 0
        nb_none.append(0.0)

    return {
        "thresholds": np.asarray(thresholds),
        "nb_model": np.asarray(nb_model),
        "nb_all": np.asarray(nb_all),
        "nb_none": np.asarray(nb_none),
        "prevalence": prev
    }



def band(a):
    if pd.isna(a): return np.nan
    if a < 1: return "<1y"
    elif a < 5: return "1-5y"
    else: return ">5y"



def cv_auc(cols):
    if len(cols)==0: return np.nan
    return cross_val_score(xgb, X[cols], y, cv=cv, scoring='roc_auc').mean()



def family_auc_CI(cols):
    if len(cols)==0: return np.nan, np.nan, np.nan
    proba = cross_val_predict(xgb_final, X[cols], y, cv=cv, method='predict_proba')[:,1]
    return bootstrap_auc(proba, y)



def age_band_five(a):
    if a < 1: return "<1y"
    elif a < 2: return "1–2y"
    elif a < 4: return "2–4y"
    elif a < 6: return "4–6y"
    elif a < 8: return "6–8y"

    elif a < 12: return "8–12y"
    else: return "≥12y"



def age_band(a):
    if pd.isna(a):
        return "unknown"
    if a < 1:
        return "<1y"
    elif a < 3:
        return "1–3y"
    elif a < 6:
        return "3–6y"
    elif a < 12:
        return "6–12y"
    else:
        return ">=12y"



def _sens_spec_at_threshold(y_true, y_prob, thr_ruleout=0.33, thr_rulein=0.67):
    """
    Rule-out sensitivity: among true positives, fraction with prob >= thr_ruleout (not ruled out)
    Rule-in specificity: among true negatives, fraction with prob <  thr_rulein  (not ruled in)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    pos = (y_true == 1)
    neg = (y_true == 0)

    # "rule-out" decision: if prob < thr_ruleout => ruled-out (predict negative)
    # sensitivity of NOT missing positives at that rule-out boundary:
    # positives should have prob >= thr_ruleout
    sens = np.nan
    if pos.sum() > 0:
        sens = (y_prob[pos] >= thr_ruleout).mean()

    # "rule-in" decision: if prob >= thr_rulein => ruled-in (predict positive)
    # specificity of NOT falsely ruling-in negatives:
    # negatives should have prob < thr_rulein
    spec = np.nan
    if neg.sum() > 0:
        spec = (y_prob[neg] < thr_rulein).mean()

    return sens, spec



def bootstrap_ci(y_true, y_prob, thr_ruleout, thr_rulein, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    sens_list, spec_list = [], []
    idx_all = np.arange(n)

    for _ in range(B):
        boot = rng.choice(idx_all, size=n, replace=True)
        s, p = _sens_spec_at_threshold(y_true[boot], y_prob[boot], thr_ruleout, thr_rulein)
        sens_list.append(s)
        spec_list.append(p)

    sens_arr = np.asarray(sens_list, dtype=float)
    spec_arr = np.asarray(spec_list, dtype=float)

    def _ci(a):
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return (np.nan, np.nan)
        return (np.quantile(a, 0.025), np.quantile(a, 0.975))

    return _ci(sens_arr), _ci(spec_arr)



def subgroup_table(df_eval, y_col="_label", prob_col="proba_xgb",
                   subgroup_specs=(("Sex", "sex_band"), ("Age band", "age_band"), ("Tachycardia", "tachy")),
                   min_n=150, B=2000, seed=42):

    rows = []
    for group_name, col in subgroup_specs:
        if col not in df_eval.columns:
            continue

        for level in pd.Series(df_eval[col]).dropna().unique():
            idx = df_eval.index[df_eval[col] == level]
            if len(idx) < min_n:
                continue

            y_true = df_eval.loc[idx, y_col].astype(int)
            y_prob = df_eval.loc[idx, prob_col].astype(float)

            # skip if only one class
            if y_true.nunique() < 2:
                continue

            auc = roc_auc_score(y_true, y_prob)
            ci_lo, ci_hi = bootstrap_ci_auc(y_true, y_prob, B=B, seed=seed)
            brier = brier_score_loss(y_true, y_prob)

            n = len(idx)
            n_pos = int((y_true == 1).sum())
            pos_pct = 100.0 * n_pos / n

            rows.append({
                "Group": group_name,
                "Level": str(level),
                "N": n,
                "N_pos": n_pos,
                "Pos_%": pos_pct,
                "AUROC": auc,
                "AUROC_CI_lo": ci_lo,
                "AUROC_CI_hi": ci_hi,
                "Brier": brier
            })

    tbl = pd.DataFrame(rows)
    if len(tbl) == 0:
        return tbl

    # Nice ordering
    tbl = tbl.sort_values(["Group", "Level"]).reset_index(drop=True)
    return tbl



def bootstrap_ci_auc(y_true, y_prob, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    aucs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) < 50:
        return np.nan, np.nan
    return np.quantile(aucs, 0.025), np.quantile(aucs, 0.975)



def bootstrap_ci_metric(y_true, y_prob, metric_fn, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        # Some metrics need both classes (AUROC)
        try:
            v = metric_fn(y_true[idx], y_prob[idx])
        except Exception:
            continue
        if np.isnan(v):
            continue
        vals.append(v)

    if len(vals) < 50:
        return np.nan, np.nan
    return np.quantile(vals, 0.025), np.quantile(vals, 0.975)



def auc_metric(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)



def ruleout_sens_npv(y, p, thr_low=0.33):
    """Rule-out set = p <= thr_low. We want high sensitivity (catch positives) and high NPV."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    ro = p <= thr_low

    # Sensitivity among positives: fraction of positives that are NOT rule-out? Wait:
    # If we "rule out CHD" when p<=thr_low, then positives in rule-out are false negatives.
    # So sensitivity of 'NOT ruled out' = 1 - FN rate in rule-out.
    # Equivalent: Sens = P(p > thr_low | y=1)
    sens = np.mean(p[y == 1] > thr_low) if np.any(y == 1) else np.nan

    # NPV among rule-out group: P(y=0 | p<=thr_low)
    if np.any(ro):
        npv = np.mean(y[ro] == 0)
    else:
        npv = np.nan

    return sens, npv



def rulein_spec_ppv(y, p, thr_high=0.67):
    """Rule-in set = p >= thr_high. We want high specificity and high PPV."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    ri = p >= thr_high

    # Specificity among negatives: P(p < thr_high | y=0)
    spec = np.mean(p[y == 0] < thr_high) if np.any(y == 0) else np.nan

    # PPV among rule-in group: P(y=1 | p>=thr_high)
    if np.any(ri):
        ppv = np.mean(y[ri] == 1)
    else:
        ppv = np.nan

    return spec, ppv



def subgroup_table_enhanced(df_eval, y_col="_label", prob_col="proba",
                            subgroup_specs=(("Sex","sex_band"),
                                            ("Age band","age_band"),
                                            ("HR band","hr_band"),
                                            ("Tachycardia","tachy_100"),
                                            ("QRS","qrs_120")),
                            thr_low=0.33, thr_high=0.67,
                            min_n=150, B=2000, seed=42):

    rows = []
    for group_name, col in subgroup_specs:
        if col not in df_eval.columns:
            continue

        for level in pd.Series(df_eval[col]).dropna().unique():
            sub = df_eval[df_eval[col] == level]
            if len(sub) < min_n:
                continue

            y_true = sub[y_col].astype(int).values
            y_prob = sub[prob_col].astype(float).values

            if len(np.unique(y_true)) < 2:
                continue

            # AUROC + CI
            auc = roc_auc_score(y_true, y_prob)
            ci_lo, ci_hi = bootstrap_ci_metric(
                y_true, y_prob,
                metric_fn=lambda yy, pp: auc_metric(yy, pp),
                B=B, seed=seed
            )

            # Brier
            brier = brier_score_loss(y_true, y_prob)

            # Rule-out sens/NPV + bootstrap CI
            sens, npv = ruleout_sens_npv(y_true, y_prob, thr_low=thr_low)
            sens_lo, sens_hi = bootstrap_ci_metric(
                y_true, y_prob,
                metric_fn=lambda yy, pp: ruleout_sens_npv(yy, pp, thr_low=thr_low)[0],
                B=B, seed=seed
            )
            npv_lo, npv_hi = bootstrap_ci_metric(
                y_true, y_prob,
                metric_fn=lambda yy, pp: ruleout_sens_npv(yy, pp, thr_low=thr_low)[1],
                B=B, seed=seed
            )

            # Rule-in spec/PPV + bootstrap CI
            spec, ppv = rulein_spec_ppv(y_true, y_prob, thr_high=thr_high)
            spec_lo, spec_hi = bootstrap_ci_metric(
                y_true, y_prob,
                metric_fn=lambda yy, pp: rulein_spec_ppv(yy, pp, thr_high=thr_high)[0],
                B=B, seed=seed
            )
            ppv_lo, ppv_hi = bootstrap_ci_metric(
                y_true, y_prob,
                metric_fn=lambda yy, pp: rulein_spec_ppv(yy, pp, thr_high=thr_high)[1],
                B=B, seed=seed
            )

            n = len(sub)
            n_pos = int((y_true == 1).sum())
            pos_pct = 100.0 * n_pos / n

            rows.append({
                "Group": group_name,
                "Level": str(level),
                "N": n,
                "N_pos": n_pos,
                "Pos_%": pos_pct,
                "AUROC": auc, "AUROC_CI_lo": ci_lo, "AUROC_CI_hi": ci_hi,
                "Brier": brier,
                "RuleOut_Sens": sens, "RuleOut_Sens_lo": sens_lo, "RuleOut_Sens_hi": sens_hi,
                "RuleOut_NPV": npv,  "RuleOut_NPV_lo": npv_lo,  "RuleOut_NPV_hi": npv_hi,
                "RuleIn_Spec": spec, "RuleIn_Spec_lo": spec_lo, "RuleIn_Spec_hi": spec_hi,
                "RuleIn_PPV": ppv,   "RuleIn_PPV_lo": ppv_lo,   "RuleIn_PPV_hi": ppv_hi,
            })

    tbl = pd.DataFrame(rows)
    if len(tbl) == 0:
        return tbl

    return tbl.sort_values(["Group","Level"]).reset_index(drop=True)



def calibration_bins(y_true, y_prob, n_bins=10, strategy="quantile"):
    """
    Per-bin mean predicted prob, observed event rate, and count.
    strategy:
      - "quantile": equal-count bins (recommended for imbalanced data)
      - "uniform": equal-width bins
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)

    valid = np.isfinite(p) & np.isfinite(y)
    y, p = y[valid], p[valid]

    if len(y) == 0:
        return []

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        # if too few unique edges, fall back
        if len(edges) < 3:
            strategy = "uniform"

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # assign bins (right-closed)
    bin_ids = np.digitize(p, edges[1:-1], right=True)

    bins = []
    for b in range(len(edges) - 1):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        bins.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b+1]),
            "n": int(idx.sum()),
            "p_mean": float(p[idx].mean()),
            "y_rate": float(y[idx].mean()),
        })
    return bins



def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile"):
    bins = calibration_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if len(bins) == 0:
        return np.nan
    N = sum(b["n"] for b in bins)
    ece = 0.0
    for b in bins:
        ece += (b["n"] / N) * abs(b["y_rate"] - b["p_mean"])
    return float(ece)



def metrics_at_threshold(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan

    return dict(
        Threshold=thr,
        Accuracy=accuracy_score(y_true, y_pred),
        BalAcc=balanced_accuracy_score(y_true, y_pred),
        Sensitivity=sens,
        Specificity=spec,
        PPV=ppv,
        NPV=npv,
        TP=tp, FP=fp, TN=tn, FN=fn
    )



def find_thresholds(y, p, target_sens=0.95, target_spec=0.95, grid_size=2001):
    # Use a dense grid between 0 and 1 (stable + fast)
    grid = np.linspace(0, 1, grid_size)

    rows = [metrics_at_threshold(y, p, t) for t in grid]
    M = pd.DataFrame(rows)

    # Rule-out: sens >= target_sens, choose highest t (best specificity under constraint)
    M_ro = M[M["sens"] >= target_sens].copy()
    t_ruleout = float(M_ro.sort_values(["t"], ascending=False).iloc[0]["t"]) if len(M_ro) else np.nan

    # Rule-in: spec >= target_spec, choose lowest t (best sensitivity under constraint)
    M_ri = M[M["spec"] >= target_spec].copy()
    t_rulein = float(M_ri.sort_values(["t"], ascending=True).iloc[0]["t"]) if len(M_ri) else np.nan

    return t_ruleout, t_rulein, M



def clinical_table(y, p, t_ruleout, t_rulein, per=1000):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    ro = metrics_at_threshold(y, p, t_ruleout)
    ri = metrics_at_threshold(y, p, t_rulein)

    # Operational counts per 1000 ECGs
    N = len(y)
    scale = per / N

    # If workflow is:
    #   low-risk (p < t1)  -> rule-out (no echo)
    #   high-risk (p >= t2)-> refer echo
    #   intermediate       -> monitor / follow-up
    n_low  = int((p <  t_ruleout).sum())
    n_high = int((p >= t_rulein).sum())
    n_mid  = N - n_low - n_high

    # Missed CHD per 1000 at rule-out threshold:
    # these are CHD in low-risk region (p < t1)
    missed_chd = int(((p < t_ruleout) & (y == 1)).sum())

    # "Echo avoided" depends on what your baseline is.
    # Common baseline: echo everyone ("treat-all").
    # Under triage, you echo only high-risk (or high+mid if you choose).
    # Here: echo only high-risk.
    echos_if_all = N
    echos_triage = n_high
    echos_avoided = echos_if_all - echos_triage

    # Optional: unnecessary echoes among high-risk (false positives at t2):
    fp_high = int(((p >= t_rulein) & (y == 0)).sum())

    table = pd.DataFrame([
        {
            "Decision": "Rule-out",
            "Threshold": t_ruleout,
            "Sensitivity": ro["sens"],
            "Specificity": ro["spec"],
            "PPV": ro["ppv"],
            "NPV": ro["npv"],
        },
        {
            "Decision": "Rule-in",
            "Threshold": t_rulein,
            "Sensitivity": ri["sens"],
            "Specificity": ri["spec"],
            "PPV": ri["ppv"],
            "NPV": ri["npv"],
        }
    ])

    extras = {
        "N_total": N,
        "N_low": n_low,
        "N_mid": n_mid,
        "N_high": n_high,
        "missed_chd_per_1000": missed_chd * scale,
        "echos_avoided_per_1000": echos_avoided * scale,
        "unnecessary_highrisk_echos_per_1000": fp_high * scale
    }

    return table, extras



def confusion_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return dict(TN=tn, FP=fp, FN=fn, TP=tp)



def band_metrics(band, proba=proba_xgb, target='sens', value=0.85):
    idx = df.index[df["age_band5"]==band]
    yt  = y.loc[idx].values
    pt  = proba[idx]
    thr, se, sp = curve_at_target(yt, pt, target=target, value=value)
    pred = (pt >= thr).astype(int)
    acc = accuracy_score(yt, pred)
    prec = precision_score(yt, pred, zero_division=0)
    rec  = recall_score(yt, pred)
    cm   = confusion_counts(yt, pred)
    return dict(Band=band, Threshold=thr, Sensitivity=se, Specificity=sp, Accuracy=acc, Precision=prec, **cm)



def ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error with equal-width bins over [0,1]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        nk = m.sum()
        if nk == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (nk / n) * abs(acc - conf)
    return float(ece)



def pooled_cv_metrics(model, X, y, label):
    """Return dict of AUROC, Acc@0.5, Acc@Youden, Se/Sp@Youden, Brier, ECE, LogLoss."""
    t0 = time.time()
    p = cross_val_predict(model, X, y, cv=cv5, method="predict_proba")[:,1]
    auc = roc_auc_score(y, p)
    acc05 = accuracy_score(y, (p>=0.5).astype(int))
    t_opt, se, sp = youden_threshold(y, p)
    accJ = accuracy_score(y, (p>=t_opt).astype(int))
    brier = brier_score_loss(y, p)
    ece   = ece_score(y, p, n_bins=10)
    lloss = log_loss(y, np.c_[1-p, p], labels=[0,1])
    t1 = time.time()
    return dict(Model=label, AUROC=auc, Acc_05=acc05, Acc_Youden=accJ, Se_Youden=se, Sp_Youden=sp,
                Brier=brier, ECE=ece, LogLoss=lloss, Seconds=t1-t0, Proba=p)



def to_table(df, title=""):
    print("\n"+title)
    display(df.style.format({c:"{:.3f}" for c in df.select_dtypes('number').columns}))



def pooled_cv_metrics_bal(model, Xsub, y, label, fset):
    p = cross_val_predict(model, Xsub, y, cv=cv5, method="predict_proba")[:,1]
    auc  = roc_auc_score(y, p)
    acc05= accuracy_score(y, (p>=0.5).astype(int))
    # Balanced accuracy at 0.5
    bacc05 = balanced_accuracy_score(y, (p>=0.5).astype(int))
    # Youden
    fpr,tpr,thr = roc_curve(y, p)
    k = (tpr - fpr).argmax()
    tJ = thr[k]; se=tpr[k]; sp=1-fpr[k]
    accJ  = accuracy_score(y, (p>=tJ).astype(int))
    baccJ = balanced_accuracy_score(y, (p>=tJ).astype(int))
    brier = brier_score_loss(y, p)
    ll    = log_loss(y, np.c_[1-p, p])
    return dict(Model=label, FeatureSet=fset, Dim=Xsub.shape[1],
                AUROC=auc, Acc_05=acc05, BalAcc_05=bacc05,
                Acc_Youden=accJ, BalAcc_Youden=baccJ, Se_Youden=se, Sp_Youden=sp,
                Brier=brier, LogLoss=ll, Proba=p)
