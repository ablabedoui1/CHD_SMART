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



def parse_age_days_to_years(age_str):
    if pd.isna(age_str):
        return np.nan
    m = re.search(r"(\d+)", str(age_str))
    if not m:
        return np.nan
    days = float(m.group(1))
    return days / 365.25



def parse_sex_to_bin(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("'", "").replace('"', "").lower()
    if s.startswith("f"):
        return 0.0
    if s.startswith("m"):
        return 1.0
    return np.nan



def find_meannn_col(cols):
    for c in cols:
        if re.search(r"meannn", c, flags=re.IGNORECASE):
            return c
    return None



def add_zscore_column_safe(df, df_ref, feature, degree=2,
                           age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                           suffix="_z", min_controls=200):
    # Fit on controls
    cols = [age_col, hr_col, sex_col, feature]
    dref = df_ref[cols].copy()
    for c in cols:
        dref[c] = _to_num(dref[c])
    dref = dref.dropna()
    if len(dref) < min_controls:
        df[f"{feature}{suffix}"] = np.nan
        print(f"[skip] {feature}: not enough clean controls (n={len(dref)})")
        return df

    A, H, S = dref[age_col].values, dref[hr_col].values, dref[sex_col].values
    X = pd.DataFrame({"Age":A,"HR":H,"Sex":S,"Age2":A**2,"HR2":H**2,"AgeHR":A*H})
    if degree >= 3:
        X["Age3"]=A**3; X["HR3"]=H**3; X["Age2HR"]=(A**2)*H; X["AgeHR2"]=A*(H**2)

    y = dref[feature].values
    m = LinearRegression().fit(X, y)
    resid = y - m.predict(X)
    sigma = np.std(resid, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        df[f"{feature}{suffix}"] = np.nan
        print(f"[skip] {feature}: sigma invalid")
        return df

    # predict for all rows with covariates
    tmp = df[[age_col, hr_col, sex_col, feature]].copy()
    for c in cols:
        tmp[c] = _to_num(tmp[c])
    ok = tmp[cols].notna().all(axis=1)

    A = tmp.loc[ok, age_col].values
    H = tmp.loc[ok, hr_col].values
    S = tmp.loc[ok, sex_col].values

    Xall = pd.DataFrame({"Age":A,"HR":H,"Sex":S,"Age2":A**2,"HR2":H**2,"AgeHR":A*H})
    if degree >= 3:
        Xall["Age3"]=A**3; Xall["HR3"]=H**3; Xall["Age2HR"]=(A**2)*H; Xall["AgeHR2"]=A*(H**2)

    mu = m.predict(Xall[X.columns])
    z = np.full(len(df), np.nan)
    z[ok.values] = (tmp.loc[ok, feature].values - mu) / sigma
    df[f"{feature}{suffix}"] = z

    print(f"{feature}: z computed for {np.isfinite(z).sum()} / {len(df)} rows (missing covariates in {(~ok).sum()})")
    return df



def cols_matching(df, pat):
    return [c for c in df.columns if re.search(pat, c, flags=re.IGNORECASE)]



def add_zscore_column_safe_v2(df, df_ref, feature, degree=2,
                              age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                              suffix="_z", min_controls=200):
    """
    Robust dev-normalization:
    - fits on controls with complete Age/HR/Sex + feature
    - skips feature if too sparse in controls (min_controls)
    - predicts only on rows with complete covariates
    - keeps feature-name consistency (no sklearn warning)
    """
    needed = [age_col, hr_col, sex_col, feature]

    # --- controls clean ---
    dref = df_ref[needed].copy()
    for c in needed:
        dref[c] = pd.to_numeric(dref[c], errors="coerce")
    dref = dref.dropna()

    if len(dref) < min_controls:
        print(f"[skip] {feature}: only {len(dref)} clean controls (<{min_controls})")
        # still create the column so downstream code doesn't break
        out = df.copy()
        out[f"{feature}{suffix}"] = np.nan
        return out

    A = dref[age_col].values
    H = dref[hr_col].values
    S = dref[sex_col].values

    X = pd.DataFrame({
        "Age": A,
        "HR": H,
        "Sex": S,
        "Age2": A**2,
        "HR2": H**2,
        "AgeHR": A*H,
    })

    if degree >= 3:
        X["Age3"] = A**3
        X["HR3"] = H**3
        X["Age2HR"] = (A**2) * H
        X["AgeHR2"] = A * (H**2)

    y = dref[feature].values
    model = LinearRegression().fit(X, y)

    resid = y - model.predict(X)
    sigma = np.std(resid, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        print(f"[skip] {feature}: sigma invalid")
        out = df.copy()
        out[f"{feature}{suffix}"] = np.nan
        return out

    # --- apply to all rows ---
    out = df.copy()
    tmp = out[needed].copy()
    for c in needed:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    A = tmp[age_col].values
    H = tmp[hr_col].values
    S = tmp[sex_col].values

    Xall = pd.DataFrame({
        "Age": A,
        "HR": H,
        "Sex": S,
        "Age2": A**2,
        "HR2": H**2,
        "AgeHR": A*H,
    })

    if degree >= 3:
        Xall["Age3"] = A**3
        Xall["HR3"] = H**3
        Xall["Age2HR"] = (A**2) * H
        Xall["AgeHR2"] = A * (H**2)

    ok = np.isfinite(Xall).all(axis=1) & np.isfinite(tmp[feature].values)

    z = np.full(len(out), np.nan, dtype=float)
    mu = np.full(len(out), np.nan, dtype=float)
    mu[ok] = model.predict(Xall.loc[ok])   # <-- DataFrame with same columns => no warning
    z[ok] = (tmp[feature].values[ok] - mu[ok]) / sigma

    out[f"{feature}{suffix}"] = z
    print(f"{feature}: z computed for {np.isfinite(z).sum()} / {len(out)} rows (missing in {(~np.isfinite(z)).sum()})")
    return out



def keep_if_enough_numeric(df, col, min_non_nan=500):
    x = pd.to_numeric(df[col], errors="coerce")
    return x.notna().sum() >= min_non_nan



def to_numeric_series(s: pd.Series) -> pd.Series:
    """Robust numeric cast (strings->float), keeps NaN if not parseable."""
    return pd.to_numeric(s, errors="coerce")



def get_numeric_feature_cols(df_in: pd.DataFrame, min_nonnull=200) -> list:
    """
    Choose candidate feature columns:
    - not meta cols
    - can be converted to numeric
    - has enough non-null numeric values
    """
    feats = []
    for c in df_in.columns:
        if c in META_COLS:
            continue
        s = to_numeric_series(df_in[c])
        if s.notna().sum() >= min_nonnull:
            feats.append(c)
    return feats



def build_design(df_in: pd.DataFrame, degree_age=2, degree_hr=2, include_interactions=False):
    """
    Create design matrix for developmental norm model:
      f ~ poly(Age, degree_age) + poly(HR, degree_hr) + Sex
    Optionally include interactions via PolynomialFeatures on [Age, HR, Sex].
    """
    Xbase = df_in[["Age_years", "HR_bpm", "Sex_bin"]].copy()
    # Ensure numeric
    for c in Xbase.columns:
        Xbase[c] = pd.to_numeric(Xbase[c], errors="coerce")

    if include_interactions:
        # Interactions among Age/HR/Sex (can be heavy; use carefully)
        pf = PolynomialFeatures(degree=max(degree_age, degree_hr), include_bias=True)
        X = pf.fit_transform(Xbase.values)
        names = pf.get_feature_names_out(["Age", "HR", "Sex"])
        return X, names, pf

    # No interactions: manual polynomial terms for Age and HR + Sex
    X = pd.DataFrame(index=df_in.index)
    X["bias"] = 1.0
    # Age poly
    for d in range(1, degree_age + 1):
        X[f"Age^{d}"] = Xbase["Age_years"] ** d
    # HR poly
    for d in range(1, degree_hr + 1):
        X[f"HR^{d}"] = Xbase["HR_bpm"] ** d
    # Sex
    X["Sex"] = Xbase["Sex_bin"]
    return X.values, X.columns.tolist(), None



def predict_mu(df_any: pd.DataFrame, fit_obj, degree_age=2, degree_hr=2, include_interactions=False):
    """
    Predict mu(Age,HR,Sex) for any dataframe (controls or cases).
    Returns mu array aligned to df_any index (NaN where covariates missing).
    """
    Xbase = df_any[["Age_years", "HR_bpm", "Sex_bin"]].copy()
    for c in Xbase.columns:
        Xbase[c] = pd.to_numeric(Xbase[c], errors="coerce")

    mu = pd.Series(index=df_any.index, dtype="float64")

    valid = Xbase.notna().all(axis=1)
    if valid.sum() == 0:
        mu.loc[:] = np.nan
        return mu.values

    if include_interactions:
        pf = fit_obj["poly_obj"]
        Xv = pf.transform(Xbase.loc[valid].values)
    else:
        # manual design must match build_design()
        Xv_df = pd.DataFrame(index=Xbase.loc[valid].index)
        Xv_df["bias"] = 1.0
        for d in range(1, degree_age + 1):
            Xv_df[f"Age^{d}"] = Xbase.loc[valid, "Age_years"] ** d
        for d in range(1, degree_hr + 1):
            Xv_df[f"HR^{d}"] = Xbase.loc[valid, "HR_bpm"] ** d
        Xv_df["Sex"] = Xbase.loc[valid, "Sex_bin"]
        Xv = Xv_df.values

    mu.loc[valid] = fit_obj["model"].predict(Xv)
    return mu.values



def _quantile_clip(x, lo=0.01, hi=0.99):
    x = np.asarray(x)
    return np.nanquantile(x, lo), np.nanquantile(x, hi)



def run_devnorm_qc_for_features(
    df_controls,
    features,
    degree_age=2,
    degree_hr=2,
    include_interactions=False,
    max_features=30,     # safety: don’t explode runtime
):
    done = 0
    failed = []

    for feat in features:
        if done >= max_features:
            break
        try:
            fit_obj = fit_developmental_model_one_feature(
                df_controls,
                feat,
                degree_age=degree_age,
                degree_hr=degree_hr,
                include_interactions=include_interactions
            )

            # Save plots
            base = re.sub(r"[^A-Za-z0-9_\-]+", "_", feat)

            plot_partial_age_curve(
                df_controls, feat, fit_obj,
                degree_age, degree_hr, include_interactions,
                savepath=os.path.join(OUT_DIR, f"{base}__vs_age.png")
            )
            plot_partial_hr_curve(
                df_controls, feat, fit_obj,
                degree_age, degree_hr, include_interactions,
                savepath=os.path.join(OUT_DIR, f"{base}__vs_hr.png")
            )
            plot_sex_stratified_age_curves(
                df_controls, feat, fit_obj,
                degree_age, degree_hr, include_interactions,
                savepath=os.path.join(OUT_DIR, f"{base}__sex_curves.png")
            )
            plot_residual_histogram(
                fit_obj, feat,
                savepath=os.path.join(OUT_DIR, f"{base}__resid_hist.png")
            )

            done += 1
            if done % 5 == 0:
                print(f"Done {done}/{min(max_features, len(features))} ...")

        except Exception as e:
            failed.append((feat, str(e)))

    print(f"\nFinished. Plotted: {done}. Failed: {len(failed)}")
    if failed:
        print("First 10 failures:")
        for f, msg in failed[:10]:
            print(" -", f, "=>", msg)

    return failed



def compute_zscores(df_all, df_controls, features, degree_age=2, degree_hr=2, include_interactions=False):
    """
    Fit mu() and sigma_res on controls, then compute:
      z = (f - mu)/sigma_res
    for all samples (controls + cases).
    Returns a dataframe with new columns <feature>_z
    """
    df_out = df_all.copy()
    z_cols = {}

    for feat in features:
        # Fit on controls
        fit_obj = fit_developmental_model_one_feature(
            df_controls, feat, degree_age, degree_hr, include_interactions
        )
        mu_all = predict_mu(df_out, fit_obj, degree_age, degree_hr, include_interactions)
        f_all = pd.to_numeric(df_out[feat], errors="coerce").values
        z = (f_all - mu_all) / (fit_obj["sigma_res"] if fit_obj["sigma_res"] > 0 else np.nan)
        z_cols[f"{feat}_z"] = z

    df_z = pd.DataFrame(z_cols, index=df_out.index)
    df_out = pd.concat([df_out, df_z], axis=1)
    return df_out



def _safe_makedirs(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)



def _to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")



def _select_feature_cols(df, include_regex=None, exclude_cols=None, min_non_nan=200):
    """
    Heuristic selector to pick numeric feature columns from a wide ECG dataframe.
    - include_regex: list of regex patterns; if provided, keep cols matching any.
    - exclude_cols: set/list of cols to skip (metadata etc.)
    - min_non_nan: require at least this many non-NaN numeric values
    """
    if exclude_cols is None:
        exclude_cols = set()
    else:
        exclude_cols = set(exclude_cols)

    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue

        if include_regex is not None:
            ok = any(re.search(pat, c, flags=re.IGNORECASE) for pat in include_regex)
            if not ok:
                continue

        # Try numeric conversion quickly (sample-based to avoid heavy conversion)
        x = pd.to_numeric(df[c], errors="coerce")
        n_ok = x.notna().sum()
        if n_ok >= min_non_nan:
            cols.append(c)
    return cols



def predict_mu_sigma(df, model, design_columns,
                     age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin"):
    """
    Build design matrix and get mu_hat for any dataframe.
    """
    A = _to_numeric_series(df[age_col]).values
    H = _to_numeric_series(df[hr_col]).values
    S = _to_numeric_series(df[sex_col]).values

    X = pd.DataFrame({
        "Age": A,
        "HR": H,
        "Sex": S,
        "Age2": A**2,
        "HR2": H**2,
        "AgeHR": A*H,
        "Age3": A**3,
        "HR3": H**3,
        "Age2HR": (A**2)*H,
        "AgeHR2": A*(H**2),
    })

    # keep only columns the model was trained on (degree=2 vs 3)
    X = X[design_columns]
    mu = model.predict(X)
    return mu



def add_zscore_column(df, df_ref, feature, degree=2,
                      age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                      out_col_suffix="_z"):
    """
    Fits dev model on df_ref and adds:
      - feature + "_mu"
      - feature + "_z"
    to df (returned copy).
    """
    model, sigma, design_cols = fit_dev_model(df_ref, feature, degree=degree,
                                              age_col=age_col, hr_col=hr_col, sex_col=sex_col)
    out = df.copy()

    # Need numeric feature too
    out[feature] = _to_numeric_series(out[feature])
    mu = predict_mu_sigma(out, model, design_cols, age_col=age_col, hr_col=hr_col, sex_col=sex_col)
    out[f"{feature}_mu"] = mu

    if sigma is None or np.isnan(sigma) or sigma == 0:
        out[f"{feature}{out_col_suffix}"] = np.nan
    else:
        out[f"{feature}{out_col_suffix}"] = (out[feature] - out[f"{feature}_mu"]) / sigma

    return out



def age_correlation_safe(df, feature, age_col="Age_years"):
    """
    Spearman correlation Age vs feature, and Age vs feature_z
    Robust to strings/object types.
    """
    age = pd.to_numeric(df[age_col], errors="coerce")
    raw = pd.to_numeric(df[feature], errors="coerce")
    z   = pd.to_numeric(df.get(f"{feature}_z"), errors="coerce")  # may not exist yet

    # RAW
    mask_raw = age.notna() & raw.notna()
    r_raw = np.nan
    if mask_raw.sum() >= 10:
        r_raw, _ = spearmanr(age[mask_raw], raw[mask_raw])

    # Z
    r_z = np.nan
    if z is not None:
        mask_z = age.notna() & z.notna()
        if mask_z.sum() >= 10:
            r_z, _ = spearmanr(age[mask_z], z[mask_z])

    return r_raw, r_z, mask_raw.sum(), (mask_z.sum() if z is not None else 0)



def age_bin_means_safe(df, feature, age_col="Age_years", bins=(0,1,5,10,15,18)):
    """
    Mean of z-score in age bins (should be near 0 in controls).
    """
    tmp = df.copy()
    tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
    tmp[f"{feature}_z"] = pd.to_numeric(tmp.get(f"{feature}_z"), errors="coerce")

    tmp = tmp.dropna(subset=[age_col, f"{feature}_z"])
    tmp["age_bin"] = pd.cut(tmp[age_col], bins=bins, include_lowest=True)

    return tmp.groupby("age_bin")[f"{feature}_z"].agg(["count","mean","std"])



def spearman_age_table(df, df_ref, features, degree=2,
                       age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin"):
    rows = []
    out = df.copy()

    for f in features:
        # add z safely
        out, ok, msg = add_zscore_column_safe(out, df_ref, f, degree=degree,
                                              age_col=age_col, hr_col=hr_col, sex_col=sex_col)
        print(msg)
        if not ok:
            continue

        age = pd.to_numeric(out[age_col], errors="coerce")
        raw = pd.to_numeric(out[f], errors="coerce")
        z   = pd.to_numeric(out[f"{f}_z"], errors="coerce")

        m_raw = age.notna() & raw.notna()
        m_z   = age.notna() & z.notna()

        r_raw, p_raw = spearmanr(age[m_raw], raw[m_raw])
        r_z,   p_z   = spearmanr(age[m_z],   z[m_z])

        rows.append({
            "feature": f,
            "n_raw": int(m_raw.sum()),
            "rho_age_raw": float(r_raw),
            "p_age_raw": float(p_raw),
            "n_z": int(m_z.sum()),
            "rho_age_z": float(r_z),
            "p_age_z": float(p_z),
            "abs_rho_drop": abs(r_raw) - abs(r_z)
        })

    res = pd.DataFrame(rows).sort_values("abs_rho_drop", ascending=False)
    return res



def _to_num(s): return pd.to_numeric(s, errors="coerce")



def zscore_feature(name):
    """Fit age/HR/sex curves on controls; fallback to robust scaling."""
    y = pd.to_numeric(df[name], errors="coerce").astype(float).values
    if controls is not None and covars and controls.sum() >= 200:
        try:
            vc = controls.copy()
            for cv in covars:
                vc &= df[cv].notna()
            vc &= np.isfinite(y)
            if vc.sum() < 50:
                raise ValueError
            Xc = df.loc[vc, covars].astype(float).values
            yc = y[vc]
            lr = LinearRegression().fit(poly.fit_transform(Xc), yc)
            mu_all = lr.predict(poly.transform(df[covars].astype(float).values))
            resid = yc - lr.predict(poly.transform(Xc))
            sd = np.nanstd(resid, ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                raise ValueError
            return (y - mu_all) / sd
        except Exception:
            pass
    # Fallback: median/MAD on controls (or all if no controls)
    ref_mask = controls if controls is not None else ~np.isnan(y)
    ref_vals = y[ref_mask & np.isfinite(y)]
    if ref_vals.size < 50:
        return np.full_like(y, np.nan)
    med = np.nanmedian(ref_vals)
    mad = np.nanmedian(np.abs(ref_vals - med))
    scale = mad * 1.4826 if (mad > 0 and np.isfinite(mad)) else np.nanstd(ref_vals, ddof=1)
    return (y - med) / (scale if scale and scale > 0 else np.nan)



def laplacian_eigs(vals):
    """Compute eigenvalues of graph Laplacian for a set of lead values."""
    x = np.array(vals, dtype=float)
    if np.isnan(x).sum() > len(x)//2:
        return [np.nan, np.nan, np.nan]
    diffs = np.abs(x[:,None]-x[None,:])
    med   = np.nanmedian(diffs[~np.isnan(diffs)]) or 1.0
    A = np.exp(-diffs/med); np.fill_diagonal(A, 0.0)
    D = np.diag(np.nansum(A, axis=1))
    L = D - A
    try:
        w = la.eigvalsh(L); w = np.sort(w)
        return [w[1] if len(w)>1 else np.nan,
                w[2] if len(w)>2 else np.nan,
                np.nanmax(w)]
    except Exception:
        return [np.nan, np.nan, np.nan]



def signed_axis(cols):
    if not cols:
        return pd.Series(np.nan, index=X.index)
    B = X.loc[:, cols].copy()
    to_flip = [c for c in cols if any(k.lower() in c.lower() for k in flip_keywords)]
    if len(to_flip):
        B[to_flip] = -B[to_flip]
    return B.mean(axis=1, skipna=True)



def add_default_subgroups(df_in):
    out = df_in.copy()

    # Sex
    if "Sex_bin" in out.columns:
        out["sex_band"] = out["Sex_bin"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    elif "gender" in out.columns:
        out["sex_band"] = out["gender"].astype(str).str.upper().map({"M": "Male", "F": "Female"}).fillna("Unknown")
    else:
        out["sex_band"] = "Unknown"

    # Age (years)
    if "Age_years" in out.columns:
        age = pd.to_numeric(out["Age_years"], errors="coerce")
    elif "age" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
    else:
        age = pd.Series(np.nan, index=out.index)

    out["age_band"] = pd.cut(
        age,
        bins=[0, 1, 2, 4, 6, 8, 12, 18, 200],
        labels=["<1y", "1–2y", "2–4y", "4–6y", "6–8y", "8–12y", "12–18y", "≥18y"]
    )

    # Heart rate
    if "HR_bpm" in out.columns:
        hr = pd.to_numeric(out["HR_bpm"], errors="coerce")
        out["hr_band"] = pd.cut(hr, bins=[0, 80, 100, 120, 300], labels=["<80", "80–100", "100–120", "≥120"])
        out["tachy_100"] = np.where(hr > 100, "Tachy", "NoTachy")
    else:
        out["hr_band"] = "Unknown"
        out["tachy_100"] = "Unknown"

    # QRS prolongation (lead II if exists)
    if "QRS_ms_lead_II" in out.columns:
        qrs = pd.to_numeric(out["QRS_ms_lead_II"], errors="coerce")
        out["qrs_120"] = np.where(qrs > 120, ">120 ms", "≤120 ms")
    else:
        out["qrs_120"] = "Unknown"

    return out



def fmt_pct(x):
    return f"{100*x:.1f}%" if pd.notnull(x) else "NA"



def add_feature_bins_quantile(df_in, features, q=3, prefix="bin_"):
    """
    For each feature, create a categorical bin column using q-quantiles.
    q=3 -> Low/Mid/High
    """
    out = df_in.copy()
    labels = ["Low", "Mid", "High"] if q == 3 else [f"Q{i+1}" for i in range(q)]

    for f in features:
        if f not in out.columns:
            print(f"[skip] feature not found: {f}")
            continue
        x = pd.to_numeric(out[f], errors="coerce")

        # qcut can fail if too many identical values; duplicates='drop' helps
        try:
            out[prefix + f] = pd.qcut(x, q=q, labels=labels, duplicates="drop")
        except Exception as e:
            print(f"[warn] qcut failed for {f}: {e}")
            out[prefix + f] = np.nan

    return out



def rule_out_sensitivity(y_true, y_prob, thr=0.33):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    chd = y_true == 1
    if chd.sum() == 0:
        return np.nan

    ruled_out_chd = (y_prob <= thr) & chd
    return 1.0 - ruled_out_chd.sum() / chd.sum()



def rule_in_specificity(y_true, y_prob, thr=0.67):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    normal = y_true == 0
    if normal.sum() == 0:
        return np.nan

    ruled_in_normal = (y_prob >= thr) & normal
    return 1.0 - ruled_in_normal.sum() / normal.sum()



def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])



def curve_at_target(y_true, proba, target='sens', value=0.85):
    """Find threshold giving sensitivity (or specificity) closest to value."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    if target=='sens':
        k = np.argmin(np.abs(tpr - value))
        return thr[k], tpr[k], 1-fpr[k]
    else:
        spec = 1-fpr
        k = np.argmin(np.abs(spec - value))
        return thr[k], tpr[k], spec[k]



def assign_risk(p):
    if p < low_cut:   return "low"
    elif p >= high_cut: return "high"
    else: return "intermediate"



def nri_counts(y, old, new):
    y = np.asarray(y); old = np.asarray(old); new = np.asarray(new)
    # Map tiers to ordinal scores
    map_ord = {"low":0,"intermediate":1,"high":2}
    o = np.vectorize(map_ord.get)(old); n = np.vectorize(map_ord.get)(new)
    pos = (y==1); neg = (y==0)
    up_pos   = np.sum(n[pos] > o[pos])/np.sum(pos)
    down_pos = np.sum(n[pos] < o[pos])/np.sum(pos)
    up_neg   = np.sum(n[neg] > o[neg])/np.sum(neg)
    down_neg = np.sum(n[neg] < o[neg])/np.sum(neg)
    nri = (up_pos - down_pos) + (down_neg - up_neg)
    return nri, up_pos, down_pos, up_neg, down_neg



def find_threshold_for_target(y_true, y_prob, target="sens", value=0.95):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    sens = tpr
    if target == "sens":
        ok = sens >= value
        if not np.any(ok): return np.nan
        j = np.argmax(spec[ok])              # maximize specificity
        return float(thr[ok][j])
    elif target == "spec":
        ok = spec >= value
        if not np.any(ok): return np.nan
        j = np.argmax(sens[ok])              # maximize sensitivity
        return float(thr[ok][j])
    else:
        raise ValueError("target must be 'sens' or 'spec'")



def build_axes_from_z(df_in):
    out = df_in.copy()

    z_cols_local = [c for c in out.columns if c.endswith("_z")]
    cond_z  = [c for c in z_cols_local if c.startswith("PR_ms") or c.startswith("QRS_ms")]
    repol_z = [c for c in z_cols_local if c.startswith("QT_ms") or c.startswith("JT_ms") or c.startswith("TpTe_ms")]
    auto_z  = [c for c in z_cols_local if c.startswith("HRV_") or ("SampEn" in c) or ("DFA" in c) or ("CMSE" in c) or ("RCMSE" in c)]

    # mean over available columns row-wise
    if len(cond_z) > 0:
        out["Axis_cond"] = out[cond_z].mean(axis=1, skipna=True)
    else:
        out["Axis_cond"] = np.nan

    if len(repol_z) > 0:
        out["Axis_repol"] = out[repol_z].mean(axis=1, skipna=True)
    else:
        out["Axis_repol"] = np.nan

    if len(auto_z) > 0:
        out["Axis_auto"] = -out[auto_z].mean(axis=1, skipna=True)   # invert: lower HRV => higher risk
    else:
        out["Axis_auto"] = np.nan

    return out



def _as_float_array(x):
    return np.asarray(x, dtype=float)



def mahalanobis_axis(Z_all, labels, cols, name):
    """Compute D² for a specific subset of features (axis)."""
    Z_axis = Z_all[cols].copy().replace([np.inf, -np.inf], np.nan)
    Z_axis = Z_axis.clip(-10, 10)

    # Controls only
    ctrl_mask = (labels == 0)
    Z_ctrl = Z_axis.loc[ctrl_mask]
    if Z_ctrl.shape[1] < 2:
        return pd.Series(np.nan, index=Z_axis.index)

    imp = SimpleImputer(strategy="median")
    Z_ctrl_imp = pd.DataFrame(imp.fit_transform(Z_ctrl), index=Z_ctrl.index, columns=Z_ctrl.columns)
    Z_all_imp  = pd.DataFrame(imp.transform(Z_axis), index=Z_axis.index, columns=Z_axis.columns)

    try:
        lw = LedoitWolf().fit(Z_ctrl_imp.values)
        Sigma_inv = np.linalg.pinv(lw.covariance_)
        Z = Z_all_imp.values
        D2 = np.einsum("ij,jk,ik->i", Z, Sigma_inv, Z)
        return pd.Series(D2, index=Z_axis.index, name=f"D2_{name}")
    except Exception as e:
        print(f"⚠️ Axis {name} failed: {e}")
        return pd.Series(np.nan, index=Z_axis.index, name=f"D2_{name}")



def make_dispersion(df, prefix):
    """Return DataFrame with dispersion/IQR across leads for a per-lead metric prefix.
       e.g., prefix='QT_ms_lead_' → scans QT_ms_lead_I ... QT_ms_lead_V6 if present."""
    lead_suffixes = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
    cols = [f"{prefix}{L}" for L in lead_suffixes if f"{prefix}{L}" in df.columns]
    if len(cols) < 3:
        return pd.DataFrame(index=df.index)
    block = df[cols].apply(pd.to_numeric, errors="coerce")
    disp = block.max(axis=1) - block.min(axis=1)
    iqr  = block.quantile(0.75, axis=1) - block.quantile(0.25, axis=1)
    out = pd.DataFrame({
        f"{prefix}disp": disp,
        f"{prefix}iqr":  iqr
    }, index=df.index)
    return out



def robust_z(v, ctrl_mask):
    v = pd.to_numeric(v, errors="coerce")
    ctrl = v[ctrl_mask]
    med  = np.nanmedian(ctrl)
    mad  = np.nanmedian(np.abs(ctrl - med))
    scale = mad*1.4826 if (np.isfinite(mad) and mad>0) else np.nanstd(ctrl, ddof=1)
    return (v - med) / (scale if (scale and scale>0) else np.nan)



def signed_block(cols, flip_list=()):
    if not cols:
        return None
    B = df.loc[mask_label, cols].apply(pd.to_numeric, errors="coerce")
    B = B.clip(-10, 10)
    if flip_list:
        to_flip = [c for c in cols if c in flip_list]
        if to_flip:
            B[to_flip] = -B[to_flip]  # increasing value should correspond to higher risk
    return B



def row_mean_robust(B):
    if B is None or B.shape[1] == 0:
        return pd.Series(np.nan, index=df.index[mask_label])
    return B.mean(axis=1, skipna=True)



def fit_predict_z(y, X_train, X_all):
    # Polynomial regression in covariates to get expected mean
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    mdl  = LinearRegression().fit(Xtr, y)
    yhat = mdl.predict(poly.transform(X_all))
    resid = y - yhat if len(y)==len(X_all) else (None)  # not used
    return yhat, mdl, poly



def axis_mean(feat_list):
    cols = [c + "_z" for c in feat_list if (c + "_z") in df_z.columns]
    if not cols:
        return pd.Series(np.nan, index=df_z.index)
    return df_z[cols].mean(axis=1)



def axis_mean_min(feat_list, min_k=5):
    cols = [c + "_z" for c in feat_list if (c + "_z") in df_z.columns]
    if not cols:
        return pd.Series(np.nan, index=df_z.index)
    row_counts = df_z[cols].notna().sum(axis=1)
    row_mean = df_z[cols].mean(axis=1, skipna=True)
    row_mean[row_counts < min_k] = np.nan
    return row_mean



def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def filter_sparse(cols, df, min_non_nan=1):
    good = []
    for c in cols:
        nn = df[c].notna().sum()
        if nn >= min_non_nan:
            good.append(c)
    return good



def axis_mean_z(cols):
    zcols = [f"{c}_z" for c in cols if f"{c}_z" in df_z.columns]
    if len(zcols) == 0:
        return pd.Series(np.nan, index=df_z.index)
    return df_z[zcols].mean(axis=1)
