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
from .evaluation import *
from .modeling import *



def plot_axis_controls_vs_pathology(
    df,
    axis_col,
    label_col="_label",
    axis_name=r"$a$",
    outfile="axis.pdf"
):
    """
    Boxplot of one mechanistic axis: controls vs pathology
    """

    # clean
    d = df[[axis_col, label_col]].dropna().copy()

    controls = d[d[label_col] == 0][axis_col].values
    pathology = d[d[label_col] == 1][axis_col].values

    fig, ax = plt.subplots(figsize=(3.2, 4))

    ax.boxplot(
        [controls, pathology],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black"),
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Controls", "Pathology"], fontsize=9)
    ax.set_ylabel(axis_name, fontsize=10)

    ax.set_title(axis_name, fontsize=11)
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.tick_params(labelsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)



def plot_partial_age_curve(df_controls, feature, fit_obj, degree_age=2, degree_hr=2, include_interactions=False,
                           n_grid=200, savepath=None):
    """
    Controls scatter: feature vs Age.
    Overlay: mu(Age) holding HR=median, Sex=mode (or 0).
    """
    d = df_controls.copy()
    d = d.dropna(subset=["Age_years", "HR_bpm", "Sex_bin", feature]).copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna(subset=[feature])

    if d.empty:
        return

    age = d["Age_years"].astype(float).values
    y = d[feature].astype(float).values

    hr_med = float(np.nanmedian(d["HR_bpm"]))
    sex_mode = float(d["Sex_bin"].mode().iloc[0]) if not d["Sex_bin"].mode().empty else 0.0

    a0, a1 = _quantile_clip(age, 0.01, 0.99)
    grid = np.linspace(a0, a1, n_grid)

    df_grid = pd.DataFrame({
        "Age_years": grid,
        "HR_bpm": hr_med,
        "Sex_bin": sex_mode
    })
    mu = predict_mu(df_grid, fit_obj, degree_age, degree_hr, include_interactions)

    plt.figure()
    plt.scatter(age, y, alpha=0.25, s=12)
    plt.plot(grid, mu, linewidth=2)
    plt.xlabel("Age (years)")
    plt.ylabel(feature)
    plt.title(f"{feature} vs Age (controls) + developmental fit")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.close()



def plot_partial_hr_curve(df_controls, feature, fit_obj, degree_age=2, degree_hr=2, include_interactions=False,
                          n_grid=200, savepath=None):
    """
    Controls scatter: feature vs HR.
    Overlay: mu(HR) holding Age=median, Sex=mode.
    """
    d = df_controls.dropna(subset=["Age_years", "HR_bpm", "Sex_bin", feature]).copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna(subset=[feature])
    if d.empty:
        return

    hr = d["HR_bpm"].astype(float).values
    y = d[feature].astype(float).values

    age_med = float(np.nanmedian(d["Age_years"]))
    sex_mode = float(d["Sex_bin"].mode().iloc[0]) if not d["Sex_bin"].mode().empty else 0.0

    h0, h1 = _quantile_clip(hr, 0.01, 0.99)
    grid = np.linspace(h0, h1, n_grid)

    df_grid = pd.DataFrame({
        "Age_years": age_med,
        "HR_bpm": grid,
        "Sex_bin": sex_mode
    })
    mu = predict_mu(df_grid, fit_obj, degree_age, degree_hr, include_interactions)

    plt.figure()
    plt.scatter(hr, y, alpha=0.25, s=12)
    plt.plot(grid, mu, linewidth=2)
    plt.xlabel("HR (bpm)")
    plt.ylabel(feature)
    plt.title(f"{feature} vs HR (controls) + developmental fit")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.close()



def plot_sex_stratified_age_curves(df_controls, feature, fit_obj, degree_age=2, degree_hr=2, include_interactions=False,
                                  n_grid=200, savepath=None):
    """
    Show two model curves over Age:
      Sex=0 vs Sex=1, holding HR=median.
    """
    d = df_controls.dropna(subset=["Age_years", "HR_bpm", "Sex_bin", feature]).copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna(subset=[feature])
    if d.empty:
        return

    age = d["Age_years"].astype(float).values
    y = d[feature].astype(float).values
    hr_med = float(np.nanmedian(d["HR_bpm"]))

    a0, a1 = _quantile_clip(age, 0.01, 0.99)
    grid = np.linspace(a0, a1, n_grid)

    df_grid0 = pd.DataFrame({"Age_years": grid, "HR_bpm": hr_med, "Sex_bin": 0.0})
    df_grid1 = pd.DataFrame({"Age_years": grid, "HR_bpm": hr_med, "Sex_bin": 1.0})

    mu0 = predict_mu(df_grid0, fit_obj, degree_age, degree_hr, include_interactions)
    mu1 = predict_mu(df_grid1, fit_obj, degree_age, degree_hr, include_interactions)

    plt.figure()
    plt.scatter(age, y, alpha=0.15, s=10)
    plt.plot(grid, mu0, linewidth=2, label="Sex=0")
    plt.plot(grid, mu1, linewidth=2, label="Sex=1")
    plt.xlabel("Age (years)")
    plt.ylabel(feature)
    plt.title(f"{feature}: sex-stratified developmental curves (HR=median)")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.close()



def plot_residual_histogram(fit_obj, feature, savepath=None):
    """Residual distribution on controls (should be ~centered; not necessarily perfectly normal)."""
    resid = fit_obj["resid_fit"]
    plt.figure()
    plt.hist(resid, bins=40)
    plt.xlabel("Residual (f - mu)")
    plt.ylabel("Count")
    plt.title(f"{feature}: residuals on controls (sigma={fit_obj['sigma_res']:.2f})")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.close()



def save_plots_to_pdf_grid(
    plot_func,
    feature_list,
    pdf_path,
    nrows=3,
    ncols=5,
    figsize=(15, 9),
    **plot_kwargs
):
    """
    plot_func(ax, feature, **plot_kwargs) should draw ONE subplot
    """
    plots_per_page = nrows * ncols
    n_pages = math.ceil(len(feature_list) / plots_per_page)

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = axes.flatten()

            start = page * plots_per_page
            end = min(start + plots_per_page, len(feature_list))
            subset = feature_list[start:end]

            for ax, feature in zip(axes, subset):
                plot_func(ax=ax, feature=feature, **plot_kwargs)

            # Turn off unused axes
            for ax in axes[len(subset):]:
                ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved PDF: {pdf_path}")



def plot_feature_vs_age_ax(
    ax,
    df_controls,
    feature,
    fit_obj,
    degree_age=2,
    degree_hr=2,
):
    d = df_controls.dropna(subset=["Age_years", "HR_bpm", "Sex_bin", feature]).copy()
    if d.empty:
        ax.set_title(feature)
        ax.text(0.5, 0.5, "No data", ha="center")
        return

    age = d["Age_years"].astype(float).values
    y = pd.to_numeric(d[feature], errors="coerce").values

    hr_med = float(np.nanmedian(d["HR_bpm"]))
    sex_mode = float(d["Sex_bin"].mode().iloc[0])

    grid = np.linspace(np.nanpercentile(age, 1), np.nanpercentile(age, 99), 200)
    df_grid = pd.DataFrame({
        "Age_years": grid,
        "HR_bpm": hr_med,
        "Sex_bin": sex_mode
    })

    mu = predict_mu(df_grid, fit_obj, degree_age, degree_hr)

    ax.scatter(age, y, s=6, alpha=0.2)
    ax.plot(grid, mu, linewidth=2)

    ax.set_title(feature, fontsize=9)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Value")



def devnorm_age_plot(ax, feature, df_controls):
    fit_obj = fit_developmental_model_one_feature(
        df_controls,
        feature,
        degree_age=2,
        degree_hr=2
    )
    plot_feature_vs_age_ax(
        ax=ax,
        df_controls=df_controls,
        feature=feature,
        fit_obj=fit_obj
    )



def plot_before_after_age(df_all, df_ref, feature, degree=2,
                          age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                          label_col=None,  # optional, e.g. "_label" where 0=control
                          savepath=None,
                          max_points=15000,
                          figsize=(10, 4.2)):
    """
    Creates a 1x2 panel:
      left: raw feature vs age + fitted developmental curve at median HR and Sex=0/1 (optional)
      right: z-score vs age (after normalization)
    """

    # add z-score
    d = add_zscore_column(df_all, df_ref, feature, degree=degree,
                          age_col=age_col, hr_col=hr_col, sex_col=sex_col)

    # Subsample for visualization
    d_plot = d[[age_col, feature, f"{feature}_z", hr_col, sex_col] + ([label_col] if label_col else [])].copy()
    d_plot[age_col] = _to_numeric_series(d_plot[age_col])
    d_plot = d_plot.dropna(subset=[age_col])

    if len(d_plot) > max_points:
        d_plot = d_plot.sample(n=max_points, random_state=42)

    # Fit model (for curve overlay)
    model, sigma, design_cols = fit_dev_model(df_ref, feature, degree=degree,
                                              age_col=age_col, hr_col=hr_col, sex_col=sex_col)

    # Build curve grid
    age_grid = np.linspace(d_plot[age_col].min(), d_plot[age_col].max(), 200)
    hr_med = np.nanmedian(_to_numeric_series(df_ref[hr_col]))
    if np.isnan(hr_med):
        hr_med = np.nanmedian(_to_numeric_series(df_all[hr_col]))
    # choose Sex=0 (female) for curve; you can overlay sex=1 too if you want
    sex0 = 0.0

    Xg = pd.DataFrame({
        "Age": age_grid,
        "HR": np.full_like(age_grid, hr_med, dtype=float),
        "Sex": np.full_like(age_grid, sex0, dtype=float),
        "Age2": age_grid**2,
        "HR2": np.full_like(age_grid, hr_med, dtype=float)**2,
        "AgeHR": age_grid*hr_med,
    })
    if "Age3" in design_cols:
        Xg["Age3"] = age_grid**3
        Xg["HR3"] = np.full_like(age_grid, hr_med, dtype=float)**3
        Xg["Age2HR"] = (age_grid**2)*hr_med
        Xg["AgeHR2"] = age_grid*(hr_med**2)

    Xg = Xg[design_cols]
    mu_grid = model.predict(Xg)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: raw
    axes[0].scatter(d_plot[age_col].values, _to_numeric_series(d_plot[feature]).values, s=10, alpha=0.25)
    axes[0].plot(age_grid, mu_grid, linewidth=2)  # no color forced
    axes[0].set_title(f"{feature} vs Age (raw)")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel(feature)

    # Right: z
    axes[1].scatter(d_plot[age_col].values, _to_numeric_series(d_plot[f"{feature}_z"]).values, s=10, alpha=0.25)
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title(f"{feature} z-score vs Age (normalized)")
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel(f"{feature}_z")

    fig.tight_layout()

    if savepath:
        _safe_makedirs(savepath)
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return d  # returns dataframe with mu and z columns for this feature



def make_before_after_pdf_grid(df_all, df_ref, features,
                               degree=2,
                               age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                               label_col=None,
                               out_pdf="figures/S2_before_after_grid.pdf",
                               nrows=3, ncols=5,
                               max_pages=None,
                               max_points=8000):
    """
    Produces multi-page PDF.
    Each feature occupies ONE panel (raw vs age + fit) OR (z vs age).
    Since you asked specifically "Feature vs age (raw) and Z-score vs age", we will create TWO pages per block:
      - Page A: Raw vs age grid
      - Page B: Z vs age grid
    (This is usually what journals prefer: two separate supplement figures.)

    If you prefer a 1x2 for each feature, tell me and I’ll adapt.
    """

    from matplotlib.backends.backend_pdf import PdfPages

    # Fit models once per feature and cache (faster)
    models = {}
    for f in features:
        try:
            m, sig, cols = fit_dev_model(df_ref, f, degree=degree,
                                         age_col=age_col, hr_col=hr_col, sex_col=sex_col)
            models[f] = (m, sig, cols)
        except Exception as e:
            print(f"[skip] {f}: {e}")

    feats = list(models.keys())
    if max_pages is not None:
        feats = feats[: max_pages * nrows * ncols]

    with PdfPages(out_pdf) as pdf:

        # ---- RAW pages ----
        chunk = nrows * ncols
        for i in range(0, len(feats), chunk):
            page_feats = feats[i:i+chunk]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.7))
            axes = np.array(axes).reshape(-1)

            for ax, f in zip(axes, page_feats):
                m, sig, cols = models[f]

                d = df_all[[age_col, hr_col, sex_col, f]].copy()
                d[age_col] = _to_numeric_series(d[age_col])
                d[hr_col] = _to_numeric_series(d[hr_col])
                d[sex_col] = _to_numeric_series(d[sex_col])
                d[f] = _to_numeric_series(d[f])
                d = d.dropna(subset=[age_col, f])

                if len(d) > max_points:
                    d = d.sample(n=max_points, random_state=42)

                ax.scatter(d[age_col].values, d[f].values, s=6, alpha=0.25)

                # overlay curve at median HR, Sex=0
                age_grid = np.linspace(d[age_col].min(), d[age_col].max(), 200)
                hr_med = np.nanmedian(_to_numeric_series(df_ref[hr_col]))
                if np.isnan(hr_med):
                    hr_med = np.nanmedian(d[hr_col])
                sex0 = 0.0

                Xg = pd.DataFrame({
                    "Age": age_grid,
                    "HR": np.full_like(age_grid, hr_med, dtype=float),
                    "Sex": np.full_like(age_grid, sex0, dtype=float),
                    "Age2": age_grid**2,
                    "HR2": np.full_like(age_grid, hr_med, dtype=float)**2,
                    "AgeHR": age_grid*hr_med,
                })
                if "Age3" in cols:
                    Xg["Age3"] = age_grid**3
                    Xg["HR3"] = np.full_like(age_grid, hr_med, dtype=float)**3
                    Xg["Age2HR"] = (age_grid**2)*hr_med
                    Xg["AgeHR2"] = age_grid*(hr_med**2)

                Xg = Xg[cols]
                mu = m.predict(Xg)
                ax.plot(age_grid, mu, linewidth=1.8)

                ax.set_title(f, fontsize=9)
                ax.set_xlabel("Age", fontsize=8)
                ax.set_ylabel("Raw", fontsize=8)
                ax.tick_params(labelsize=7)

            # turn off unused axes
            for ax in axes[len(page_feats):]:
                ax.axis("off")

            fig.suptitle("Developmental curves (raw feature vs age)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ---- Z-score pages ----
        for i in range(0, len(feats), chunk):
            page_feats = feats[i:i+chunk]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.7))
            axes = np.array(axes).reshape(-1)

            for ax, f in zip(axes, page_feats):
                m, sig, cols = models[f]

                d = df_all[[age_col, hr_col, sex_col, f]].copy()
                d[age_col] = _to_numeric_series(d[age_col])
                d[hr_col] = _to_numeric_series(d[hr_col])
                d[sex_col] = _to_numeric_series(d[sex_col])
                d[f] = _to_numeric_series(d[f])
                d = d.dropna(subset=[age_col, hr_col, sex_col, f])

                if len(d) > max_points:
                    d = d.sample(n=max_points, random_state=42)

                # predict mu and compute z
                A = d[age_col].values
                H = d[hr_col].values
                S = d[sex_col].values

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
                })[cols]

                mu = m.predict(X)
                z = (d[f].values - mu) / sig if (sig is not None and not np.isnan(sig) and sig > 0) else np.full(len(d), np.nan)

                ax.scatter(d[age_col].values, z, s=6, alpha=0.25)
                ax.axhline(0.0, linewidth=1,color='red')

                ax.set_title(f, fontsize=9)
                ax.set_xlabel("Age", fontsize=8)
                ax.set_ylabel("Z", fontsize=8)
                ax.tick_params(labelsize=7)

            for ax in axes[len(page_feats):]:
                ax.axis("off")

            fig.suptitle("After developmental normalization (z-score vs age)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {out_pdf}")



def make_raw_and_z_grid_pdf(df, df_ref, features, out_pdf,
                           nrows=3, ncols=5, degree=2, max_points=8000):

    chunk = nrows * ncols
    valid_feats = []

    # First pass: keep only features that can be fitted on controls
    for f in features:
        _, ok, msg = add_zscore_column_safe(df, df_ref, f, degree=degree)
        if ok:
            valid_feats.append(f)
        else:
            print("[skip]", msg)

    print("Valid features kept:", len(valid_feats), "/", len(features))

    with PdfPages(out_pdf) as pdf:

        # -------- RAW pages --------
        for i in range(0, len(valid_feats), chunk):
            page_feats = valid_feats[i:i+chunk]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.7))
            axes = np.array(axes).reshape(-1)

            for ax, f in zip(axes, page_feats):
                tmp = df[["Age_years", f]].copy()
                tmp["Age_years"] = pd.to_numeric(tmp["Age_years"], errors="coerce")
                tmp[f] = pd.to_numeric(tmp[f], errors="coerce")
                tmp = tmp.dropna()

                if len(tmp) > max_points:
                    tmp = tmp.sample(n=max_points, random_state=42)

                ax.scatter(tmp["Age_years"], tmp[f], s=6, alpha=0.25)
                ax.set_title(f, fontsize=9)
                ax.set_xlabel("Age", fontsize=8)
                ax.set_ylabel("Raw", fontsize=8)
                ax.tick_params(labelsize=7)

            for ax in axes[len(page_feats):]:
                ax.axis("off")

            fig.suptitle("Raw feature vs Age", fontsize=12)
            fig.tight_layout(rect=[0,0,1,0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # -------- Z pages --------
        for i in range(0, len(valid_feats), chunk):
            page_feats = valid_feats[i:i+chunk]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.7))
            axes = np.array(axes).reshape(-1)

            for ax, f in zip(axes, page_feats):
                dfz, ok, msg = add_zscore_column_safe(df, df_ref, f, degree=degree)
                print(msg)

                tmp = dfz[["Age_years", f"{f}_z"]].copy()
                tmp["Age_years"] = pd.to_numeric(tmp["Age_years"], errors="coerce")
                tmp[f"{f}_z"] = pd.to_numeric(tmp[f"{f}_z"], errors="coerce")
                tmp = tmp.dropna()

                if len(tmp) > max_points:
                    tmp = tmp.sample(n=max_points, random_state=42)

                ax.scatter(tmp["Age_years"], tmp[f"{f}_z"], s=6, alpha=0.25)
                ax.axhline(0, linewidth=1)
                ax.set_title(f, fontsize=9)
                ax.set_xlabel("Age", fontsize=8)
                ax.set_ylabel("Z", fontsize=8)
                ax.tick_params(labelsize=7)

            for ax in axes[len(page_feats):]:
                ax.axis("off")

            fig.suptitle("After developmental normalization (z-score vs Age)", fontsize=12)
            fig.tight_layout(rect=[0,0,1,0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved:", out_pdf)



def plot_feature_vs_age_controls(
    df_ref, feature,
    degree=2,
    age_col="Age_years",
    hr_col="HR_bpm",
    sex_col="Sex_bin",
    savepath="figures/S1_feature_vs_age.pdf",
    max_points=8000
):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    d = df_ref[[age_col, hr_col, sex_col, feature]].copy()
    d = d.apply(pd.to_numeric, errors="coerce").dropna()

    if len(d) > max_points:
        d = d.sample(max_points, random_state=42)

    A, H, S = d[age_col].values, d[hr_col].values, d[sex_col].values
    y = d[feature].values

    X = pd.DataFrame({
        "Age": A,
        "HR": H,
        "Sex": S,
        "Age2": A**2,
    })

    model = LinearRegression().fit(X, y)

    age_grid = np.linspace(A.min(), A.max(), 200)
    hr_med = np.median(H)
    sex_med = np.median(S)

    Xg = pd.DataFrame({
        "Age": age_grid,
        "HR": hr_med,
        "Sex": sex_med,
        "Age2": age_grid**2,
    })

    yhat = model.predict(Xg)

    plt.figure(figsize=(6,4))
    plt.scatter(A, y, s=8, alpha=0.25)
    plt.plot(age_grid, yhat, linewidth=2)
    plt.xlabel("Age (years)")
    plt.ylabel(feature)
    plt.title(f"{feature} vs Age (controls)")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()



def plot_feature_vs_hr_controls(
    df_ref, feature,
    degree=2,
    age_col="Age_years",
    hr_col="HR_bpm",
    sex_col="Sex_bin",
    savepath="figures/S2_feature_vs_hr.pdf",
    max_points=8000
):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    d = df_ref[[age_col, hr_col, sex_col, feature]].copy()
    d = d.apply(pd.to_numeric, errors="coerce").dropna()

    if len(d) > max_points:
        d = d.sample(max_points, random_state=42)

    A, H, S = d[age_col].values, d[hr_col].values, d[sex_col].values
    y = d[feature].values

    X = pd.DataFrame({
        "HR": H,
        "Age": A,
        "Sex": S,
        "HR2": H**2,
    })

    model = LinearRegression().fit(X, y)

    hr_grid = np.linspace(H.min(), H.max(), 200)
    age_med = np.median(A)
    sex_med = np.median(S)

    Xg = pd.DataFrame({
        "HR": hr_grid,
        "Age": age_med,
        "Sex": sex_med,
        "HR2": hr_grid**2,
    })

    yhat = model.predict(Xg)

    plt.figure(figsize=(6,4))
    plt.scatter(H, y, s=8, alpha=0.25)
    plt.plot(hr_grid, yhat, linewidth=2)
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel(feature)
    plt.title(f"{feature} vs HR (controls)")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()



def make_S3_raw_vs_age_pdf(df, df_ref, features,
                           out_pdf="figures/S3_raw_vs_age_intervals.pdf",
                           degree=2, nrows=3, ncols=5,
                           age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                           max_points=8000):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    # cache models
    models = {}
    for f in features:
        m, sigma, cols = fit_dev_model_safe(df_ref, f, degree=degree,
                                            age_col=age_col, hr_col=hr_col, sex_col=sex_col,
                                            min_controls=200)
        if m is not None:
            models[f] = (m, cols)

    feats = list(models.keys())
    chunk = nrows * ncols

    hr_med = np.nanmedian(_to_num(df_ref[hr_col]))
    if np.isnan(hr_med):
        hr_med = np.nanmedian(_to_num(df[hr_col]))
    sex0 = 0.0

    with PdfPages(out_pdf) as pdf:
        for i in range(0, len(feats), chunk):
            page_feats = feats[i:i+chunk]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.7))
            axes = np.array(axes).reshape(-1)

            for ax, f in zip(axes, page_feats):
                m, cols = models[f]
                tmp = df[[age_col, hr_col, sex_col, f]].copy()
                tmp[age_col] = _to_num(tmp[age_col])
                tmp[f] = _to_num(tmp[f])
                tmp = tmp.dropna(subset=[age_col, f])

                if len(tmp) > max_points:
                    tmp = tmp.sample(n=max_points, random_state=42)

                ax.scatter(tmp[age_col].values, tmp[f].values, s=6, alpha=0.25)

                # curve overlay
                age_grid = np.linspace(tmp[age_col].min(), tmp[age_col].max(), 200)
                Xg = pd.DataFrame({
                    "Age": age_grid,
                    "HR": np.full_like(age_grid, hr_med, dtype=float),
                    "Sex": np.full_like(age_grid, sex0, dtype=float),
                    "Age2": age_grid**2,
                    "HR2": np.full_like(age_grid, hr_med, dtype=float)**2,
                    "AgeHR": age_grid*hr_med,
                    "Age3": age_grid**3,
                    "HR3": np.full_like(age_grid, hr_med, dtype=float)**3,
                    "Age2HR": (age_grid**2)*hr_med,
                    "AgeHR2": age_grid*(hr_med**2),
                })[cols]

                mu = m.predict(Xg)
                ax.plot(age_grid, mu, linewidth=1.8)

                ax.set_title(f, fontsize=9)
                ax.set_xlabel("Age", fontsize=8)
                ax.set_ylabel("Raw", fontsize=8)
                ax.tick_params(labelsize=7)

            for ax in axes[len(page_feats):]:
                ax.axis("off")

            fig.suptitle("Supplementary Fig. S3 — Developmental curves (raw feature vs age)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved:", out_pdf)



def make_S4_z_vs_age_pdf(df, df_ref, features,
                         out_pdf="figures/S4_z_vs_age_intervals.pdf",
                         degree=2, nrows=3, ncols=5,
                         age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                         max_points=8000):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    # compute z columns (skip those that cannot fit)
    feats_ok = []
    for f in features:
        if f"{f}_z" not in df.columns:
            df = add_zscore_column_safe(df, df_ref, f, degree=degree,
                                        age_col=age_col, hr_col=hr_col, sex_col=sex_col,
                                        suffix="_z", min_controls=200)
        if f"{f}_z" in df.columns and df[f"{f}_z"].notna().sum() > 0:
            feats_ok.append(f)

    chunk = nrows * ncols

    with PdfPages(out_pdf) as pdf:
        for i in range(0, len(feats_ok), chunk):
            page_feats = feats_ok[i:i+chunk]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.7))
            axes = np.array(axes).reshape(-1)

            for ax, f in zip(axes, page_feats):
                tmp = df[[age_col, f"{f}_z"]].copy()
                tmp[age_col] = _to_num(tmp[age_col])
                tmp[f"{f}_z"] = _to_num(tmp[f"{f}_z"])
                tmp = tmp.dropna()

                if len(tmp) > max_points:
                    tmp = tmp.sample(n=max_points, random_state=42)

                ax.scatter(tmp[age_col].values, tmp[f"{f}_z"].values, s=6, alpha=0.25)
                ax.axhline(0.0, linewidth=1)

                ax.set_title(f, fontsize=9)
                ax.set_xlabel("Age", fontsize=8)
                ax.set_ylabel("Z", fontsize=8)
                ax.tick_params(labelsize=7)

            for ax in axes[len(page_feats):]:
                ax.axis("off")

            fig.suptitle("Supplementary Fig. S4 — Z-scores vs age (after developmental normalization)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved:", out_pdf)



def plot_feature_vs_hr(df, df_ref, feature, savepath=None):
    model = fit_developmental_model(df_ref, feature)

    hr_grid = np.linspace(df["HR"].min(), df["HR"].max(), 100)
    age_mean = df_ref["age"].mean()
    sex_mode = df_ref["sex"].mode()[0]

    X_pred = pd.DataFrame({
        "age": age_mean,
        "HR": hr_grid,
        "sex": sex_mode,
        "age2": age_mean ** 2
    })

    y_pred = model.predict(X_pred)

    plt.figure(figsize=(7,5))
    plt.scatter(df_ref["HR"], df_ref[feature], alpha=0.4, label="Reference data")
    plt.plot(hr_grid, y_pred, color="darkgreen", linewidth=2, label="HR-adjusted fit")
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel(feature)
    plt.title(f"{feature} vs Heart Rate")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()



def plot_residuals(df_ref, feature, savepath=None):
    model = fit_developmental_model(df_ref, feature)

    X = df_ref[["age", "HR", "sex"]].copy()
    X["age2"] = X["age"] ** 2
    residuals = df_ref[feature] - model.predict(X)

    plt.figure(figsize=(7,5))
    plt.hist(residuals, bins=30, density=True, alpha=0.7)
    plt.axvline(residuals.mean(), color="red", linestyle="--")
    plt.xlabel("Residual value")
    plt.ylabel("Density")
    plt.title(f"Residual distribution: {feature}")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()



def plot_before_after_normalization(df, df_ref, feature, savepath=None):
    model = fit_developmental_model(df_ref, feature)

    X = df[["age", "HR", "sex"]].copy()
    X["age2"] = X["age"] ** 2

    residuals = df[feature] - model.predict(X)
    z = (residuals - residuals.mean()) / residuals.std()

    fig, axs = plt.subplots(1, 2, figsize=(10,4))

    axs[0].hist(df[feature], bins=30, alpha=0.7)
    axs[0].set_title("Raw feature")

    axs[1].hist(z, bins=30, alpha=0.7)
    axs[1].set_title("Age-adjusted z-score")

    for ax in axs:
        ax.set_ylabel("Count")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()



def plot_subgroup_forest(tbl, x_sens=0.90, x_spec=0.90, outpath=None):
    """
    x_sens, x_spec: reference vertical lines (e.g., target 0.90)
    marker size ∝ n
    """
    if tbl is None or len(tbl)==0:
        raise ValueError("Empty subgroup table. Lower min_n or check subgroup columns.")

    # Create display labels and y positions
    labels = (tbl["Group"] + " : " + tbl["Level"]).tolist()
    y = np.arange(len(tbl))[::-1]  # top to bottom

    # marker size proportional to n
    n = tbl["n"].values.astype(float)
    ms = 30 * (n / np.nanmax(n)) + 10

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.35*len(tbl) + 2), sharey=True)

    # Left: rule-out sensitivity
    ax = axes[0]
    x = tbl["ruleout_sens"].values
    xerr = np.vstack([x - tbl["ruleout_sens_lo"].values, tbl["ruleout_sens_hi"].values - x])
    ax.errorbar(x, y, xerr=xerr, fmt="s", markersize=0, elinewidth=1.2, capsize=2)
    ax.scatter(x, y, s=ms, marker="s")
    ax.axvline(x_sens, linestyle="--", linewidth=1)
    ax.set_xlabel("Rule-Out Sensitivity")
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="x", alpha=0.25)

    # Right: rule-in specificity
    ax = axes[1]
    x = tbl["rulein_spec"].values
    xerr = np.vstack([x - tbl["rulein_spec_lo"].values, tbl["rulein_spec_hi"].values - x])
    ax.errorbar(x, y, xerr=xerr, fmt="s", markersize=0, elinewidth=1.2, capsize=2)
    ax.scatter(x, y, s=ms, marker="s")
    ax.axvline(x_spec, linestyle="--", linewidth=1)
    ax.set_xlabel("Rule-In Specificity")
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="x", alpha=0.25)

    # Y tick labels in the middle area
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)

    # Add n and n_pos as text (like the paper)
    # (put it to the far right of the left panel)
    for i, yi in enumerate(y):
        ax0 = axes[0]
        txt = f"n={int(tbl.loc[i,'n'])}, pos={int(tbl.loc[i,'n_pos'])}"
        ax0.text(0.505, yi, txt, va="center", fontsize=9)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    return fig



def plot_subgroup_forest_auc(tbl, outpath="figures/Fig_subgroups_AUROC.pdf", x_center=0.5):
    if tbl is None or len(tbl) == 0:
        print("No subgroups passed min_n or no valid AUROC computed.")
        return

    labels = []
    for _, r in tbl.iterrows():
        labels.append(f'{r["Group"]}: {r["Level"]} (n={int(r["N"])}, pos={r["Pos_%"]:.1f}%)')

    y = np.arange(len(tbl))[::-1]
    fig_h = max(4, 0.30 * len(tbl))
    plt.figure(figsize=(10, fig_h))

    x = tbl["AUROC"].values
    xerr_lo = x - tbl["AUROC_CI_lo"].values
    xerr_hi = tbl["AUROC_CI_hi"].values - x
    plt.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)

    plt.yticks(y, labels)
    plt.axvline(x_center, linestyle="--")
    plt.xlabel("AUROC (95% CI)")
    plt.title("Subgroup robustness (AUROC with 95% CI)")
    plt.tight_layout()

    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()
    print("Saved:", outpath)



def plot_ruleout_rulein_two_panel(tbl, thr_line=0.90, outpath="figures/Fig_subgroups_ruleout_rulein.pdf"):
    if tbl is None or len(tbl) == 0:
        print("No rows to plot.")
        return

    # y labels
    labels = [
        f'{r["Group"]}: {r["Level"]} (n={int(r["N"])}, pos={r["Pos_%"]:.1f}%)'
        for _, r in tbl.iterrows()
    ]
    y = np.arange(len(tbl))[::-1]

    fig_h = max(5, 0.30 * len(tbl))
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_h), sharey=True)

    # Left: Rule-out sensitivity
    x = tbl["RuleOut_Sens"].values
    xerr = [x - tbl["RuleOut_Sens_lo"].values, tbl["RuleOut_Sens_hi"].values - x]
    axes[0].errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    axes[0].axvline(thr_line, color="red", ls="--")
    axes[0].set_xlabel("Rule-out sensitivity (95% CI)")
    axes[0].set_title("Rule-out (p ≤ 0.33)")

    # Right: Rule-in specificity
    x2 = tbl["RuleIn_Spec"].values
    xerr2 = [x2 - tbl["RuleIn_Spec_lo"].values, tbl["RuleIn_Spec_hi"].values - x2]
    axes[1].errorbar(x2, y, xerr=xerr2, fmt="o", capsize=3)
    axes[1].axvline(thr_line, color="red", ls="--")
    axes[1].set_xlabel("Rule-in specificity (95% CI)")
    axes[1].set_title("Rule-in (p ≥ 0.67)")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)

    plt.tight_layout()

    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()
    print("Saved:", outpath)



def plot_calibration(y_true, y_prob, n_bins=10, strategy="quantile",
                     title="Calibration curve", outpath=None, show_hist=True):
    """
    Reliability diagram + optional histogram of predicted probabilities.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)

    ece = expected_calibration_error(y, p, n_bins=n_bins, strategy=strategy)
    brier = brier_score_loss(y, p)

    bins = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    x = [b["p_mean"] for b in bins]
    y_rate = [b["y_rate"] for b in bins]
    n = [b["n"] for b in bins]

    if show_hist:
        fig, axes = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios":[3,1]})
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
    ax.plot(x, y_rate, marker="o", linewidth=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"{title}\nECE={ece:.3f} | Brier={brier:.3f}")

    if show_hist:
        axh = axes[1]
        axh.hist(p[np.isfinite(p)], bins=20)
        axh.set_xlim(0,1)
        axh.set_xlabel("Predicted probability")
        axh.set_ylabel("Count")
        plt.tight_layout()
    else:
        plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight")
        print("Saved:", outpath)

    plt.show()
    return {"ECE": ece, "Brier": brier, "bins": bins}



def plot_decision_curve(y_true, y_prob, thr_min=0.01, thr_max=0.99, n=99,
                        title="Decision curve analysis", outpath=None):
    thresholds = np.linspace(thr_min, thr_max, n)
    res = net_benefit(y_true, y_prob, thresholds)

    plt.figure(figsize=(7,5))
    plt.plot(res["thresholds"], res["nb_model"], label="Model")
    plt.plot(res["thresholds"], res["nb_all"], linestyle="--", label="Treat-all")
    plt.plot(res["thresholds"], res["nb_none"], linestyle="--", label="Treat-none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight")
        print("Saved:", outpath)

    plt.show()
    return res



def plot_risk_distributions(y, p, t1, t2, bins=40):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    plt.figure(figsize=(8,4))
    plt.hist(p[y==0], bins=bins, alpha=0.6, density=True, label="Non-CHD")
    plt.hist(p[y==1], bins=bins, alpha=0.6, density=True, label="CHD")

    plt.axvline(t1, linestyle="--", linewidth=2)
    plt.axvline(t2, linestyle="--", linewidth=2)

    plt.xlabel("Predicted CHD probability (OOF)")
    plt.ylabel("Density")
    plt.title("Predicted risk distribution with clinical operating thresholds")
    plt.legend()
    plt.tight_layout()
    plt.show()



def pick_representative_ecgs(
    df_eval,
    id_col="ECG_ID",
    y_col="_label",
    p_col="proba_xgb",
    t_ruleout=None,      # e.g., t1 from your threshold code
    t_rulein=None,       # e.g., t2 from your threshold code
    mid_target=0.50,     # pick something around 0.5
    low_target=0.08,     # nice low-risk example
    high_target=0.90,    # nice high-risk example
    seed=42
):
    df = df_eval[[id_col, y_col, p_col]].dropna().copy()
    df[y_col] = df[y_col].astype(int)
    df[p_col] = df[p_col].astype(float)

    if t_ruleout is None or t_rulein is None:
        raise ValueError("Please pass t_ruleout (t1) and t_rulein (t2) from your computed thresholds.")

    # Helper: choose row closest to a target probability within a filter
    def choose_closest(subdf, target):
        if len(subdf) == 0:
            return None
        idx = (subdf[p_col] - target).abs().idxmin()
        return subdf.loc[idx]

    picks = []

    # 1) True Negative – Low Risk (y=0 and p < t1). Choose closest to low_target
    tn_pool = df[(df[y_col] == 0) & (df[p_col] < t_ruleout)]
    r = choose_closest(tn_pool, low_target)
    if r is not None:
        picks.append(("True Negative – Low Risk", r))

    # 2) True Positive – High Risk (y=1 and p >= t2). Choose closest to high_target
    tp_pool = df[(df[y_col] == 1) & (df[p_col] >= t_rulein)]
    r = choose_closest(tp_pool, high_target)
    if r is not None:
        picks.append(("True Positive – High Risk", r))

    # 3) Borderline / Mid Risk (any y) in (t1, t2). Choose closest to mid_target
    mid_pool = df[(df[p_col] >= t_ruleout) & (df[p_col] < t_rulein)]
    r = choose_closest(mid_pool, mid_target)
    if r is not None:
        # Also label whether it’s CHD or not (helpful for clinician context)
        picks.append(("Borderline / Mid Risk", r))

    # 4) False Negative (y=1 but p < t1). Choose the *highest* risk among FN (closest to t1 from below)
    fn_pool = df[(df[y_col] == 1) & (df[p_col] < t_ruleout)].copy()
    if len(fn_pool):
        r = fn_pool.sort_values(p_col, ascending=False).iloc[0]
        picks.append(("False Negative (CHD but Low Risk)", r))

    # 5) False Positive (optional) (y=0 but p >= t2). Choose the *lowest* risk among FP (closest to t2 from above)
    fp_pool = df[(df[y_col] == 0) & (df[p_col] >= t_rulein)].copy()
    if len(fp_pool):
        r = fp_pool.sort_values(p_col, ascending=True).iloc[0]
        picks.append(("False Positive (Non-CHD but High Risk)", r))

    # Build output table + de-duplicate if any ID repeats
    out_rows = []
    seen = set()
    for case_type, row in picks:
        ecg_id = row[id_col]
        if ecg_id in seen:
            continue
        seen.add(ecg_id)
        out_rows.append({
            "case_type": case_type,
            "ecg_id": ecg_id,
            "model_score": float(row[p_col]),
            "ground_truth_CHD": int(row[y_col]),
            "risk_zone": ("Low (rule-out)" if row[p_col] < t_ruleout else ("High (rule-in)" if row[p_col] >= t_rulein else "Intermediate"))
        })

    out = pd.DataFrame(out_rows)

    # nice ordering
    order = [
        "True Negative – Low Risk",
        "True Positive – High Risk",
        "Borderline / Mid Risk",
        "False Negative (CHD but Low Risk)",
        "False Positive (Non-CHD but High Risk)"
    ]
    out["case_type"] = pd.Categorical(out["case_type"], categories=order, ordered=True)
    out = out.sort_values("case_type").reset_index(drop=True)

    return out



def compute_intervals_for_lead(signal, fs):
    """Return median intervals (ms) for one lead. Robust + no plotting."""
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="biosppy")

    _, rpeaks = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")

    # Delineation (no show)
    _, waves = nk.ecg_delineate(
        clean, rpeaks, sampling_rate=fs,
        method="peak", show=False, show_type="peaks"
    )
    df = pd.DataFrame(waves)

    def interval(on, off):
        if on not in df.columns or off not in df.columns:
            return pd.Series(dtype=float)
        mask = df[on].notna() & df[off].notna()
        if mask.sum() == 0:
            return pd.Series(dtype=float)
        idx_on  = df.loc[mask, on].astype(int)
        idx_off = df.loc[mask, off].astype(int)
        out = (idx_off - idx_on) / fs * 1000.0
        return out[out > 0]

    # Prefer offsets if available
    QT_series = interval("ECG_Q_Peaks", "ECG_T_Offsets")
    if QT_series.empty:
        QT_series = interval("ECG_Q_Peaks", "ECG_T_Peaks")  # fallback

    QRS_series = interval("ECG_Q_Peaks", "ECG_S_Peaks")
    PR_series  = interval("ECG_P_Peaks", "ECG_Q_Peaks")
    ST_series  = interval("ECG_S_Peaks", "ECG_T_Offsets") if "ECG_T_Offsets" in df.columns else interval("ECG_S_Peaks", "ECG_T_Peaks")

    return {
        "PR_ms":  PR_series.median() if not PR_series.empty else np.nan,
        "QRS_ms": QRS_series.median() if not QRS_series.empty else np.nan,
        "QT_ms":  QT_series.median() if not QT_series.empty else np.nan,
        "JT_ms":  (QT_series - QRS_series).median() if (not QT_series.empty and not QRS_series.empty) else np.nan,
        "ST_ms":  ST_series.median() if not ST_series.empty else np.nan,
    }



def save_ecg_strip_png(record_path, out_png, seconds=10, title=None):
    """
    Save a clean ECG strip for all leads.
    """
    record = wfdb.rdrecord(record_path)
    fs = int(record.fs)
    sig = record.p_signal
    sig_names = record.sig_name

    # Determine the number of leads
    num_leads = sig.shape[1]

    # Determine figure layout (e.g., 4 rows x 3 columns for up to 12 leads)
    # Adjust rows/cols dynamically or based on common ECG layouts
    nrows = int(np.ceil(num_leads / 3))
    ncols = 3

    # Adjust overall figure size dynamically
    fig_height = nrows * 2.0  # Adjust as needed
    fig_width = ncols * 5.0   # Adjust as needed

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    n = min(sig.shape[0], fs * seconds)
    t = np.arange(n) / fs

    for i in range(num_leads):
        ax = axes[i]
        y = sig[:n, i]
        ax.plot(t, y)
        ax.set_title(sig_names[i], fontsize=8)
        ax.set_ylabel("mV", fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)

    # Turn off any unused subplots
    for j in range(num_leads, len(axes)):
        fig.delaxes(axes[j])

    # Add a common title
    if title:
        fig.suptitle(title, fontsize=12)

    # Add common x-label (at the bottom of the figure)
    fig.text(0.5, 0.02, "Time (s)", ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle and common x-label

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()



def ecg_id_to_path(ecg_id):
    # ecg_id = "P00135_E01"  -> patient folder "P00135"
    pid = ecg_id.split("_")[0]           # "P00135"
    prefix = pid[:3]                     # "P00" if that's how yours is organized
    return f"{BASE}/{prefix}/{pid}//{ecg_id}"



def subgroup_table_feature_bins(df_eval, features, y_col="_label", prob_col="proba_xgb",
                                prefix="bin_", min_n=150, B=2000, seed=42):

    rows = []
    for f in features:
        bin_col = prefix + f
        if bin_col not in df_eval.columns:
            continue

        for level in pd.Series(df_eval[bin_col]).dropna().unique():
            idx = df_eval.index[df_eval[bin_col] == level]
            if len(idx) < min_n:
                continue

            y_true = df_eval.loc[idx, y_col].astype(int)
            y_prob = df_eval.loc[idx, prob_col].astype(float)

            if y_true.nunique() < 2:
                continue

            auc = roc_auc_score(y_true, y_prob)
            ci_lo, ci_hi = bootstrap_ci_auc(y_true, y_prob, B=B, seed=seed)
            brier = brier_score_loss(y_true, y_prob)

            n = len(idx)
            n_pos = int((y_true == 1).sum())
            pos_pct = 100.0 * n_pos / n

            rows.append({
                "Feature": f,
                "Bin": str(level),
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

    # order: Feature then Low/Mid/High
    bin_order = {"Low": 0, "Mid": 1, "High": 2}
    tbl["Bin_order"] = tbl["Bin"].map(bin_order).fillna(99)
    tbl = tbl.sort_values(["Feature", "Bin_order"]).drop(columns=["Bin_order"]).reset_index(drop=True)
    return tbl



def plot_featurebin_forest_auc(tbl, outpath="figures/Fig_feature_subgroups_AUROC.pdf", x_center=0.5):
    if tbl is None or len(tbl) == 0:
        print("No rows to plot (min_n too strict or bins missing).")
        return

    labels = []
    for _, r in tbl.iterrows():
        labels.append(f'{r["Feature"]}: {r["Bin"]}   (n={int(r["N"])}, pos={int(r["N_pos"])}={r["Pos_%"]:.1f}%)')

    y = np.arange(len(tbl))[::-1]
    fig_h = max(4, 0.28 * len(tbl))
    plt.figure(figsize=(11, fig_h))

    x = tbl["AUROC"].values
    xerr_lo = x - tbl["AUROC_CI_lo"].values
    xerr_hi = tbl["AUROC_CI_hi"].values - x

    plt.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    plt.yticks(y, labels)
    plt.axvline(x_center, linestyle="--")
    plt.xlabel("AUROC (95% CI)")
    plt.title("Feature-stratified robustness (AUROC with 95% CI)")
    plt.tight_layout()

    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()
    print("Saved:", outpath)



def plot_calibration_only(y_true, y_prob, n_bins=10, strategy="quantile",
                          title="Calibration", outpath=None, show_hist=True, hist_bins=20):
    """
    Calibration curve + optional histogram underneath (npj-friendly, compact).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)

    valid = np.isfinite(p) & np.isfinite(y)
    y, p = y[valid], p[valid]

    ece = expected_calibration_error(y, p, n_bins=n_bins, strategy=strategy)
    brier = brier_score_loss(y, p)

    bins = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    p_mean = [b["p_mean"] for b in bins]
    y_rate = [b["y_rate"] for b in bins]

    if show_hist:
        fig, axes = plt.subplots(
            2, 1, figsize=(6.2, 7.0),
            gridspec_kw={"height_ratios": [3, 1]}
        )
        ax = axes[0]
        axh = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2))
        axh = None

    # perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # calibration curve (no explicit colors)
    ax.plot(p_mean, y_rate, marker="o", linewidth=1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"{title}\nECE={ece:.3f} | Brier={brier:.3f}")

    if show_hist and axh is not None:
        axh.hist(p, bins=hist_bins)
        axh.set_xlim(0, 1)
        axh.set_xlabel("Predicted probability")
        axh.set_ylabel("Count")

    plt.tight_layout()

    if outpath:
        import os
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print("Saved:", outpath)

    plt.show()

    return {"ECE": ece, "Brier": brier, "bins": bins, "n_valid": int(len(y))}



def plot_train_test_roc(X, y, models, title_prefix="",
                        test_size=0.2, random_state=42, out_prefix="figures/Fig_ROC"):
    """
    X: DataFrame (features)
    y: Series/array (0/1)
    models: dict name->estimator (must support predict_proba)
    Saves:
      - {out_prefix}_train.pdf
      - {out_prefix}_test.pdf
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    def _plot_split(split_name, Xs, ys, fitted_models, savepath):
        plt.figure(figsize=(7, 6))
        plt.plot([0,1],[0,1],'k--', lw=1)

        for name, mdl in fitted_models.items():
            p = mdl.predict_proba(Xs)[:, 1]
            fpr, tpr, _ = roc_curve(ys, p)
            auc = roc_auc_score(ys, p)
            plt.plot(fpr, tpr, lw=2, label=f"{name} = {auc:.3f}")

        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"{title_prefix} ROC — {split_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.show()

    # Fit on TRAIN only (no leakage)
    fitted = {}
    for name, model in models.items():
        mdl = model
        mdl.fit(X_train, y_train)
        fitted[name] = mdl

    _plot_split("Training set", X_train, y_train, fitted, f"{out_prefix}_train.pdf")
    _plot_split("Internal test set", X_test, y_test, fitted, f"{out_prefix}_test.pdf")

    return (X_train, X_test, y_train, y_test, fitted)
