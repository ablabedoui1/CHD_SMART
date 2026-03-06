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



def fit_developmental_model_one_feature(
    df_controls: pd.DataFrame,
    feature: str,
    degree_age=2,
    degree_hr=2,
    include_interactions=False,
):
    """
    Fit on controls only:
      y = mu(Age, HR, Sex) + eps
    Return model + residual sigma (on controls) + feature clean frame.
    """
    use_cols = ["Age_years", "HR_bpm", "Sex_bin", feature]
    d = df_controls[use_cols].copy()

    # numeric
    d["Age_years"] = pd.to_numeric(d["Age_years"], errors="coerce")
    d["HR_bpm"] = pd.to_numeric(d["HR_bpm"], errors="coerce")
    d["Sex_bin"] = pd.to_numeric(d["Sex_bin"], errors="coerce")
    d[feature] = pd.to_numeric(d[feature], errors="coerce")

    d = d.dropna()
    if len(d) < 50:
        raise ValueError(f"Too few clean control points for {feature}: {len(d)}")

    X, names, pf = build_design(d, degree_age, degree_hr, include_interactions)
    y = d[feature].values

    lr = LinearRegression()
    lr.fit(X, y)

    yhat = lr.predict(X)
    resid = y - yhat
    sigma_res = np.nanstd(resid, ddof=1)  # residual SD

    return {
        "model": lr,
        "design_names": names,
        "poly_obj": pf,
        "sigma_res": float(sigma_res),
        "df_fit": d,          # cleaned controls used for fit
        "yhat_fit": yhat,
        "resid_fit": resid,
    }



def fit_dev_model(df_ref, feature, degree=2,
                  age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin"):
    """
    Fit polynomial regression:
      f ~ Age + HR + Sex + Age^2 + HR^2 + Age*HR (+ Age^3 + HR^3 + Age^2*HR + Age*HR^2 if degree=3)
    Returns:
      model, sigma_res, design_columns
    """
    needed = [age_col, hr_col, sex_col, feature]
    d = df_ref[needed].copy()

    # numeric
    d[age_col] = _to_numeric_series(d[age_col])
    d[hr_col]  = _to_numeric_series(d[hr_col])
    d[sex_col] = _to_numeric_series(d[sex_col])
    d[feature] = _to_numeric_series(d[feature])

    d = d.dropna()
    if len(d) < 30:
        raise ValueError(f"Not enough clean reference samples to fit {feature}: n={len(d)}")

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
    })

    if degree >= 3:
        X["Age3"] = A**3
        X["HR3"] = H**3
        X["Age2HR"] = (A**2)*H
        X["AgeHR2"] = A*(H**2)

    y = d[feature].values

    model = LinearRegression()
    model.fit(X, y)

    # residual std on reference
    yhat = model.predict(X)
    resid = y - yhat
    sigma = np.std(resid, ddof=1)  # unbiased
    if sigma == 0 or np.isnan(sigma):
        sigma = np.nan

    return model, sigma, list(X.columns)



def fit_dev_model_safe(df_ref, feature, degree=2,
                       age_col="Age_years", hr_col="HR_bpm", sex_col="Sex_bin",
                       min_controls=200):
    cols = [age_col, hr_col, sex_col, feature]
    d = df_ref[cols].copy()
    for c in cols:
        d[c] = _to_num(d[c])
    d = d.dropna()
    if len(d) < min_controls:
        return None, None, None

    A, H, S = d[age_col].values, d[hr_col].values, d[sex_col].values
    X = pd.DataFrame({
        "Age": A, "HR": H, "Sex": S,
        "Age2": A**2, "HR2": H**2, "AgeHR": A*H
    })
    if degree >= 3:
        X["Age3"] = A**3
        X["HR3"] = H**3
        X["Age2HR"] = (A**2)*H
        X["AgeHR2"] = A*(H**2)

    y = d[feature].values
    m = LinearRegression().fit(X, y)
    resid = y - m.predict(X)
    sigma = np.std(resid, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = None
    return m, sigma, list(X.columns)



def fit_developmental_model(df_ref, feature, degree=2):
    """
    Fit polynomial regression for one ECG feature
    """
    # Drop rows with NaN values in relevant columns before fitting
    # This is similar to the more robust handling in cell AEbT7P5Kn3pH
    relevant_cols = ["Age_years", "HR_bpm", "Sex_bin", feature]
    df_clean = df_ref[relevant_cols].dropna().copy()

    if df_clean.empty:
        raise ValueError(f"No valid data points to fit model for feature {feature}.")

    X = df_clean[["Age_years", "HR_bpm", "Sex_bin"]].copy() # Ensure X is a copy before adding 'age2'
    X["age2"] = X["Age_years"] ** 2

    y = df_clean[feature].values

    model = LinearRegression()
    model.fit(X, y)

    return model



class KerasDNNWrapper(BaseEstimator, ClassifierMixin):
    """
    Simple sklearn-compatible wrapper around a Keras binary classifier.
    Works with cross_val_predict(method='predict_proba').
    """
    def __init__(self, epochs=40, batch_size=32, verbose=0,
                 hidden_units=(128, 64), dropout=(0.3, 0.2),
                 learning_rate=1e-3):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_ = None

    def _build_model(self, n_features):
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_features,)))
        # first hidden layer
        model.add(layers.Dense(self.hidden_units[0], activation="relu"))
        if self.dropout[0] > 0:
            model.add(layers.Dropout(self.dropout[0]))
        # second hidden layer
        if len(self.hidden_units) > 1:
            model.add(layers.Dense(self.hidden_units[1], activation="relu"))
            if len(self.dropout) > 1 and self.dropout[1] > 0:
                model.add(layers.Dropout(self.dropout[1]))
        # output
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        self.model_ = self._build_model(n_features)
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self

    def predict_proba(self, X):
        X = check_array(X)
        proba_pos = self.model_.predict(X, verbose=0).ravel()
        proba_neg = 1.0 - proba_pos
        return np.vstack([proba_neg, proba_pos]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)



def get_preprocessed_X_and_feature_names(pipe, X_df):
    """
    Works when the pipeline has a step that outputs numpy arrays.
    For SimpleImputer only: feature names are the same as X_df columns.
    """
    # find the transformer part (everything except final estimator)
    if hasattr(pipe, "named_steps"):
        steps = list(pipe.named_steps.items())
        # last step is estimator
        transformer = pipe[:-1]
        Xp = transformer.transform(X_df)
        feature_names = list(X_df.columns)  # imputer doesn't change names
        return Xp, feature_names
    else:
        raise ValueError("Expected a sklearn Pipeline with named_steps.")



def cv_proba(model, Xsub, y):
    return cross_val_predict(model, Xsub, y, cv=cv5, method='predict_proba')[:,1]



def logistic_axes_model():
    return make_pipeline(base_imputer, StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", C=1.0, l1_ratio=0.5, penalty="elasticnet"))



def eval_feature_set_cv(estimator, Xset, y_true, cv, n_bins_ece=10,
                        ruleout_sens=0.95, rulein_spec=0.95):
    """Out-of-fold proba + AUROC/Brier/ECE + operating points."""
    # numeric coercion (prevents xgboost 'object' dtype crash)
    Xn = Xset.copy()
    Xn = Xn.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # if all columns empty → skip
    Xn = Xn.dropna(axis=1, how="all")
    if Xn.shape[1] == 0:
        return None

    est = clone(estimator)
    proba = cross_val_predict(est, Xn, y_true, cv=cv, method="predict_proba")[:,1]

    # threshold-free
    out = dict(
        Dim=int(Xn.shape[1]),
        AUROC=float(roc_auc_score(y_true, proba)),
        Brier=float(brier_score_loss(y_true, proba)),
        ECE=float(ece_score(y_true, proba, n_bins=n_bins_ece))
    )

    # operating points
    t_y = youden_threshold(y_true, proba)
    t_ro = find_threshold_for_target(y_true, proba, target="sens", value=ruleout_sens)
    t_ri = find_threshold_for_target(y_true, proba, target="spec", value=rulein_spec)

    m_y  = metrics_at_threshold(y_true, proba, t_y)
    m_ro = metrics_at_threshold(y_true, proba, t_ro) if np.isfinite(t_ro) else {}
    m_ri = metrics_at_threshold(y_true, proba, t_ri) if np.isfinite(t_ri) else {}

    return out, proba, (t_y, m_y), (t_ro, m_ro), (t_ri, m_ri)



def build_labels(df, mode="CHD_vs_nonCHD"):
    """
    Returns y (0/1) and a mask of rows to keep.
    mode:
      - "CHD_vs_nonCHD": 1=CHD, 0=everything else
      - "CHD_vs_normal": 1=CHD, 0=normal only (exclude other diseases)
    Requirements:
      df must contain the columns:
        - congenital (1/0) or your _label column
        - other disease flags if you want "normal-only" (e.g., myocarditis, cardiomyopathy, kawasaki)
    """
    # --- Positive class (CHD)
    if "congenital" in df.columns:
        y_chd = pd.to_numeric(df["congenital"], errors="coerce")
    elif "_label" in df.columns:
        y_chd = pd.to_numeric(df["_label"], errors="coerce")
    else:
        raise ValueError("Need either 'congenital' or '_label' in df.")

    # force 0/1
    y_chd = (y_chd == 1).astype(int)

    if mode == "CHD_vs_nonCHD":
        keep = y_chd.notna()
        y = y_chd.loc[keep].astype(int)

    elif mode == "CHD_vs_normal":
        # Define "normal" = NOT CHD and NOT other diseases
        needed_flags = ["myocarditis", "cardiomyopathy", "kawasaki"]
        missing = [c for c in needed_flags if c not in df.columns]
        if missing:
            raise ValueError(f"To do CHD_vs_normal, missing flags: {missing}")

        other = np.zeros(len(df), dtype=bool)
        for c in needed_flags:
            other |= (pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) == 1).values

        normal = (y_chd.values == 0) & (~other)
        keep = normal | (y_chd.values == 1)

        y = y_chd.loc[keep].astype(int)

    else:
        raise ValueError("mode must be 'CHD_vs_nonCHD' or 'CHD_vs_normal'")

    return y, keep
