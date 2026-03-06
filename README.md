# ECG paper code (cleaned Python modules)

This package contains Python files extracted from the notebook used for the paper
submission. The notebook logic was reorganized into modules to make GitHub upload
and review easier.

## Structure

- `src/ecgpaper/preprocessing.py` — parsing, feature cleanup, normalization, axis helpers
- `src/ecgpaper/modeling.py` — model fitting wrappers and cross-validation helpers
- `src/ecgpaper/evaluation.py` — AUROC, calibration, thresholds, subgroup metrics
- `src/ecgpaper/visualization.py` — figures, PDFs, calibration plots, ECG strip utilities

## Notes

- Optional dependencies such as TensorFlow, XGBoost, SHAP, NeuroKit2, and WFDB are imported
  with safe fallbacks so the modules can still be imported in lighter environments.
- These files were extracted from the notebook for version control. You can trim unused
  functions further before public release if desired.


