# %% [markdown]
# # Global pK Prediction Model Training (vfinal)
#
# Trains 3 FLAML regression models to predict true pK (binding affinity)
# using the same filtered feature set as the metamodels.
#
# Three test set variants:
#   Model 1 — CASF-2016 test set
#   Model 2 — Zero ligand bias test set
#   Model 3 — OOD test set (split column in index_oodtest.csv)
#
# Features: same 207 from metamodel_outputs_final/metrics/feature_columns.txt
# Label: pred_true_pK from merged_descriptors_predictions.csv
# Contaminating columns dropped: pred_error, pred_abs_error, pred_pred_*, pred_pred_pK
#
# Conda environment: metamodels

# %% Configuration
from pathlib import Path

DATA_PATH       = Path.home() / "Desktop" / "merged_descriptors_predictions.csv"
FEATURE_LIST    = Path.home() / "Desktop" / "metamodel_outputs_final" / "metrics" / "feature_columns.txt"
BENCHMARK_DIR   = Path.home() / "AEV-PLIG-modifications" / "benchmarks"
OUTPUT_DIR      = Path.home() / "Desktop" / "global_model_outputs_final"

CASF_CSV        = BENCHMARK_DIR / "casf_2016_test.csv"
ZERO_BIAS_CSV   = BENCHMARK_DIR / "zero_ligand_bias_test.csv"
OOD_CSV         = BENCHMARK_DIR / "index_oodtest.csv"

LABEL_COL       = "pred_true_pK"
ID_COL          = "unique_id"

# Columns derived from the model predictions — must not be used as features
CONTAMINATING_COLS = (
    ["pred_error", "pred_abs_error", "pred_pred_pK",
     "weighted_error", "weighted_abs_error",
     "dataset", "protein_path", "ligand_path"]
    + [f"pred_pred_{i}" for i in range(10)]
)

# === Training Parameters (match metamodel settings) ===
TIME_BUDGET_SECONDS = 1800
TEST_SIZE           = None   # not used — test set is externally defined
N_SPLITS            = 5
RANDOM_STATE        = 42
VERBOSE             = 2
ESTIMATOR_LIST      = "auto"

print("Configuration loaded")
print(f"  Data:       {DATA_PATH}")
print(f"  Output:     {OUTPUT_DIR}")
print(f"  Time budget:{TIME_BUDGET_SECONDS}s per model ({TIME_BUDGET_SECONDS/60:.0f} min)")

# %% Imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from scipy import stats

from flaml import AutoML

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for sub in ["models", "metrics", "plots", "feature_importance", "logs", "shap"]:
    (OUTPUT_DIR / sub).mkdir(exist_ok=True)

print(f"FLAML version: {__import__('flaml').__version__}")
print("Imports OK")

# %% Load feature list
with open(FEATURE_LIST) as f:
    feature_columns = [line.strip() for line in f if line.strip()]
print(f"Feature list loaded: {len(feature_columns)} features")

# %% Load data
print(f"\nLoading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Verify label and features present
assert LABEL_COL in df.columns, f"Label column '{LABEL_COL}' not found"
missing_feats = [c for c in feature_columns if c not in df.columns]
if missing_feats:
    print(f"  WARNING: {len(missing_feats)} feature columns not found in data — will be NaN")
    for c in missing_feats:
        df[c] = np.nan

label_nan = df[LABEL_COL].isna().sum()
if label_nan > 0:
    print(f"  Dropping {label_nan} rows with NaN label")
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

print(f"  Label mean={df[LABEL_COL].mean():.3f}  std={df[LABEL_COL].std():.3f}")

# %% Load test set PDB code lists
casf_ids     = set(pd.read_csv(CASF_CSV)["key"].astype(str))
zero_ids     = set(pd.read_csv(ZERO_BIAS_CSV)["key"].astype(str))

ood_df       = pd.read_csv(OOD_CSV)
ood_test_ids = set(ood_df.loc[ood_df["split"] == "test", "PDB_code"].astype(str))

print(f"\nTest set sizes (unique PDB codes):")
print(f"  CASF-2016:         {len(casf_ids)}")
print(f"  Zero ligand bias:  {len(zero_ids)}")
print(f"  OOD test:          {len(ood_test_ids)}")

# Rows matched in df
for label, ids in [("CASF", casf_ids), ("Zero bias", zero_ids), ("OOD", ood_test_ids)]:
    n = df[ID_COL].isin(ids).sum()
    print(f"  {label} rows in data: {n:,}")

# %% Preprocessing helper
def preprocess_features(df_train, df_test, feature_cols):
    """Convert to numpy, replace inf, clip, impute with median fitted on train only."""
    def _to_clean_array(frame):
        X = frame[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.where(np.isinf(X), np.nan, X)
        X = np.clip(X, -1e30, 1e30)
        return X

    X_tr = _to_clean_array(df_train)
    X_te = _to_clean_array(df_test)

    imputer = SimpleImputer(strategy='median')
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    return X_tr, X_te, imputer

# %% Training + evaluation helpers (same as metamodel script)
def train_model(X_train, y_train, model_name):
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}   Features: {X_train.shape[1]}")
    print(f"  Time budget: {TIME_BUDGET_SECONDS}s")

    automl = AutoML()
    settings = {
        "time_budget": TIME_BUDGET_SECONDS,
        "metric": "rmse",
        "task": "regression",
        "n_jobs": -1,
        "eval_method": "cv",
        "n_splits": N_SPLITS,
        "verbose": VERBOSE,
        "seed": RANDOM_STATE,
        "early_stop": True,
        "ensemble": False,
        "skip_transform": True,
        "log_file_name": str(OUTPUT_DIR / "logs" / f"{model_name}_flaml.log"),
    }
    if ESTIMATOR_LIST != "auto":
        settings["estimator_list"] = ESTIMATOR_LIST

    start = datetime.now()
    print(f"  Starting at {start.strftime('%H:%M:%S')} ...")
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    duration = (datetime.now() - start).total_seconds()

    print(f"  ✓ Done in {duration:.0f}s | best: {automl.best_estimator} | CV RMSE: {automl.best_loss:.4f}")
    return automl


def evaluate_model(automl, X_test, y_test, model_name):
    y_pred = automl.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r, p = stats.pearsonr(y_test, y_pred)

    print(f"\n  Test metrics for {model_name}:")
    print(f"    R²={r2:.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}   r={r:.4f}")

    metrics = {
        "model": model_name,
        "r2": r2, "rmse": rmse, "mae": mae,
        "pearson_r": r, "pearson_p": p,
        "n_test": len(y_test),
        "best_estimator": automl.best_estimator,
        "best_cv_rmse": automl.best_loss,
    }
    return metrics, y_pred


def get_feature_importance(automl, feat_names):
    est = automl.model.estimator
    if hasattr(est, 'feature_importances_'):
        imp = est.feature_importances_
    elif hasattr(est, 'coef_'):
        imp = np.abs(est.coef_)
    else:
        return None
    fi = pd.DataFrame({'feature': feat_names, 'importance': imp})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    fi['rank'] = fi.index + 1
    fi['importance_normalized'] = fi['importance'] / fi['importance'].sum()
    return fi


def plot_actual_vs_predicted(y_true, y_pred, model_name, metrics, save_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, c='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('True pK', fontsize=11)
    ax.set_ylabel('Predicted pK', fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    textstr = (f"R²={metrics['r2']:.4f}\nRMSE={metrics['rmse']:.4f}\n"
               f"r={metrics['pearson_r']:.4f}\n{metrics['best_estimator']}\nn={metrics['n_test']:,}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# %% Define the three training jobs
JOBS = [
    {
        "name":     "casf_2016",
        "test_ids": casf_ids,
        "id_col":   ID_COL,
    },
    {
        "name":     "zero_ligand_bias",
        "test_ids": zero_ids,
        "id_col":   ID_COL,
    },
    {
        "name":     "ood_test",
        "test_ids": ood_test_ids,
        "id_col":   ID_COL,
    },
]

# %% Train all 3 models
all_metrics = []
all_predictions = {}
all_feature_importance = {}

pipeline_start = datetime.now()

for job in JOBS:
    name     = job["name"]
    test_ids = job["test_ids"]

    test_mask  = df[ID_COL].isin(test_ids)
    if "train_ids" in job:
        train_mask = df[ID_COL].isin(job["train_ids"])
    else:
        train_mask = ~test_mask

    df_train = df[train_mask].reset_index(drop=True)
    df_test  = df[test_mask].reset_index(drop=True)

    print(f"\n{'#'*60}")
    print(f"# MODEL: {name}")
    print(f"#   Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")
    print(f"{'#'*60}")

    if len(df_test) == 0:
        print(f"  WARNING: no test rows found for {name} — skipping")
        continue

    y_train = df_train[LABEL_COL].values
    y_test  = df_test[LABEL_COL].values

    X_train, X_test, imputer = preprocess_features(df_train, df_test, feature_columns)

    # Save imputer (test-set specific)
    joblib.dump(imputer, OUTPUT_DIR / "models" / f"{name}_imputer.joblib")

    automl = train_model(X_train, y_train, name)
    metrics, y_pred = evaluate_model(automl, X_test, y_test, name)

    all_metrics.append(metrics)
    all_predictions[name] = {
        "y_true": y_test, "y_pred": y_pred,
        "unique_ids": df_test[ID_COL].values,
    }

    # Feature importance
    fi = get_feature_importance(automl, feature_columns)
    if fi is not None:
        all_feature_importance[name] = fi
        fi.to_csv(OUTPUT_DIR / "feature_importance" / f"{name}_feature_importance.csv", index=False)

    # Actual vs predicted plot
    plot_actual_vs_predicted(
        y_test, y_pred, name, metrics,
        OUTPUT_DIR / "plots" / f"{name}_actual_vs_predicted.png"
    )

    # Save model
    joblib.dump(automl, OUTPUT_DIR / "models" / f"{name}_flaml_model.pkl")
    print(f"  Model saved: {name}_flaml_model.pkl")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "unique_id": df_test[ID_COL].values,
        "true_pK":   y_test,
        "pred_pK":   y_pred,
        "error":     y_pred - y_test,
        "abs_error": np.abs(y_pred - y_test),
    })
    pred_df.to_csv(OUTPUT_DIR / "metrics" / f"{name}_predictions.csv", index=False)

pipeline_end = datetime.now()
total_duration = (pipeline_end - pipeline_start).total_seconds()

print(f"\n\n{'#'*60}")
print(f"# ALL MODELS COMPLETE — {total_duration:.0f}s ({total_duration/60:.1f} min)")
print(f"{'#'*60}")

# %% Summary table
summary_df = pd.DataFrame(all_metrics)
summary_df = summary_df[['model', 'best_estimator', 'r2', 'rmse', 'mae', 'pearson_r', 'best_cv_rmse', 'n_test']]
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(summary_df.to_string(index=False))
summary_df.to_csv(OUTPUT_DIR / "metrics" / "all_models_summary.csv", index=False)
print(f"\n✓ Summary saved")

# %% Combined actual vs predicted (1×3)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (name, preds) in enumerate(all_predictions.items()):
    ax = axes[i]
    m = next(m for m in all_metrics if m["model"] == name)
    ax.scatter(preds["y_true"], preds["y_pred"], alpha=0.3, s=8, c='steelblue')
    lims = [min(preds["y_true"].min(), preds["y_pred"].min()),
            max(preds["y_true"].max(), preds["y_pred"].max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('True pK', fontsize=10)
    ax.set_ylabel('Predicted pK', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.text(0.05, 0.95,
            f"R²={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nr={m['pearson_r']:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')

plt.suptitle('Global pK Model — All Test Sets', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "all_models_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Combined plot saved")

# %% Feature importance (1×3)
fig, axes = plt.subplots(1, 3, figsize=(18, 9))
for i, name in enumerate(all_predictions.keys()):
    ax = axes[i]
    if name in all_feature_importance:
        fi = all_feature_importance[name].head(20)
        ax.barh(range(len(fi)), fi['importance_normalized'].values, color='steelblue')
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi['feature'].values, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Normalised importance', fontsize=9)
        ax.set_title(f'{name}\nTop 20 features', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax.set_title(name)

plt.suptitle('Feature Importance — Global pK Models', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "feature_importance_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Feature importance plot saved")

# %% Save feature list and metadata
with open(OUTPUT_DIR / "metrics" / "feature_columns.txt", 'w') as f:
    f.write('\n'.join(feature_columns))

metadata = {
    "data_path": str(DATA_PATH),
    "feature_list_source": str(FEATURE_LIST),
    "label": LABEL_COL,
    "n_features": len(feature_columns),
    "time_budget_seconds": TIME_BUDGET_SECONDS,
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "timestamp": datetime.now().isoformat(),
    "total_duration_seconds": total_duration,
    "models": {m["model"]: {k: v for k, v in m.items() if k != "model"} for m in all_metrics},
}
with open(OUTPUT_DIR / "metrics" / "training_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved")

# %% SHAP Analysis — all 3 models
import shap

SHAP_SAMPLE_SIZE     = 1000
SHAP_BACKGROUND_SIZE = 100

def _compute_shap(estimator, X_shap, X_bg, mode="path_dependent"):
    if mode == "path_dependent":
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.TreeExplainer(
            estimator, data=X_bg, feature_perturbation="interventional"
        )
    sv = explainer.shap_values(X_shap, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    return sv

def _beeswarm(sv, X_shap, feat_names, title, save_path, max_display=10):
    ax = plt.subplots(figsize=(10, 8))[1]
    plt.sca(ax)
    shap.summary_plot(sv, X_shap, feature_names=feat_names,
                      max_display=max_display, plot_type='dot', show=False)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

rng = np.random.default_rng(RANDOM_STATE)

for job in JOBS:
    name = job["name"]
    model_path = OUTPUT_DIR / "models" / f"{name}_flaml_model.pkl"
    if not model_path.exists():
        print(f"  SKIP SHAP {name}: model not found")
        continue

    automl   = joblib.load(model_path)
    imputer  = joblib.load(OUTPUT_DIR / "models" / f"{name}_imputer.joblib")
    estimator = automl.model.estimator

    # Rebuild train set for SHAP sample
    test_ids = job["test_ids"]
    if "train_ids" in job:
        train_mask = df[ID_COL].isin(job["train_ids"])
    else:
        train_mask = ~df[ID_COL].isin(test_ids)
    df_train = df[train_mask].reset_index(drop=True)

    X_tr_raw = df_train[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
    X_tr_raw = np.where(np.isinf(X_tr_raw), np.nan, X_tr_raw)
    X_tr_raw = np.clip(X_tr_raw, -1e30, 1e30)
    X_tr_clean = imputer.transform(X_tr_raw)

    n_shap = min(SHAP_SAMPLE_SIZE, len(X_tr_clean))
    n_bg   = min(SHAP_BACKGROUND_SIZE, len(X_tr_clean))
    shap_idx = rng.choice(len(X_tr_clean), size=n_shap, replace=False)
    bg_idx   = rng.choice(len(X_tr_clean), size=n_bg,   replace=False)

    X_shap = X_tr_clean[shap_idx]
    X_bg   = X_tr_clean[bg_idx]

    print(f"\nSHAP: {name}")
    try:
        sv_pd = _compute_shap(estimator, X_shap, X_bg, mode="path_dependent")
        _beeswarm(sv_pd, X_shap, feature_columns,
                  f"SHAP (path-dependent) — {name}",
                  OUTPUT_DIR / "shap" / f"{name}_beeswarm_pathdep.png")
        print(f"  ✓ Path-dependent beeswarm saved")
    except Exception as e:
        print(f"  ERROR (path-dependent): {e}")

    try:
        sv_iv = _compute_shap(estimator, X_shap, X_bg, mode="interventional")
        _beeswarm(sv_iv, X_shap, feature_columns,
                  f"SHAP (interventional) — {name}",
                  OUTPUT_DIR / "shap" / f"{name}_beeswarm_interventional.png")
        print(f"  ✓ Interventional beeswarm saved")

        imp_df = pd.DataFrame({
            'feature': feature_columns,
            'mean_abs_shap_pathdep':        np.abs(sv_pd).mean(axis=0),
            'mean_abs_shap_interventional': np.abs(sv_iv).mean(axis=0),
        }).sort_values('mean_abs_shap_interventional', ascending=False)
        imp_df['rank_pathdep']        = imp_df['mean_abs_shap_pathdep'].rank(ascending=False).astype(int)
        imp_df['rank_interventional'] = range(1, len(imp_df) + 1)
        imp_df.to_csv(OUTPUT_DIR / "shap" / f"{name}_shap_importance.csv", index=False)
        print(f"  ✓ SHAP importance CSV saved")

    except Exception as e:
        print(f"  ERROR (interventional): {e}")

print(f"\n✓ All done. Outputs in: {OUTPUT_DIR}")
# %%
