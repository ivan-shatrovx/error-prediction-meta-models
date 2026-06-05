
# FLAML Metamodel Training Pipeline
#
# Trains regression metamodels to predict weighted signed and absolute errors
# using the filtered feature set from metadata_inputs_final.csv
#
# Targets: weighted_error, weighted_abs_error
# Features: rdk_*, arp_*, dpk_* columns remaining after final filtering

# Configuration
import os
from pathlib import Path

# File Paths
DATA_PATH = Path.home() / "Desktop" / "metamodel_inputs_final.csv"
OUTPUT_DIR = Path.home() / "Desktop" / "metamodel_outputs_final"

# Target Variables
TARGET_COLUMNS = [
    "weighted_error",
    "weighted_abs_error",
]

# Columns to drop immediately after loading
LEAKAGE_COLUMNS = ["pred_error", "pred_abs_error"]

# Prefixes 
FEATURE_PREFIXES = ("rdk_", "arp_", "dpk_")

# Columns to Exclude 
EXCLUDE_COLUMNS = [
    "unique_id", "dataset", "protein_path", "ligand_path", "pred_fold",
    "pred_pred_pK", "pred_true_pK"
] + [f"pred_pred_{i}" for i in range(10)]

TIME_BUDGET_SECONDS = 1800  # 30 minutes per model 

# Holdout test set proportion
TEST_SIZE = 0.10

# Cross-validation folds (FLAML default is 5)
N_SPLITS = 5

# Random seed for reproducibility
RANDOM_STATE = 42

# FLAML verbosity (0=silent, 1=progress, 2=detailed, 3=debug)
VERBOSE = 2

# Memory limit in MB (set based on your 32GB RAM)
MEMORY_LIMIT_MB = 28000  # Leave ~4GB for system

# Estimator List (auto is FLAML default)
# Options: 'lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'catboost'
ESTIMATOR_LIST = "auto"  

print(f"Configuration loaded:")
print(f"  Data path: {DATA_PATH}")
print(f"  Output dir: {OUTPUT_DIR}")
print(f"  Time budget: {TIME_BUDGET_SECONDS}s per model ({TIME_BUDGET_SECONDS/60:.1f} min)")
print(f"  Test size: {TEST_SIZE*100:.0f}%")
print(f"  CV folds: {N_SPLITS}")

# Imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

from flaml import AutoML

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("All imports successful!")
print(f"FLAML version: {__import__('flaml').__version__}")

# Create Output Directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory created/verified: {OUTPUT_DIR}")

(OUTPUT_DIR / "models").mkdir(exist_ok=True)
(OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "feature_importance").mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)

# Load and Explore Data
print(f"\nLoading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Drop leakage columns 
leakage_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
if leakage_present:
    df = df.drop(columns=leakage_present)
    print(f"Dropped leakage columns: {leakage_present}")

print(f"\n{'='*60}")
print("DATA OVERVIEW")
print(f"{'='*60}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Check target columns exist
missing_targets = [t for t in TARGET_COLUMNS if t not in df.columns]
if missing_targets:
    raise ValueError(f"Missing target columns: {missing_targets}")
print(f"\n✓ All {len(TARGET_COLUMNS)} target columns found")

# Feature Selection
print(f"\n{'='*60}")
print("FEATURE SELECTION")
print(f"{'='*60}")

all_columns = df.columns.tolist()

feature_columns = [
    col for col in all_columns
    if col.startswith(FEATURE_PREFIXES)
    and col not in TARGET_COLUMNS
    and col not in EXCLUDE_COLUMNS
]

prefix_counts = {}
for prefix in FEATURE_PREFIXES:
    count = sum(1 for col in feature_columns if col.startswith(prefix))
    prefix_counts[prefix] = count
    print(f"  {prefix}* features: {count}")

print(f"\nTotal features selected: {len(feature_columns)}")

overlap = set(feature_columns) & set(TARGET_COLUMNS)
if overlap:
    raise ValueError(f"Target columns found in features: {overlap}")
print("✓ No overlap between features and targets")

# Data Preprocessing
print(f"\n{'='*60}")
print("DATA PREPROCESSING")
print(f"{'='*60}")

y_dict = {target: df[target].copy() for target in TARGET_COLUMNS}

for target in TARGET_COLUMNS:
    nan_count = y_dict[target].isna().sum()
    if nan_count > 0:
        print(f"  Warning: {target} has {nan_count} NaN values - will be dropped during training")

# Step 1: Convert to numpy float64, coercing pd.NA → np.nan
X_raw = df[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
print(f"\nRaw feature matrix: {X_raw.shape}")

# Step 2: Replace inf with NaN
n_inf = np.isinf(X_raw).sum()
if n_inf > 0:
    print(f"  Replaced {n_inf:,} inf values with NaN")
X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)

# Step 3: Clip extreme values to ±1e30
n_extreme = ((X_raw > 1e30) | (X_raw < -1e30)).sum()
if n_extreme > 0:
    print(f"  Clipped {n_extreme:,} extreme values to ±1e30")
X_raw = np.clip(X_raw, -1e30, 1e30)

# Step 4: Impute remaining NaN with median
n_nan = np.isnan(X_raw).sum()
print(f"  NaN values to impute: {n_nan:,}")

imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X_raw)

joblib.dump(imputer, OUTPUT_DIR / "models" / "feature_imputer.joblib")
print("✓ Imputer fitted and saved to models/feature_imputer.joblib")

X = pd.DataFrame(X_clean, columns=feature_columns, index=df.index)

assert not np.any(np.isnan(X.values)), "NaN values remain after preprocessing!"
assert not np.any(np.isinf(X.values)), "Inf values remain after preprocessing!"
print(f"\nFinal feature matrix: {X.shape} — no NaN, no inf ✓")

# Train-Test Split
print(f"\n{'='*60}")
print("TRAIN-TEST SPLIT")
print(f"{'='*60}")

X_train, X_test, indices_train, indices_test = train_test_split(
    X, X.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training set: {len(X_train):,} samples ({100-TEST_SIZE*100:.0f}%)")
print(f"Test set: {len(X_test):,} samples ({TEST_SIZE*100:.0f}%)")

np.save(OUTPUT_DIR / "metrics" / "test_indices.npy", indices_test.values)
print("✓ Test indices saved")

# Define Training Functions
def train_metamodel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    time_budget: int = TIME_BUDGET_SECONDS,
    n_splits: int = N_SPLITS,
    verbose: int = VERBOSE
) -> AutoML:
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {target_name}")
    print(f"{'='*60}")
    print(f"Time budget: {time_budget}s ({time_budget/60:.1f} min)")
    print(f"Training samples: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")

    valid_mask = ~y_train.isna()
    if not valid_mask.all():
        print(f"Dropping {(~valid_mask).sum()} samples with NaN target")
        X_train_clean = X_train[valid_mask]
        y_train_clean = y_train[valid_mask]
    else:
        X_train_clean = X_train
        y_train_clean = y_train

    automl = AutoML()

    settings = {
        "time_budget": time_budget,
        "metric": "rmse",
        "task": "regression",
        "n_jobs": -1,
        "eval_method": "cv",
        "n_splits": n_splits,
        "verbose": verbose,
        "seed": RANDOM_STATE,
        "early_stop": True,
        "ensemble": False,
        "skip_transform": True,
        "log_file_name": str(OUTPUT_DIR / "logs" / f"{target_name}_flaml.log"),
    }

    if ESTIMATOR_LIST != "auto":
        settings["estimator_list"] = ESTIMATOR_LIST

    start_time = datetime.now()
    print(f"\nStarting training at {start_time.strftime('%H:%M:%S')}...")

    automl.fit(
        X_train=X_train_clean.to_numpy() if hasattr(X_train_clean, 'to_numpy') else X_train_clean,
        y_train=y_train_clean.to_numpy() if hasattr(y_train_clean, 'to_numpy') else y_train_clean,
        **settings
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n✓ Training complete!")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Best estimator: {automl.best_estimator}")
    print(f"  Best CV score (RMSE): {automl.best_loss:.4f}")

    return automl


def evaluate_model(
    automl: AutoML,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str
) -> dict:
    valid_mask = ~y_test.isna()
    if not valid_mask.all():
        print(f"  Dropping {(~valid_mask).sum()} test samples with NaN target")
        X_test_clean = X_test[valid_mask]
        y_test_clean = y_test[valid_mask]
    else:
        X_test_clean = X_test
        y_test_clean = y_test

    X_test_np = X_test_clean.to_numpy() if hasattr(X_test_clean, 'to_numpy') else X_test_clean
    y_pred = automl.predict(X_test_np)

    r2 = r2_score(y_test_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred))
    mae = mean_absolute_error(y_test_clean, y_pred)

    metrics = {
        "target": target_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "n_test": len(y_test_clean),
        "best_estimator": automl.best_estimator,
        "best_cv_rmse": automl.best_loss
    }

    print(f"\n  Holdout Test Metrics for {target_name}:")
    print(f"    R²:   {r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")

    return metrics, y_test_clean, y_pred


def get_feature_importance(automl: AutoML, feature_names: list) -> pd.DataFrame:
    model = automl.model.estimator
    importance = None

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)

    if importance is not None:
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        fi_df['rank'] = range(1, len(fi_df) + 1)
        fi_df['importance_normalized'] = fi_df['importance'] / fi_df['importance'].sum()
        return fi_df
    else:
        print(f"  Warning: Could not extract feature importance from {type(model)}")
        return None


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    metrics: dict,
    save_path: Path
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')

    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'Actual vs Predicted: {target_name}', fontsize=14, fontweight='bold')

    textstr = '\n'.join([
        f"R² = {metrics['r2']:.4f}",
        f"RMSE = {metrics['rmse']:.4f}",
        f"MAE = {metrics['mae']:.4f}",
        f"Model: {metrics['best_estimator']}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {save_path.name}")

# Train All Models
print(f"\n{'#'*60}")
print("# STARTING TRAINING PIPELINE")
print(f"# Total models to train: {len(TARGET_COLUMNS)}")
print(f"# Estimated total time: {len(TARGET_COLUMNS) * TIME_BUDGET_SECONDS / 60:.0f} min")
print(f"{'#'*60}")

all_models = {}
all_metrics = []
all_feature_importance = {}
all_predictions = {}

pipeline_start = datetime.now()

for i, target in enumerate(TARGET_COLUMNS, 1):
    print(f"\n\n{'='*60}")
    print(f"[{i}/{len(TARGET_COLUMNS)}] Processing target: {target}")
    print(f"{'='*60}")

    y_train = y_dict[target].loc[indices_train]
    y_test = y_dict[target].loc[indices_test]

    automl = train_metamodel(X_train, y_train, target)
    all_models[target] = automl

    metrics, y_test_clean, y_pred = evaluate_model(automl, X_test, y_test, target)
    all_metrics.append(metrics)
    all_predictions[target] = {'y_true': y_test_clean, 'y_pred': y_pred}

    fi_df = get_feature_importance(automl, feature_columns)
    if fi_df is not None:
        all_feature_importance[target] = fi_df
        fi_path = OUTPUT_DIR / "feature_importance" / f"{target}_feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importance saved: {fi_path.name}")

    plot_path = OUTPUT_DIR / "plots" / f"{target}_actual_vs_predicted.png"
    plot_actual_vs_predicted(y_test_clean.values, y_pred, target, metrics, plot_path)

    model_path = OUTPUT_DIR / "models" / f"{target}_flaml_model.pkl"
    with open(model_path, 'wb') as f:
        joblib.dump(automl, f)
    print(f"  Model saved: {model_path.name}")

pipeline_end = datetime.now()
total_duration = (pipeline_end - pipeline_start).total_seconds()

print(f"\n\n{'#'*60}")
print("# PIPELINE COMPLETE")
print(f"# Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
print(f"{'#'*60}")

# Summary Metrics Table
print(f"\n{'='*60}")
print("SUMMARY: MODEL COMPARISON")
print(f"{'='*60}\n")

summary_df = pd.DataFrame(all_metrics)
summary_df = summary_df[['target', 'best_estimator', 'r2', 'rmse', 'mae', 'best_cv_rmse', 'n_test']]

print(summary_df.to_string(index=False))

summary_path = OUTPUT_DIR / "metrics" / "model_comparison_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary saved to: {summary_path}")

summary_txt_path = OUTPUT_DIR / "metrics" / "model_comparison_summary.txt"
with open(summary_txt_path, 'w') as f:
    f.write("FLAML Metamodel Training Summary (vfinal)\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Time budget per model: {TIME_BUDGET_SECONDS}s\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Test samples: {len(X_test):,}\n")
    f.write(f"Features: {len(feature_columns)}\n")
    f.write(f"\n{'='*80}\n\n")
    f.write(summary_df.to_string(index=False))
    f.write(f"\n\n{'='*80}\n")
    f.write(f"Total pipeline duration: {total_duration:.1f}s ({total_duration/60:.1f} min)\n")

# Combined Visualization 
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, target in enumerate(TARGET_COLUMNS):
    ax = axes[i]

    y_true = all_predictions[target]['y_true']
    y_pred = all_predictions[target]['y_pred']
    metrics = all_metrics[i]

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    ax.plot(lims, lims, 'r--', lw=2)

    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{target}', fontsize=12, fontweight='bold')

    textstr = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}\n{metrics['best_estimator']}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_aspect('equal')

plt.suptitle('Metamodel Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
combined_path = OUTPUT_DIR / "plots" / "all_models_comparison.png"
plt.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Combined plot saved: {combined_path}")

# Feature Importance Visualization 
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for i, target in enumerate(TARGET_COLUMNS):
    ax = axes[i]

    if target in all_feature_importance:
        fi_df = all_feature_importance[target].head(20)

        ax.barh(range(len(fi_df)), fi_df['importance_normalized'].values, color='steelblue')
        ax.set_yticks(range(len(fi_df)))
        ax.set_yticklabels(fi_df['feature'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Normalized Importance', fontsize=10)
        ax.set_title(f'{target} - Top 20 Features', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Feature importance\nnot available',
                ha='center', va='center', fontsize=12)
        ax.set_title(f'{target}', fontsize=11)

plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
fi_plot_path = OUTPUT_DIR / "plots" / "feature_importance_comparison.png"
plt.savefig(fi_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Feature importance plot saved: {fi_plot_path}")

# Save Metadata
metadata = {
    "data_path": str(DATA_PATH),
    "output_dir": str(OUTPUT_DIR),
    "targets": TARGET_COLUMNS,
    "leakage_columns_dropped": LEAKAGE_COLUMNS,
    "feature_prefixes": FEATURE_PREFIXES,
    "n_features": len(feature_columns),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "test_size": TEST_SIZE,
    "time_budget_seconds": TIME_BUDGET_SECONDS,
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "timestamp": datetime.now().isoformat(),
    "total_duration_seconds": total_duration,
    "feature_columns": feature_columns
}

metadata_path = OUTPUT_DIR / "metrics" / "training_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

feature_list_path = OUTPUT_DIR / "metrics" / "feature_columns.txt"
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_columns))
print(f"✓ Feature list saved: {feature_list_path}")

# Final Summary
print(f"\n\n{'#'*60}")
print("# TRAINING COMPLETE - OUTPUT SUMMARY")
print(f"{'#'*60}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nmodels/")
for target in TARGET_COLUMNS:
    print(f"   ├── {target}_flaml_model.pkl")
print(f"   └── feature_imputer.joblib")
print(f"\nmetrics/")
print(f"   ├── model_comparison_summary.csv")
print(f"   ├── model_comparison_summary.txt")
print(f"   ├── training_metadata.json")
print(f"   ├── feature_columns.txt")
print(f"   └── test_indices.npy")
print(f"\nplots/")
print(f"   ├── all_models_comparison.png")
print(f"   ├── feature_importance_comparison.png")
for target in TARGET_COLUMNS:
    print(f"   ├── {target}_actual_vs_predicted.png")
print(f"\nfeature_importance/")
for target in TARGET_COLUMNS:
    print(f"   ├── {target}_feature_importance.csv")
print(f"\nlogs/")
for target in TARGET_COLUMNS:
    print(f"   ├── {target}_flaml.log")

# Load pK columns from merged_descriptors_predictions.csv

MERGED_PATH = Path.home() / "Desktop" / "merged_descriptors_predictions.csv"

print(f"\nLoading pK columns from: {MERGED_PATH}")
df_merged = pd.read_csv(MERGED_PATH, usecols=["pred_pred_pK", "pred_true_pK"])

assert len(df_merged) == len(df), (
    f"Row count mismatch: merged={len(df_merged)}, features={len(df)}. "
    "Cannot safely align pK values."
)

# Align index to match X
df_merged.index = X.index

pred_pk_values = df_merged["pred_pred_pK"].values
true_pk_values = df_merged["pred_true_pK"].values

print(f"  pred_pred_pK: mean={pred_pk_values.mean():.3f}  std={pred_pk_values.std():.3f}  NaN={np.isnan(pred_pk_values).sum()}")
print(f"  pred_true_pK: mean={true_pk_values.mean():.3f}  std={true_pk_values.std():.3f}  NaN={np.isnan(true_pk_values).sum()}")

# Build augmented feature DataFrames (add pK as extra column)
X_pred_pk = X.copy()
X_pred_pk["pred_pk"] = pred_pk_values

X_true_pk = X.copy()
X_true_pk["true_pk"] = true_pk_values

feature_columns_pred_pk = feature_columns + ["pred_pk"]
feature_columns_true_pk = feature_columns + ["true_pk"]

# Apply same train/test split indices
X_pred_pk_train = X_pred_pk.loc[indices_train]
X_pred_pk_test  = X_pred_pk.loc[indices_test]

X_true_pk_train = X_true_pk.loc[indices_train]
X_true_pk_test  = X_true_pk.loc[indices_test]

# Save augmented feature lists
pk_feature_list_path = OUTPUT_DIR / "metrics" / "feature_columns_pred_pk.txt"
with open(pk_feature_list_path, 'w') as f:
    f.write('\n'.join(feature_columns_pred_pk))
print(f"✓ Feature list (pred_pk) saved: {pk_feature_list_path}")

true_pk_feature_list_path = OUTPUT_DIR / "metrics" / "feature_columns_true_pk.txt"
with open(true_pk_feature_list_path, 'w') as f:
    f.write('\n'.join(feature_columns_true_pk))
print(f"✓ Feature list (true_pk) saved: {true_pk_feature_list_path}")

# Train Models 3 & 4 — features + pred_pK
print(f"\n{'#'*60}")
print("# TRAINING PIPELINE: FEATURES + PRED_PK (models 3-4)")
print(f"{'#'*60}")

all_models_pred_pk = {}
all_metrics_pred_pk = []
all_feature_importance_pred_pk = {}
all_predictions_pred_pk = {}

pipeline_start_pred_pk = datetime.now()

for i, target in enumerate(TARGET_COLUMNS, 1):
    print(f"\n\n{'='*60}")
    print(f"[{i}/{len(TARGET_COLUMNS)}] Processing target: {target} (+pred_pk)")
    print(f"{'='*60}")

    y_train = y_dict[target].loc[indices_train]
    y_test  = y_dict[target].loc[indices_test]

    model_tag = f"{target}_pred_pk"

    automl = train_metamodel(X_pred_pk_train, y_train, model_tag)
    all_models_pred_pk[target] = automl

    metrics, y_test_clean, y_pred = evaluate_model(automl, X_pred_pk_test, y_test, model_tag)
    all_metrics_pred_pk.append(metrics)
    all_predictions_pred_pk[target] = {'y_true': y_test_clean, 'y_pred': y_pred}

    fi_df = get_feature_importance(automl, feature_columns_pred_pk)
    if fi_df is not None:
        all_feature_importance_pred_pk[target] = fi_df
        fi_path = OUTPUT_DIR / "feature_importance" / f"{model_tag}_feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importance saved: {fi_path.name}")

    plot_path = OUTPUT_DIR / "plots" / f"{model_tag}_actual_vs_predicted.png"
    plot_actual_vs_predicted(y_test_clean.values, y_pred, model_tag, metrics, plot_path)

    model_path = OUTPUT_DIR / "models" / f"{model_tag}_flaml_model.pkl"
    with open(model_path, 'wb') as f:
        joblib.dump(automl, f)
    print(f"  Model saved: {model_path.name}")

pipeline_end_pred_pk = datetime.now()
duration_pred_pk = (pipeline_end_pred_pk - pipeline_start_pred_pk).total_seconds()
print(f"\n# Pipeline (pred_pk) complete: {duration_pred_pk:.1f}s ({duration_pred_pk/60:.1f} min)")

# Train Models 5 & 6 — features + true_pK
print(f"\n{'#'*60}")
print("# TRAINING PIPELINE: FEATURES + TRUE_PK (models 5-6)")
print(f"{'#'*60}")

all_models_true_pk = {}
all_metrics_true_pk = []
all_feature_importance_true_pk = {}
all_predictions_true_pk = {}

pipeline_start_true_pk = datetime.now()

for i, target in enumerate(TARGET_COLUMNS, 1):
    print(f"\n\n{'='*60}")
    print(f"[{i}/{len(TARGET_COLUMNS)}] Processing target: {target} (+true_pk)")
    print(f"{'='*60}")

    y_train = y_dict[target].loc[indices_train]
    y_test  = y_dict[target].loc[indices_test]

    model_tag = f"{target}_true_pk"

    automl = train_metamodel(X_true_pk_train, y_train, model_tag)
    all_models_true_pk[target] = automl

    metrics, y_test_clean, y_pred = evaluate_model(automl, X_true_pk_test, y_test, model_tag)
    all_metrics_true_pk.append(metrics)
    all_predictions_true_pk[target] = {'y_true': y_test_clean, 'y_pred': y_pred}

    fi_df = get_feature_importance(automl, feature_columns_true_pk)
    if fi_df is not None:
        all_feature_importance_true_pk[target] = fi_df
        fi_path = OUTPUT_DIR / "feature_importance" / f"{model_tag}_feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importance saved: {fi_path.name}")

    plot_path = OUTPUT_DIR / "plots" / f"{model_tag}_actual_vs_predicted.png"
    plot_actual_vs_predicted(y_test_clean.values, y_pred, model_tag, metrics, plot_path)

    model_path = OUTPUT_DIR / "models" / f"{model_tag}_flaml_model.pkl"
    with open(model_path, 'wb') as f:
        joblib.dump(automl, f)
    print(f"  Model saved: {model_path.name}")

pipeline_end_true_pk = datetime.now()
duration_true_pk = (pipeline_end_true_pk - pipeline_start_true_pk).total_seconds()
print(f"\n# Pipeline (true_pk) complete: {duration_true_pk:.1f}s ({duration_true_pk/60:.1f} min)")

# Combined Summary — All 6 Models
print(f"\n{'='*60}")
print("SUMMARY: ALL 6 MODELS")
print(f"{'='*60}\n")

all_metrics_combined = (
    [dict(m, variant="features_only") for m in all_metrics] +
    [dict(m, variant="features+pred_pk") for m in all_metrics_pred_pk] +
    [dict(m, variant="features+true_pk") for m in all_metrics_true_pk]
)

summary_all_df = pd.DataFrame(all_metrics_combined)
summary_all_df = summary_all_df[['variant', 'target', 'best_estimator', 'r2', 'rmse', 'mae', 'best_cv_rmse', 'n_test']]
print(summary_all_df.to_string(index=False))

summary_all_path = OUTPUT_DIR / "metrics" / "all_models_summary.csv"
summary_all_df.to_csv(summary_all_path, index=False)
print(f"\n✓ Combined summary saved: {summary_all_path}")

# Combined Actual vs Predicted — All 6 Models 
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

variants = [
    ("features_only",    all_predictions,         all_metrics,         "steelblue"),
    ("features+pred_pk", all_predictions_pred_pk,  all_metrics_pred_pk,  "darkorange"),
    ("features+true_pk", all_predictions_true_pk,  all_metrics_true_pk,  "seagreen"),
]

for row, (variant_label, preds_dict, metrics_list, color) in enumerate(variants):
    for col, target in enumerate(TARGET_COLUMNS):
        ax = axes[row][col]

        y_true = preds_dict[target]['y_true']
        y_pred = preds_dict[target]['y_pred']
        m = metrics_list[col]

        ax.scatter(y_true, y_pred, alpha=0.3, s=8, c=color)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=1.5)

        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{variant_label}\n{target}', fontsize=10, fontweight='bold')

        textstr = f"R²={m['r2']:.3f}  RMSE={m['rmse']:.3f}\n{m['best_estimator']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_aspect('equal')

plt.suptitle('All 6 Metamodels — Actual vs Predicted', fontsize=13, fontweight='bold')
plt.tight_layout()
all6_plot_path = OUTPUT_DIR / "plots" / "all_6_models_comparison.png"
plt.savefig(all6_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ All-6 plot saved: {all6_plot_path}")

# Feature Importance — All 6 Models (3×2 grid)
fig, axes = plt.subplots(3, 2, figsize=(16, 22))

fi_variants = [
    ("features_only",    all_feature_importance),
    ("features+pred_pk", all_feature_importance_pred_pk),
    ("features+true_pk", all_feature_importance_true_pk),
]

for row, (variant_label, fi_dict) in enumerate(fi_variants):
    for col, target in enumerate(TARGET_COLUMNS):
        ax = axes[row][col]
        if target in fi_dict:
            fi_df = fi_dict[target].head(20)
            ax.barh(range(len(fi_df)), fi_df['importance_normalized'].values, color='steelblue')
            ax.set_yticks(range(len(fi_df)))
            ax.set_yticklabels(fi_df['feature'].values, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel('Normalized Importance', fontsize=9)
            ax.set_title(f'{variant_label}\n{target} — Top 20', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=11)
            ax.set_title(f'{variant_label}\n{target}', fontsize=9)

plt.suptitle('Feature Importance — All 6 Metamodels', fontsize=13, fontweight='bold')
plt.tight_layout()
fi_all_path = OUTPUT_DIR / "plots" / "all_6_feature_importance.png"
plt.savefig(fi_all_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ All-6 feature importance plot saved: {fi_all_path}")

# SHAP Analysis — All 6 Models (path-dependent + interventional)
"""
Path-dependent (default TreeExplainer):
  - Exact for trees, fast
  - Can over-attribute to correlated features (correlations already reduced in preprocessing)

Interventional (feature_perturbation="interventional"):
  - Marginalises out feature correlations using a background sample
  - More theoretically correct when features are correlated
  - Slower (~background_size × sample_size evaluations)

Both are computed here. Compare the two to see where correlations still inflate importance.
"""
import shap

SHAP_SAMPLE_SIZE     = 1000   # rows for SHAP computation
SHAP_BACKGROUND_SIZE = 100    # background rows for interventional SHAP

(OUTPUT_DIR / "shap").mkdir(exist_ok=True)

# Fixed sample indices for reproducibility
rng = np.random.default_rng(RANDOM_STATE)
shap_idx = rng.choice(len(X_train), size=min(SHAP_SAMPLE_SIZE, len(X_train)), replace=False)
bg_idx   = rng.choice(len(X_train), size=min(SHAP_BACKGROUND_SIZE, len(X_train)), replace=False)

X_shap_base    = X_train.iloc[shap_idx].to_numpy()
X_shap_pred_pk = X_pred_pk_train.iloc[shap_idx].to_numpy()
X_shap_true_pk = X_true_pk_train.iloc[shap_idx].to_numpy()

X_bg_base    = X_train.iloc[bg_idx].to_numpy()
X_bg_pred_pk = X_pred_pk_train.iloc[bg_idx].to_numpy()
X_bg_true_pk = X_true_pk_train.iloc[bg_idx].to_numpy()

shap_variants = [
    ("features_only",    all_models,          X_shap_base,    X_bg_base,    feature_columns),
    ("features+pred_pk", all_models_pred_pk,  X_shap_pred_pk, X_bg_pred_pk, feature_columns_pred_pk),
    ("features+true_pk", all_models_true_pk,  X_shap_true_pk, X_bg_true_pk, feature_columns_true_pk),
]

def _compute_shap(estimator, X_shap, X_bg, mode="path_dependent"):
    """Return shap values array for the given mode."""
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

def _beeswarm(sv, X_shap, feat_names, title, save_path, max_display=5):
    """Save a beeswarm (summary_plot dot) to file."""
    ax = plt.subplots(figsize=(10, 8))[1]
    plt.sca(ax)
    shap.summary_plot(sv, X_shap, feature_names=feat_names,
                      max_display=max_display, plot_type='dot', show=False)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

for variant_label, models_dict, X_shap, X_bg, feat_names in shap_variants:
    for target in TARGET_COLUMNS:
        if target not in models_dict:
            print(f"  SKIP {variant_label} | {target}: model not available")
            continue

        model_tag = (
            target if variant_label == "features_only"
            else f"{target}_{variant_label.split('+')[1]}"
        )
        print(f"\nSHAP: {variant_label} | {target}")
        estimator = models_dict[target].model.estimator

        try:
            # --- Path-dependent ---
            sv_pd = _compute_shap(estimator, X_shap, X_bg, mode="path_dependent")
            _beeswarm(sv_pd, X_shap, feat_names,
                      f"SHAP (path-dependent) — {variant_label}\n{target}",
                      OUTPUT_DIR / "shap" / f"{model_tag}_beeswarm_pathdep.png")
            print(f"  ✓ Path-dependent beeswarm saved")

            # --- Interventional ---
            sv_iv = _compute_shap(estimator, X_shap, X_bg, mode="interventional")
            _beeswarm(sv_iv, X_shap, feat_names,
                      f"SHAP (interventional) — {variant_label}\n{target}",
                      OUTPUT_DIR / "shap" / f"{model_tag}_beeswarm_interventional.png")
            print(f"  ✓ Interventional beeswarm saved")

            # --- Save mean |SHAP| importance (both modes) ---
            imp_df = pd.DataFrame({
                'feature': feat_names,
                'mean_abs_shap_pathdep': np.abs(sv_pd).mean(axis=0),
                'mean_abs_shap_interventional': np.abs(sv_iv).mean(axis=0),
            }).sort_values('mean_abs_shap_interventional', ascending=False)
            imp_df['rank_pathdep']       = imp_df['mean_abs_shap_pathdep'].rank(ascending=False).astype(int)
            imp_df['rank_interventional'] = range(1, len(imp_df) + 1)
            imp_df.to_csv(OUTPUT_DIR / "shap" / f"{model_tag}_shap_importance.csv", index=False)
            print(f"  ✓ SHAP importance CSV saved")

        except Exception as e:
            print(f"  ERROR computing SHAP for {model_tag}: {e}")

# Combined summary plot — weighted_error 
# shap.summary_plot (plot_type='dot') is the beeswarm equivalent and supports subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for col, (variant_label, models_dict, X_shap, X_bg, feat_names) in enumerate(shap_variants):
    ax = axes[col]
    target = "weighted_error"
    if target not in models_dict:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax.set_title(variant_label)
        continue
    try:
        estimator = models_dict[target].model.estimator
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv = sv[0]
        plt.sca(ax)
        shap.summary_plot(
            sv, X_shap,
            feature_names=feat_names,
            max_display=15,
            plot_type='dot',
            show=False,
        )
        ax.set_title(f"{variant_label}\nweighted_error", fontsize=10, fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=7, wrap=True)
        ax.set_title(variant_label)

plt.suptitle('SHAP Beeswarm — weighted_error across variants', fontsize=12, fontweight='bold')
plt.tight_layout()
combined_beeswarm_path = OUTPUT_DIR / "shap" / "weighted_error_beeswarm_all_variants.png"
plt.savefig(combined_beeswarm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Combined beeswarm saved: {combined_beeswarm_path}")
# %%
