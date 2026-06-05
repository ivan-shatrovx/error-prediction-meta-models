
# Metamodel Evaluation on Ross FEP Benchmark — All 6 Models
# Evaluates all 6 trained metamodels on the Ross FEP benchmark (ross_combined.csv):
# ross_combined.csv uses unprefixed column names (e.g. "proximal", "TPSA", "ALA").
# Training used prefixed names (arp_proximal, rdk_TPSA, dpk_ALA). Remapped automatically.


# Configuration
from pathlib import Path

ROSS_CSV   = Path.home() / "Desktop" / "binding_free_energy_benchmark" / "ross_combined.csv"
MODEL_DIR  = Path.home() / "Desktop" / "metamodel_outputs_final"
OUTPUT_DIR = Path.home() / "Desktop" / "metamodel_outputs_final" / "benchmark_evaluation"

TARGET_COLUMNS = ["weighted_error", "weighted_abs_error"]
PREFIXES = {"arp_", "dpk_", "rdk_"}

print(f"Ross CSV:  {ROSS_CSV}")
print(f"Model dir: {MODEL_DIR}")
print(f"Output:    {OUTPUT_DIR}")

# Imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Imports OK")

# Load feature lists and imputer
def load_feature_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

feature_list_base     = load_feature_list(MODEL_DIR / "metrics" / "feature_columns.txt")
feature_list_pred_pk  = load_feature_list(MODEL_DIR / "metrics" / "feature_columns_pred_pk.txt")
feature_list_true_pk  = load_feature_list(MODEL_DIR / "metrics" / "feature_columns_true_pk.txt")

imputer = joblib.load(MODEL_DIR / "models" / "feature_imputer.joblib")

print(f"Base features:     {len(feature_list_base)}")
print(f"+ pred_pk features:{len(feature_list_pred_pk)}")
print(f"+ true_pk features:{len(feature_list_true_pk)}")

# Load ross_combined.csv
print(f"\nLoading {ROSS_CSV} ...")
df = pd.read_csv(ROSS_CSV)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

if "preds" not in df.columns:
    raise ValueError("Expected column 'preds' not found in ross_combined.csv")

# Parse experimental dG from SDF files
BENCHMARK_DIR = Path.home() / "Desktop" / "binding_free_energy_benchmark"
SDF_ROOT      = BENCHMARK_DIR / "fep_benchmark_inputs" / "structure_inputs"
RT_LN10       = 1.3592  # kcal/mol at 297 K  (= R * T * ln10 = 0.5902 * 2.3026)

def parse_sdf_exp_dg(sdf_path, group, target):
    """Yield (unique_id, exp_dG) pairs from a multi-mol SDF with r_exp_dg property."""
    records = []
    with open(sdf_path) as f:
        mol_name, exp_dg = None, None
        for line in f:
            line = line.rstrip()
            if mol_name is None:
                mol_name = line.strip()
            if line == "> <r_exp_dg>":
                next_line = next(f).strip()
                try:
                    exp_dg = float(next_line)
                except ValueError:
                    exp_dg = None
            if line == "$$$$":
                if mol_name and exp_dg is not None:
                    uid = f"{group}/{target}/{mol_name}"
                    records.append((uid, exp_dg))
                mol_name, exp_dg = None, None
    return records

exp_dg_records = []
for group_dir in SDF_ROOT.iterdir():
    if not group_dir.is_dir():
        continue
    group = group_dir.name
    for sdf_file in group_dir.glob("*_ligands.sdf"):
        target = sdf_file.stem.replace("_ligands", "")
        exp_dg_records.extend(parse_sdf_exp_dg(sdf_file, group, target))

exp_dg_df = pd.DataFrame(exp_dg_records, columns=["unique_id", "exp_dG_kcal"])
exp_dg_df["exp_pK"] = -exp_dg_df["exp_dG_kcal"] / RT_LN10

print(f"Parsed {len(exp_dg_df)} experimental values from SDF files")
print(f"  exp_dG: mean={exp_dg_df['exp_dG_kcal'].mean():.2f}  std={exp_dg_df['exp_dG_kcal'].std():.2f} kcal/mol")
print(f"  exp_pK: mean={exp_dg_df['exp_pK'].mean():.3f}  std={exp_dg_df['exp_pK'].std():.3f}")

# Join exp_pK onto main df
df = df.merge(exp_dg_df[["unique_id", "exp_pK"]], on="unique_id", how="left")
n_matched = df["exp_pK"].notna().sum()
print(f"  Matched {n_matched}/{len(df)} rows in ross_combined.csv")

# Compute actual prediction error
df["pred_error"] = df["preds"] - df["exp_pK"]
print(f"  pred_error: mean={df['pred_error'].mean():.3f}  std={df['pred_error'].std():.3f}")

# Build column remap (unprefixed → prefixed)
col_remap = {}
for train_col in feature_list_base:
    for prefix in PREFIXES:
        if train_col.startswith(prefix):
            raw = train_col[len(prefix):]
            if raw in df.columns:
                col_remap[raw] = train_col
            break

df_feat = df.rename(columns=col_remap)

missing = [c for c in feature_list_base if c not in df_feat.columns]
print(f"\nFeature coverage: {len(feature_list_base)-len(missing)}/{len(feature_list_base)} found, {len(missing)} missing (will be imputed)")
for c in missing:
    df_feat[c] = np.nan

# Preprocess base feature matrix (same pipeline as training)
X_raw = df_feat[feature_list_base].to_numpy(dtype=np.float64, na_value=np.nan)
X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
X_raw = np.clip(X_raw, -1e30, 1e30)
X_base = imputer.transform(X_raw)  # use saved imputer, not fit_transform
print(f"Base feature matrix: {X_base.shape} — no NaN ✓")

# Augmented matrices: add pK column after imputation
X_pred_pk = np.hstack([X_base, df["preds"].values.reshape(-1, 1)])
X_true_pk = np.hstack([X_base, df["exp_pK"].values.reshape(-1, 1)])

# Define model variants
VARIANTS = [
    ("features_only",    TARGET_COLUMNS,                          X_base,     feature_list_base,     "steelblue"),
    ("features+pred_pk", [f"{t}_pred_pk" for t in TARGET_COLUMNS], X_pred_pk, feature_list_pred_pk,  "darkorange"),
    ("features+true_pk", [f"{t}_true_pk" for t in TARGET_COLUMNS], X_true_pk, feature_list_true_pk,  "seagreen"),
]

# Run inference for all 6 models
print(f"\n{'='*60}")
print("INFERENCE — ALL 6 MODELS")
print(f"{'='*60}")

all_results = {}  # variant -> {target -> y_pred array}

for variant_label, model_tags, X_variant, feat_list, color in VARIANTS:
    all_results[variant_label] = {}
    for target, model_tag in zip(TARGET_COLUMNS, model_tags):
        model_path = MODEL_DIR / "models" / f"{model_tag}_flaml_model.pkl"
        if not model_path.exists():
            print(f"  SKIP {model_tag}: model not found")
            continue
        automl = joblib.load(model_path)
        y_pred = automl.predict(X_variant)
        all_results[variant_label][target] = y_pred
        print(f"  {model_tag}: mean={y_pred.mean():.3f}  std={y_pred.std():.3f}")

# Evaluate all 6 models
actual_error = df["pred_error"].values
exp_pK_vals  = df["exp_pK"].values
preds_orig   = df["preds"].values

eval_rows = []

print(f"\n{'='*60}")
print("EVALUATION RESULTS")
print(f"{'='*60}")

for variant_label, _, _, _, _ in VARIANTS:
    if variant_label not in all_results:
        continue
    for target in TARGET_COLUMNS:
        if target not in all_results[variant_label]:
            continue
        y_pred = all_results[variant_label][target]

        valid = ~np.isnan(actual_error) & ~np.isnan(y_pred)
        ae = np.abs(actual_error[valid]) if "abs" in target else actual_error[valid]
        pe = y_pred[valid]

        r2   = r2_score(ae, pe)
        rmse = np.sqrt(mean_squared_error(ae, pe))
        mae  = mean_absolute_error(ae, pe)
        r, p = stats.pearsonr(ae, pe)

        # Corrected prediction quality (only for weighted_error — signed)
        corr_r2, corr_rmse = np.nan, np.nan
        if "abs" not in target:
            preds_corr = preds_orig - y_pred
            valid2 = ~np.isnan(exp_pK_vals) & ~np.isnan(preds_corr)
            corr_r2   = r2_score(exp_pK_vals[valid2], preds_corr[valid2])
            corr_rmse = np.sqrt(mean_squared_error(exp_pK_vals[valid2], preds_corr[valid2]))

        row = {
            "variant": variant_label, "target": target,
            "pearson_r": round(r, 4), "r2_pred_vs_error": round(r2, 4),
            "rmse_pred_vs_error": round(rmse, 4), "mae_pred_vs_error": round(mae, 4),
            "corrected_r2": round(corr_r2, 4) if not np.isnan(corr_r2) else "",
            "corrected_rmse": round(corr_rmse, 4) if not np.isnan(corr_rmse) else "",
            "n": int(valid.sum()),
        }
        eval_rows.append(row)

        print(f"\n  {variant_label} | {target}")
        print(f"    Pearson r (pred vs actual error): {r:.4f}  (p={p:.2e})")
        print(f"    R²:   {r2:.4f}   RMSE: {rmse:.4f}   MAE: {mae:.4f}")
        if not np.isnan(corr_r2):
            print(f"    Corrected pK — R²: {corr_r2:.4f}   RMSE: {corr_rmse:.4f}")

eval_df = pd.DataFrame(eval_rows)
eval_path = OUTPUT_DIR / "all_6_models_evaluation.csv"
eval_df.to_csv(eval_path, index=False)
print(f"\n✓ Evaluation table saved: {eval_path}")

# Per-series breakdown (weighted_error only)
if "subset" in df.columns or "target" in df.columns:
    series_col = "target" if "target" in df.columns else "subset"
    series_rows = []

    for variant_label, _, _, _, _ in VARIANTS:
        if variant_label not in all_results:
            continue
        if "weighted_error" not in all_results[variant_label]:
            continue
        y_pred = all_results[variant_label]["weighted_error"]
        df_tmp = df.copy()
        df_tmp["y_pred"] = y_pred

        for series, grp in df_tmp.groupby(series_col):
            ae_s = grp["pred_error"].values
            pe_s = grp["y_pred"].values
            valid_s = ~(np.isnan(ae_s) | np.isnan(pe_s))
            if valid_s.sum() < 3:
                continue
            r_s, _ = stats.pearsonr(ae_s[valid_s], pe_s[valid_s])
            rmse_s = np.sqrt(mean_squared_error(ae_s[valid_s], pe_s[valid_s]))
            series_rows.append({
                "variant": variant_label, "series": series,
                "n": int(valid_s.sum()),
                "pearson_r": round(r_s, 4),
                "rmse": round(rmse_s, 4),
            })

    series_df = pd.DataFrame(series_rows)
    series_path = OUTPUT_DIR / "per_series_all_variants.csv"
    series_df.to_csv(series_path, index=False)
    print(f"✓ Per-series breakdown saved: {series_path}")

# Save predictions
out_cols = [c for c in ["unique_id", "preds", "exp_pK", "pred_error"] if c in df.columns]
pred_out = df[out_cols].copy()
for variant_label, _, _, _, _ in VARIANTS:
    for target in TARGET_COLUMNS:
        if target in all_results.get(variant_label, {}):
            col = f"{variant_label}__{target}"
            pred_out[col] = all_results[variant_label][target]

pred_out_path = OUTPUT_DIR / "ross_all_6_predictions.csv"
pred_out.to_csv(pred_out_path, index=False)
print(f"✓ All predictions saved: {pred_out_path}")

# Plot 1: Predicted vs actual error — all 6 (3×2 grid)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

for row, (variant_label, _, _, _, color) in enumerate(VARIANTS):
    for col, target in enumerate(TARGET_COLUMNS):
        ax = axes[row][col]
        if target not in all_results.get(variant_label, {}):
            ax.text(0.5, 0.5, 'Model not available', ha='center', va='center')
            continue

        y_pred = all_results[variant_label][target]
        valid = ~np.isnan(actual_error) & ~np.isnan(y_pred)
        ae = np.abs(actual_error[valid]) if "abs" in target else actual_error[valid]
        pe = y_pred[valid]
        r, _ = stats.pearsonr(ae, pe)
        r2 = r2_score(ae, pe)

        ax.scatter(ae, pe, alpha=0.2, s=6, c=color)
        lims = [min(ae.min(), pe.min()), max(ae.max(), pe.max())]
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.axhline(0, color='gray', lw=0.7, linestyle=':')
        ax.axvline(0, color='gray', lw=0.7, linestyle=':')
        ax.set_xlabel('Actual error', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.set_title(f'{variant_label}\n{target}', fontsize=9, fontweight='bold')
        ax.text(0.05, 0.95, f"r={r:.3f}  R²={r2:.3f}\nn={valid.sum():,}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Predicted vs Actual Error — All 6 Models\nRoss FEP Benchmark', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "all_6_pred_vs_error.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Plot saved: all_6_pred_vs_error.png")

# Plot 2: Corrected vs uncorrected pK (weighted_error models only, 1×3)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
valid_pk = ~np.isnan(exp_pK_vals)

for col, (variant_label, _, _, _, color) in enumerate(VARIANTS):
    ax = axes[col]
    if "weighted_error" not in all_results.get(variant_label, {}):
        continue
    y_pred_err = all_results[variant_label]["weighted_error"]
    preds_corr = preds_orig - y_pred_err
    valid2 = valid_pk & ~np.isnan(preds_corr)

    r2_orig = r2_score(exp_pK_vals[valid2], preds_orig[valid2])
    r2_corr = r2_score(exp_pK_vals[valid2], preds_corr[valid2])

    ax.scatter(exp_pK_vals[valid2], preds_orig[valid2], alpha=0.2, s=5, c='steelblue', label=f'Uncorrected R²={r2_orig:.3f}')
    ax.scatter(exp_pK_vals[valid2], preds_corr[valid2], alpha=0.2, s=5, c=color, label=f'Corrected R²={r2_corr:.3f}')
    lims = [exp_pK_vals[valid2].min(), exp_pK_vals[valid2].max()]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('exp_pK (ground truth)', fontsize=10)
    ax.set_ylabel('Predicted pK', fontsize=10)
    ax.set_title(variant_label, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

plt.suptitle('Corrected vs Uncorrected pK Predictions\nRoss FEP Benchmark', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "corrected_vs_uncorrected_pK.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Plot saved: corrected_vs_uncorrected_pK.png")

print(f"\n✓ Done. All results in: {OUTPUT_DIR}")

