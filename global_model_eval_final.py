
# Global pK Model Evaluation 
# Evaluates the 3 trained global pK models on their respective test sets.
# Includes per-series breakdown, SHAP beeswarm plots, and corrected pK analysis.


# Configuration
from pathlib import Path

DATA_PATH     = Path.home() / "Desktop" / "merged_descriptors_predictions.csv"
MODEL_DIR     = Path.home() / "Desktop" / "global_model_outputs_final"
OUTPUT_DIR    = Path.home() / "Desktop" / "global_model_outputs_final" / "evaluation"
BENCHMARK_DIR = Path.home() / "AEV-PLIG-modifications" / "benchmarks"

LABEL_COL     = "pred_true_pK"
ID_COL        = "unique_id"

CASF_CSV      = BENCHMARK_DIR / "casf_2016_test.csv"
ZERO_BIAS_CSV = BENCHMARK_DIR / "zero_ligand_bias_test.csv"
OOD_CSV       = BENCHMARK_DIR / "index_oodtest.csv"

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
import shap

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "shap").mkdir(exist_ok=True)
print("Imports OK")

# Load feature list
with open(MODEL_DIR / "metrics" / "feature_columns.txt") as f:
    feature_columns = [line.strip() for line in f if line.strip()]
print(f"Features: {len(feature_columns)}")

# Load data and test sets
print(f"\nLoading data ...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
for c in feature_columns:
    if c not in df.columns:
        df[c] = np.nan

casf_ids      = set(pd.read_csv(CASF_CSV)["key"].astype(str))
zero_ids      = set(pd.read_csv(ZERO_BIAS_CSV)["key"].astype(str))
ood_df        = pd.read_csv(OOD_CSV)
ood_test_ids  = set(ood_df.loc[ood_df["split"] == "test", "PDB_code"].astype(str))

JOBS = [
    {"name": "casf_2016",        "test_ids": casf_ids,     "train_ids": None},
    {"name": "zero_ligand_bias", "test_ids": zero_ids,     "train_ids": None},
    {"name": "ood_test",         "test_ids": ood_test_ids, "train_ids": None},
]

def get_splits(job):
    test_mask  = df[ID_COL].isin(job["test_ids"])
    if job["train_ids"] is not None:
        train_mask = df[ID_COL].isin(job["train_ids"])
    else:
        train_mask = ~test_mask
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)

def preprocess(df_tr, df_te, imputer):
    def _clean(frame):
        X = frame[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.where(np.isinf(X), np.nan, X)
        return np.clip(X, -1e30, 1e30)
    return imputer.transform(_clean(df_tr)), imputer.transform(_clean(df_te))

# Evaluate all models
eval_rows = []
all_results = {}

print(f"\n{'='*60}")
print("EVALUATION RESULTS")
print(f"{'='*60}")

for job in JOBS:
    name = job["name"]
    model_path   = MODEL_DIR / "models" / f"{name}_flaml_model.pkl"
    imputer_path = MODEL_DIR / "models" / f"{name}_imputer.joblib"

    if not model_path.exists():
        print(f"  SKIP {name}: model not found at {model_path}")
        continue

    automl  = joblib.load(model_path)
    imputer = joblib.load(imputer_path)

    df_train, df_test = get_splits(job)
    _, X_test = preprocess(df_train, df_test, imputer)
    y_test    = df_test[LABEL_COL].values
    y_pred    = automl.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r, p = stats.pearsonr(y_test, y_pred)

    print(f"\n  {name}  (n={len(y_test):,})")
    print(f"    R²={r2:.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}   r={r:.4f}")

    eval_rows.append({"model": name, "r2": r2, "rmse": rmse, "mae": mae,
                      "pearson_r": r, "n_test": len(y_test),
                      "best_estimator": automl.best_estimator})

    all_results[name] = {
        "y_true": y_test, "y_pred": y_pred,
        "unique_ids": df_test[ID_COL].values,
        "df_test": df_test,
    }

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)
print(f"\n✓ Summary saved")

# Per-series breakdown (by protein_path if available)
if "protein_path" in df.columns:
    series_rows = []
    for name, res in all_results.items():
        df_te = res["df_test"].copy()
        df_te["y_pred"] = res["y_pred"]
        for series, grp in df_te.groupby("protein_path"):
            if len(grp) < 3:
                continue
            r_s, _ = stats.pearsonr(grp[LABEL_COL].values, grp["y_pred"].values)
            rmse_s = np.sqrt(mean_squared_error(grp[LABEL_COL].values, grp["y_pred"].values))
            series_rows.append({"model": name, "protein": series,
                                 "n": len(grp), "pearson_r": round(r_s, 4),
                                 "rmse": round(rmse_s, 4)})
    series_df = pd.DataFrame(series_rows)
    if not series_df.empty:
        series_df = series_df.sort_values(["model", "pearson_r"], ascending=[True, False])
    series_df.to_csv(OUTPUT_DIR / "per_series_evaluation.csv", index=False)
    print(f"✓ Per-series breakdown saved ({len(series_df)} series)")

# Save full predictions
for name, res in all_results.items():
    pd.DataFrame({
        "unique_id": res["unique_ids"],
        "true_pK":   res["y_true"],
        "pred_pK":   res["y_pred"],
        "error":     res["y_pred"] - res["y_true"],
        "abs_error": np.abs(res["y_pred"] - res["y_true"]),
    }).to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=False)
print(f"✓ Prediction CSVs saved")

# Combined actual vs predicted plots
names = list(all_results.keys())
fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6))
if len(names) == 1:
    axes = [axes]

for ax, name in zip(axes, names):
    res = all_results[name]
    m   = next(r for r in eval_rows if r["model"] == name)
    ax.scatter(res["y_true"], res["y_pred"], alpha=0.3, s=8, c='steelblue')
    lims = [min(res["y_true"].min(), res["y_pred"].min()),
            max(res["y_true"].max(), res["y_pred"].max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('True pK', fontsize=10)
    ax.set_ylabel('Predicted pK', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.text(0.05, 0.95,
            f"R²={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nr={m['pearson_r']:.3f}\nn={m['n_test']:,}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')

plt.suptitle('Global pK Model — Test Set Performance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "all_models_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Combined plot saved")

# Error distribution per model
fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
if len(names) == 1:
    axes = [axes]

for ax, name in zip(axes, names):
    errors = all_results[name]["y_pred"] - all_results[name]["y_true"]
    ax.hist(errors, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', lw=1.5, linestyle='--')
    ax.axvline(errors.mean(), color='orange', lw=1.5, linestyle='--', label=f'mean={errors.mean():.3f}')
    ax.set_xlabel('Prediction error (pK units)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Prediction Error Distribution', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Error distribution plot saved")

# SHAP Analysis plots
SHAP_SAMPLE_SIZE     = 1000
SHAP_BACKGROUND_SIZE = 100
rng = np.random.default_rng(42)

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

def _beeswarm(sv, X_shap, feat_names, title, save_path):
    ax = plt.subplots(figsize=(10, 8))[1]
    plt.sca(ax)
    shap.summary_plot(sv, X_shap, feature_names=feat_names,
                      max_display=20, plot_type='dot', show=False)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

for job in JOBS:
    name = job["name"]
    model_path   = MODEL_DIR / "models" / f"{name}_flaml_model.pkl"
    imputer_path = MODEL_DIR / "models" / f"{name}_imputer.joblib"
    if not model_path.exists():
        continue

    automl    = joblib.load(model_path)
    imputer   = joblib.load(imputer_path)
    estimator = automl.model.estimator

    df_train, _ = get_splits(job)
    X_tr_raw = df_train[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
    X_tr_raw = np.where(np.isinf(X_tr_raw), np.nan, X_tr_raw)
    X_tr_raw = np.clip(X_tr_raw, -1e30, 1e30)
    X_tr_clean = imputer.transform(X_tr_raw)

    shap_idx = rng.choice(len(X_tr_clean), size=min(SHAP_SAMPLE_SIZE, len(X_tr_clean)), replace=False)
    bg_idx   = rng.choice(len(X_tr_clean), size=min(SHAP_BACKGROUND_SIZE, len(X_tr_clean)), replace=False)
    X_shap   = X_tr_clean[shap_idx]
    X_bg     = X_tr_clean[bg_idx]

    print(f"\nSHAP: {name}")
    try:
        sv_pd = _compute_shap(estimator, X_shap, X_bg, mode="path_dependent")
        _beeswarm(sv_pd, X_shap, feature_columns,
                  f"SHAP (path-dependent) — {name}",
                  OUTPUT_DIR / "shap" / f"{name}_beeswarm_pathdep.png")
        print(f"  ✓ Path-dependent beeswarm saved")
    except Exception as e:
        print(f"  ERROR (path-dependent): {e}")
        sv_pd = None

    try:
        sv_iv = _compute_shap(estimator, X_shap, X_bg, mode="interventional")
        _beeswarm(sv_iv, X_shap, feature_columns,
                  f"SHAP (interventional) — {name}",
                  OUTPUT_DIR / "shap" / f"{name}_beeswarm_interventional.png")
        print(f"  ✓ Interventional beeswarm saved")

        if sv_pd is not None:
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

# Combined SHAP beeswarm — path-dependent, all 3 models side by side
fig, axes = plt.subplots(1, len(names), figsize=(10 * len(names), 8))
if len(names) == 1:
    axes = [axes]

for ax, job in zip(axes, JOBS):
    name = job["name"]
    model_path   = MODEL_DIR / "models" / f"{name}_flaml_model.pkl"
    imputer_path = MODEL_DIR / "models" / f"{name}_imputer.joblib"
    if not model_path.exists():
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax.set_title(name)
        continue
    try:
        automl    = joblib.load(model_path)
        imputer   = joblib.load(imputer_path)
        estimator = automl.model.estimator
        df_train, _ = get_splits(job)
        X_tr_raw = df_train[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
        X_tr_raw = np.where(np.isinf(X_tr_raw), np.nan, X_tr_raw)
        X_tr_raw = np.clip(X_tr_raw, -1e30, 1e30)
        X_tr_clean = imputer.transform(X_tr_raw)
        idx = rng.choice(len(X_tr_clean), size=min(SHAP_SAMPLE_SIZE, len(X_tr_clean)), replace=False)
        X_shap = X_tr_clean[idx]
        sv = _compute_shap(estimator, X_shap, X_shap[:100], mode="path_dependent")
        plt.sca(ax)
        shap.summary_plot(sv, X_shap, feature_names=feature_columns,
                          max_display=15, plot_type='dot', show=False)
        ax.set_title(name, fontsize=10, fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:80], ha='center', va='center', fontsize=7)
        ax.set_title(name)

plt.suptitle('SHAP (path-dependent) — All 3 Global Models', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap" / "all_models_beeswarm_pathdep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Combined SHAP plot saved")
print(f"\n✓ All done. Results in: {OUTPUT_DIR}")

# Evaluate CASF-2016 global model on Ross FEP benchmark
ROSS_CSV  = Path.home() / "Desktop" / "binding_free_energy_benchmark" / "ross_combined.csv"
SDF_ROOT  = Path.home() / "Desktop" / "binding_free_energy_benchmark" / "fep_benchmark_inputs" / "structure_inputs"
RT_LN10   = 1.3592  # kcal/mol at 297 K

def parse_sdf_exp_dg(sdf_path, group, target):
    records = []
    with open(sdf_path) as f:
        mol_name, exp_dg = None, None
        for line in f:
            line = line.rstrip()
            if mol_name is None:
                mol_name = line.strip()
            if line == "> <r_exp_dg>":
                try:
                    exp_dg = float(next(f).strip())
                except ValueError:
                    exp_dg = None
            if line == "$$$$":
                if mol_name and exp_dg is not None:
                    records.append((f"{group}/{target}/{mol_name}", exp_dg))
                mol_name, exp_dg = None, None
    return records

exp_dg_records = []
for group_dir in SDF_ROOT.iterdir():
    if not group_dir.is_dir():
        continue
    group = group_dir.name
    for sdf_file in group_dir.glob("*_ligands.sdf"):
        target_name = sdf_file.stem.replace("_ligands", "")
        exp_dg_records.extend(parse_sdf_exp_dg(sdf_file, group, target_name))

exp_dg_df = pd.DataFrame(exp_dg_records, columns=["unique_id", "exp_dG_kcal"])
exp_dg_df["exp_pK"] = -exp_dg_df["exp_dG_kcal"] / RT_LN10
print(f"Parsed {len(exp_dg_df)} experimental values from SDF files")

# load in Ross FEP benchmark
ross_df = pd.read_csv(ROSS_CSV)
ross_df = ross_df.merge(exp_dg_df[["unique_id", "exp_pK"]], on="unique_id", how="left")
print(f"Ross CSV: {len(ross_df)} rows, {ross_df['exp_pK'].notna().sum()} matched to exp_pK")

PREFIXES = {"arp_", "dpk_", "rdk_"}
col_remap = {}
for fc in feature_columns:
    for prefix in PREFIXES:
        if fc.startswith(prefix):
            raw = fc[len(prefix):]
            if raw in ross_df.columns:
                col_remap[raw] = fc
            break

ross_feat = ross_df.rename(columns=col_remap)
for c in feature_columns:
    if c not in ross_feat.columns:
        ross_feat[c] = np.nan

X_ross_raw = ross_feat[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
X_ross_raw = np.where(np.isinf(X_ross_raw), np.nan, X_ross_raw)
X_ross_raw = np.clip(X_ross_raw, -1e30, 1e30)

casf_model_path   = MODEL_DIR / "models" / "casf_2016_flaml_model.pkl"
casf_imputer_path = MODEL_DIR / "models" / "casf_2016_imputer.joblib"

casf_automl  = joblib.load(casf_model_path)
casf_imputer = joblib.load(casf_imputer_path)

X_ross = casf_imputer.transform(X_ross_raw)
y_pred_ross = casf_automl.predict(X_ross)

exp_pK_ross = ross_df["exp_pK"].values
valid = ~np.isnan(exp_pK_ross)

r2_r   = r2_score(exp_pK_ross[valid], y_pred_ross[valid])
rmse_r = np.sqrt(mean_squared_error(exp_pK_ross[valid], y_pred_ross[valid]))
mae_r  = mean_absolute_error(exp_pK_ross[valid], y_pred_ross[valid])
r_r, p_r = stats.pearsonr(exp_pK_ross[valid], y_pred_ross[valid])

print(f"\nCASF-2016 model — Ross FEP benchmark  (n={valid.sum():,})")
print(f"  R²={r2_r:.4f}   RMSE={rmse_r:.4f}   MAE={mae_r:.4f}   r={r_r:.4f}  (p={p_r:.2e})")

# unique_id format: {group}/{target}/{ligand} → series = "group/target"
ross_df["pred_pK"]  = y_pred_ross
ross_df["series"]   = ross_df["unique_id"].str.split("/").str[:2].str.join("/")

per_target_rows = []
for series, grp in ross_df.groupby("series"):
    grp_valid = grp.dropna(subset=["exp_pK", "pred_pK"])
    if len(grp_valid) < 3:
        continue
    r_t, _ = stats.pearsonr(grp_valid["exp_pK"].values, grp_valid["pred_pK"].values)
    per_target_rows.append({"series": series, "n": len(grp_valid), "pcc": r_t})

per_target_df = pd.DataFrame(per_target_rows).sort_values("pcc", ascending=False)

def weighted_mean_pcc(df, min_n):
    sub = df[df["n"] >= min_n]
    if sub.empty:
        return np.nan, 0
    return (sub["pcc"] * sub["n"]).sum() / sub["n"].sum(), len(sub)

mean_pcc        = per_target_df["pcc"].mean()
weighted_pcc,  n3  = weighted_mean_pcc(per_target_df, 3)
weighted_pcc5, n5  = weighted_mean_pcc(per_target_df, 5)
weighted_pcc10, n10 = weighted_mean_pcc(per_target_df, 10)

print(f"\n  Per-target PCC  (all series n≥3: {n3} targets)")
print(f"    Mean PCC:              {mean_pcc:.4f}")
print(f"    Weighted PCC (n≥3,  {n3:2d} targets): {weighted_pcc:.4f}")
print(f"    Weighted PCC (n≥5,  {n5:2d} targets): {weighted_pcc5:.4f}")
print(f"    Weighted PCC (n≥10, {n10:2d} targets): {weighted_pcc10:.4f}")
print()
print(per_target_df.to_string(index=False))

per_target_df.to_csv(OUTPUT_DIR / "casf2016_ross_per_target_pcc.csv", index=False)
print(f"\n✓ Per-target PCC saved")

# Save predictions
pd.DataFrame({
    "unique_id": ross_df["unique_id"].values,
    "series":    ross_df["series"].values,
    "exp_pK":    exp_pK_ross,
    "pred_pK":   y_pred_ross,
    "error":     y_pred_ross - exp_pK_ross,
    "abs_error": np.abs(y_pred_ross - exp_pK_ross),
}).to_csv(OUTPUT_DIR / "casf2016_ross_predictions.csv", index=False)
print(f"✓ Predictions saved")

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(exp_pK_ross[valid], y_pred_ross[valid], alpha=0.3, s=8, c='mediumpurple')
lims = [exp_pK_ross[valid].min(), exp_pK_ross[valid].max()]
ax.plot(lims, lims, 'r--', lw=1.5)
ax.set_xlabel('Experimental pK', fontsize=11)
ax.set_ylabel('Predicted pK', fontsize=11)
ax.set_title('CASF-2016 Global Model — Ross FEP Benchmark', fontsize=11, fontweight='bold')
ax.text(0.05, 0.95,
        f"R²={r2_r:.3f}\nRMSE={rmse_r:.3f}\nr={r_r:.3f}\n"
        f"Wt.PCC(n≥3)={weighted_pcc:.3f}\n"
        f"Wt.PCC(n≥5)={weighted_pcc5:.3f}\n"
        f"Wt.PCC(n≥10)={weighted_pcc10:.3f}\nn={valid.sum():,}",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "casf2016_ross_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Scatter plot saved")

# Per-target PCC bar chart
fig, ax = plt.subplots(figsize=(max(8, len(per_target_df) * 0.4), 5))
colors = ['steelblue' if v >= 0 else 'tomato' for v in per_target_df["pcc"]]
ax.bar(per_target_df["series"], per_target_df["pcc"], color=colors, edgecolor='white')
ax.axhline(mean_pcc,     color='orange', lw=1.5, linestyle='--', label=f'Mean={mean_pcc:.3f}')
ax.axhline(weighted_pcc, color='green',  lw=1.5, linestyle='--', label=f'Wt. mean={weighted_pcc:.3f}')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Pearson r', fontsize=10)
ax.set_title('Per-target PCC — CASF-2016 Global Model on Ross FEP', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=90, labelsize=7)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "casf2016_ross_per_target_pcc.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Per-target PCC bar chart saved")

# Global model vs AEV-PLIG — PCC comparison across 4 benchmarks
import plotly.graph_objects as go

# Global model PCC on each test set (from eval_df computed earlier in this script)
global_pcc = {row["model"]: row["pearson_r"] for _, row in eval_df.iterrows()}

benchmarks = ["CASF-2016", "Zero Ligand Bias", "OOD Test", "FEP Benchmark"]

# Global model: per-test-set PCC from eval_df; FEP uses weighted mean PCC (n≥10)
global_vals = [
    global_pcc.get("casf_2016",        float("nan")),
    global_pcc.get("zero_ligand_bias",  float("nan")),
    global_pcc.get("ood_test",          float("nan")),
    weighted_pcc10,
]

# AEV-PLIG: measured from prediction CSVs; FEP value provided
aevplig_vals = [0.8686, 0.3402, 0.6671, 0.62]

fig = go.Figure()

fig.add_trace(go.Bar(
    name="Global Model",
    x=benchmarks,
    y=global_vals,
    marker_color="#50C878",
    text=[f"{v:.3f}" for v in global_vals],
    textposition="outside",
    textfont=dict(color="black"),
))

fig.add_trace(go.Bar(
    name="AEV-PLIG with Augmented Data",
    x=benchmarks,
    y=aevplig_vals,
    marker_color="#FF6B6B",
    text=[f"{v:.3f}" for v in aevplig_vals],
    textposition="outside",
    textfont=dict(color="black"),
))

fig.update_layout(
    title=dict(
        text="Global Model Evaluation",
        x=0.5,
        xanchor="center",
        font=dict(size=16, color="black"),
    ),
    yaxis=dict(title="Pearson Correlation Coefficient (PCC)", range=[0, 1.05]),
    xaxis=dict(title=""),
    barmode="group",
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgrey", borderwidth=1),
    template="plotly_white",
    width=800,
    height=500,
)

out_html = OUTPUT_DIR / "global_vs_aevplig_pcc.html"
out_pdf  = OUTPUT_DIR / "global_vs_aevplig_pcc.pdf"
fig.write_html(str(out_html))
fig.write_image(str(out_pdf))
fig.show(renderer="browser")
print(f"✓ Comparison chart saved: {out_html.name}  +  {out_pdf.name}")
