# Feature Selection
#
# Filters metamodel_input.csv down to the 207 features used for model training.
# Two steps:
#   1. Drop manually specified columns (correlated, redundant, or leaking)
#   2. Drop invalid columns (non-numeric, constant, or extreme values)
#
# Output: metamodel_inputs_final.csv

# %% Configuration
from pathlib import Path

INPUT_PATH  = Path.home() / "Desktop" / "metamodel_input.csv"
OUTPUT_PATH = Path.home() / "Desktop" / "metamodel_inputs_final.csv"

# %% Imports
import pandas as pd

df = pd.read_csv(INPUT_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# %% Step 1 — Drop manually specified columns
COLS_TO_DROP = [
    'pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4',
    'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9',
    'rdk_VSA_EState1', 'rdk_VSA_EState10', 'rdk_VSA_EState2', 'rdk_VSA_EState3',
    'rdk_VSA_EState4', 'rdk_VSA_EState5', 'rdk_VSA_EState6', 'rdk_VSA_EState7',
    'rdk_VSA_EState8', 'rdk_VSA_EState9',
    'rdk_SMR_VSA1', 'rdk_SMR_VSA10', 'rdk_SMR_VSA2', 'rdk_SMR_VSA3', 'rdk_SMR_VSA4',
    'rdk_SMR_VSA5', 'rdk_SMR_VSA6', 'rdk_SMR_VSA7', 'rdk_SMR_VSA8', 'rdk_SMR_VSA9',
    'rdk_PEOE_VSA1', 'rdk_PEOE_VSA10', 'rdk_PEOE_VSA11', 'rdk_PEOE_VSA12',
    'rdk_PEOE_VSA13', 'rdk_PEOE_VSA14', 'rdk_PEOE_VSA2', 'rdk_PEOE_VSA3',
    'rdk_PEOE_VSA4', 'rdk_PEOE_VSA5', 'rdk_PEOE_VSA6', 'rdk_PEOE_VSA7',
    'rdk_PEOE_VSA8', 'rdk_PEOE_VSA9',
    'rdk_MinAbsEStateIndex', 'rdk_MaxAbsEStateIndex',
    'rdk_MaxEStateIndex', 'rdk_MinEStateIndex',
    'rdk_Kappa1', 'rdk_Kappa2', 'rdk_Kappa3',
    'rdk_EState_VSA1', 'rdk_EState_VSA10', 'rdk_EState_VSA11', 'rdk_EState_VSA2',
    'rdk_EState_VSA3', 'rdk_EState_VSA4', 'rdk_EState_VSA5', 'rdk_EState_VSA6',
    'rdk_EState_VSA7', 'rdk_EState_VSA8', 'rdk_EState_VSA9',
    'error', 'abs_error', 'arp_polar', 'dpk_lig_vol',
    'rdk_BertzCT', 'rdk_Chi0', 'rdk_Chi0n', 'rdk_Chi0v',
    'rdk_Chi1', 'rdk_Chi1n', 'rdk_Chi1v', 'rdk_Chi2n', 'rdk_Chi2v',
    'rdk_Chi3n', 'rdk_Chi3v', 'rdk_Chi4n', 'rdk_Chi4v',
    'rdk_ExactMolWt', 'rdk_HeavyAtomCount', 'rdk_HeavyAtomMolWt',
    'rdk_LabuteASA', 'rdk_MolMR', 'rdk_NumRotatableBonds',
    'rdk_NumValenceElectrons', 'rdk_Phi', 'rdk_Asphericity', 'rdk_NPR1',
    'rdk_FpDensityMorgan2', 'rdk_FpDensityMorgan3',
    'rdk_fr_Nhpyrrole', 'rdk_fr_phenol', 'rdk_fr_phenol_noOrthoHbond',
    'rdk_fr_nitro_arom', 'rdk_fr_phos_ester', 'rdk_PMI3',
]

df = df.drop(columns=COLS_TO_DROP, errors="ignore")
print(f"After manual drop: {df.shape[1]} columns")

# %% Step 2 — Drop invalid columns (non-numeric, constant, or extreme values)
def get_invalid_columns(df):
    invalid = {}
    for col in df.columns:
        reasons = []
        if not pd.api.types.is_numeric_dtype(df[col]):
            reasons.append("non-numeric")
        else:
            if (df[col].abs() > 1e10).any():
                reasons.append("values exceed 1e10")
            if df[col].nunique() <= 1:
                reasons.append("constant column")
        if reasons:
            invalid[col] = reasons
    return invalid

invalid_columns = get_invalid_columns(df)
if invalid_columns:
    print(f"Dropping {len(invalid_columns)} invalid columns: {list(invalid_columns.keys())}")
df = df.drop(columns=invalid_columns.keys())

# %% Save
print(f"\nFinal shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
for prefix in ['arp', 'dpk', 'rdk']:
    n = sum(1 for c in df.columns if c.startswith(prefix + '_'))
    print(f"  {prefix}_: {n} columns")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")
# %%
