import pandas as pd
import numpy as np

df = pd.read_excel("All - From Intention to Decision - 319 Respondent.xlsx")
print(f"Initial shape: {df.shape}")

col_a1 = df.columns[1]
col_a2 = df.columns[2]
col_a3 = df.columns[3]

print("A1 unique values:", df[col_a1].value_counts(dropna=False).to_dict())
print("A2 unique values:", df[col_a2].value_counts(dropna=False).to_dict())
print("A3 unique values:", df[col_a3].value_counts(dropna=False).to_dict())

mask_screen = (
    df[col_a1].astype(str).str.contains("Ya|18", case=False, na=False) &
    df[col_a2].astype(str).str.contains("Ya|Yes", case=False, na=False) &
    df[col_a3].astype(str).str.contains("Ya|Yes", case=False, na=False)
)
print(f"Lolos screening count: {mask_screen.sum()}")

df_screened = df[mask_screen].copy()

# Likert items are columns 15 to 52 (38 items)
# 1. IQ (15-19: IQ1-IQ5)
# 2. PP (20-23: PP1-PP4)
# 3. PT (24-27: PT1-PT4)
# 4. IR (28-31: IR1-IR4)
# 5. PC (32-35: PC1-PC4)
# 6. DC (36-39: DC1-DC4)
# 7. DCO (40-43: DCO1-DCO4)
# 8. SHR1 (44: SHR1)
# 9. SHL (45-48: SHL1-SHL4)
# 10. SCT (49-52: SCT1-SCT4)
likert_cols = df.columns[15:53].tolist()
print("Likert columns count:", len(likert_cols))

for col in likert_cols:
    if "SHR1" in col:
        # Extract numeric leading digit
        df_screened[col] = df_screened[col].astype(str).str.extract(r'^(\d+)')[0].astype(float)
    else:
        df_screened[col] = pd.to_numeric(df_screened[col], errors='coerce')

# Check listwise deletion for core model (IQ, PP, PT, IR, PC, DC, DCO - 29 items)
core_likert = df.columns[15:44].tolist()
df_core_clean = df_screened.dropna(subset=core_likert).copy()
print(f"Lolos screening & Core Model (N) = {len(df_core_clean)}")

# Check listwise deletion for full model including Halal specific dimensions (SHL & SCT)
df_full_clean = df_screened.dropna(subset=likert_cols).copy()
print(f"Lolos screening & Full Model termasuk Halal (N) = {len(df_full_clean)}")

