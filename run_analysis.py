import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_excel("All - From Intention to Decision - 319 Respondent.xlsx")
col_a1, col_a2, col_a3 = df.columns[1], df.columns[2], df.columns[3]
mask_screen = (
    df[col_a1].astype(str).str.contains("Ya|18", case=False, na=False) &
    df[col_a2].astype(str).str.contains("Ya|Yes", case=False, na=False) &
    df[col_a3].astype(str).str.contains("Ya|Yes", case=False, na=False)
)
df_screened = df[mask_screen].copy()

demog_cols = {
    'Gender': df.columns[53],
    'Age': df.columns[54],
    'Education': df.columns[55],
    'Occupation': df.columns[56],
    'TravelFreq': df.columns[57],
    'GenAI_Used': df.columns[8],
    'OTA_Used': df.columns[10],
    'Pattern': df.columns[12],
    'CrossCheck': df.columns[13]
}

for k, col in demog_cols.items():
    counts = df_screened[col].value_counts(dropna=False)
    pcts = df_screened[col].value_counts(normalize=True, dropna=False) * 100
    for idx in counts.index:
        print(f"{k} - {idx}: {counts[idx]} ({pcts[idx]:.1f}%)")

constructs = {
    'IQ': [df.columns[15], df.columns[16], df.columns[17], df.columns[18], df.columns[19]],
    'PP': [df.columns[20], df.columns[21], df.columns[22], df.columns[23]],
    'PT': [df.columns[24], df.columns[25], df.columns[26], df.columns[27]],
    'IR': [df.columns[28], df.columns[29], df.columns[30], df.columns[31]],
    'PC': [df.columns[32], df.columns[33], df.columns[34], df.columns[35]],
    'DC': [df.columns[36], df.columns[37], df.columns[38], df.columns[39]],
    'DCO': [df.columns[40], df.columns[41], df.columns[42], df.columns[43]],
    'SHL': [df.columns[45], df.columns[46], df.columns[47], df.columns[48]],
    'SCT': [df.columns[49], df.columns[50], df.columns[51], df.columns[52]],
}

likert_all = df.columns[15:53].tolist()
for col in likert_all:
    if "SHR1" in col:
        df_screened[col] = df_screened[col].astype(str).str.extract(r'^(\d+)')[0].astype(float)
    else:
        df_screened[col] = pd.to_numeric(df_screened[col], errors='coerce')

for cname, items in constructs.items():
    df_screened[f"{cname}_score"] = df_screened[items].mean(axis=1)

def cronbach_alpha(data):
    d = data.dropna()
    k = d.shape[1]
    if k <= 1:
        return 0.0
    item_vars = d.var(axis=0, ddof=1)
    total_var = d.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def calc_rel_val(data):
    d = data.dropna()
    corr = d.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    loadings = eigvecs[:, -1] * np.sqrt(eigvals[-1])
    if np.mean(loadings) < 0:
        loadings = -loadings
    loadings_sq = loadings ** 2
    ave = np.mean(loadings_sq)
    cr = (np.sum(loadings) ** 2) / ((np.sum(loadings) ** 2) + np.sum(1 - loadings_sq))
    alpha = cronbach_alpha(d)
    return alpha, cr, ave, loadings

res_mm = []
for cname, items in constructs.items():
    d = df_screened[items]
    alpha, cr, ave, loadings = calc_rel_val(d)
    m_score = df_screened[f"{cname}_score"].dropna().mean()
    sd_score = df_screened[f"{cname}_score"].dropna().std()
    res_mm.append({
        'Construct': cname,
        'Items': len(items),
        'N': len(d.dropna()),
        'Mean': m_score,
        'SD': sd_score,
        'Alpha': alpha,
        'CR': cr,
        'AVE': ave,
        'Sqrt_AVE': np.sqrt(ave),
        'Loadings': [round(float(x), 3) for x in loadings]
    })
    print(f"{cname:5s} | N={len(d.dropna()):3d} | Mean={m_score:.2f} | SD={sd_score:.2f} | Alpha={alpha:.3f} | CR={cr:.3f} | AVE={ave:.3f}")

core_cnames = ['IQ', 'PP', 'PT', 'IR', 'PC', 'DC', 'DCO']
scores_df = df_screened[[f"{c}_score" for c in core_cnames]].dropna()
corr_matrix = scores_df.corr()
for i, c1 in enumerate(core_cnames):
    row_str = f"{c1}\t"
    for j, c2 in enumerate(core_cnames):
        if i == j:
            sqrt_ave = [x['Sqrt_AVE'] for x in res_mm if x['Construct'] == c1][0]
            row_str += f"[{sqrt_ave:.3f}]\t"
        elif j < i:
            r = corr_matrix.iloc[i, j]
            row_str += f"{r:.3f}\t"
        else:
            row_str += "-\t"
    print(row_str)

def calc_htmt(items1, items2, df_in):
    between_corrs = []
    for it1 in items1:
        for it2 in items2:
            r = df_in[[it1, it2]].dropna().corr().iloc[0, 1]
            between_corrs.append(abs(r))
    mean_between = np.mean(between_corrs)
    within1 = [abs(df_in[[items1[i], items1[j]]].dropna().corr().iloc[0, 1]) for i in range(len(items1)) for j in range(i + 1, len(items1))]
    within2 = [abs(df_in[[items2[i], items2[j]]].dropna().corr().iloc[0, 1]) for i in range(len(items2)) for j in range(i + 1, len(items2))]
    return mean_between / np.sqrt(np.mean(within1) * np.mean(within2))

for i, c1 in enumerate(core_cnames):
    row_str = f"{c1}\t"
    for j, c2 in enumerate(core_cnames):
        if j < i:
            h = calc_htmt(constructs[c1], constructs[c2], df_screened)
            row_str += f"{h:.3f}\t"
        else:
            row_str += "-\t"
    print(row_str)

data_model = df_screened[[f"{c}_score" for c in core_cnames]].dropna().copy()
for c in core_cnames:
    data_model[f"{c}_z"] = (data_model[f"{c}_score"] - data_model[f"{c}_score"].mean()) / data_model[f"{c}_score"].std()

X1 = data_model[['IQ_z', 'PP_z', 'IR_z', 'PC_z']]
ols1 = sm.OLS(data_model['PT_z'], sm.add_constant(X1)).fit()
for var in ['IQ_z', 'PP_z', 'IR_z', 'PC_z']:
    print(f"PT ~ {var}: Beta={ols1.params[var]:+.3f}, t={ols1.tvalues[var]:.3f}, p={ols1.pvalues[var]:.4f}")

X2 = data_model[['PT_z', 'PC_z', 'IR_z', 'PP_z', 'IQ_z']]
ols2 = sm.OLS(data_model['DC_z'], sm.add_constant(X2)).fit()
for var in ['PT_z', 'PC_z', 'IR_z', 'PP_z', 'IQ_z']:
    print(f"DC ~ {var}: Beta={ols2.params[var]:+.3f}, t={ols2.tvalues[var]:.3f}, p={ols2.pvalues[var]:.4f}")

X3 = data_model[['DC_z', 'PT_z', 'PC_z']]
ols3 = sm.OLS(data_model['DCO_z'], sm.add_constant(X3)).fit()
for var in ['DC_z', 'PT_z', 'PC_z']:
    print(f"DCO ~ {var}: Beta={ols3.params[var]:+.3f}, t={ols3.tvalues[var]:.3f}, p={ols3.pvalues[var]:.4f}")

np.random.seed(42)
boot_results = {
    'PT -> DC -> DCO': [],
    'IQ -> PT -> DC': [],
    'PP -> PT -> DC': [],
    'PC -> PT -> DC': [],
    'PC -> DC -> DCO': []
}
N = len(data_model)
for _ in range(5000):
    sample = data_model.sample(n=N, replace=True)
    m1 = sm.OLS(sample['PT_z'], sm.add_constant(sample[['IQ_z', 'PP_z', 'IR_z', 'PC_z']])).fit()
    m2 = sm.OLS(sample['DC_z'], sm.add_constant(sample[['PT_z', 'PC_z', 'IR_z', 'PP_z', 'IQ_z']])).fit()
    m3 = sm.OLS(sample['DCO_z'], sm.add_constant(sample[['DC_z', 'PT_z', 'PC_z']])).fit()
    boot_results['PT -> DC -> DCO'].append(m2.params['PT_z'] * m3.params['DC_z'])
    boot_results['IQ -> PT -> DC'].append(m1.params['IQ_z'] * m2.params['PT_z'])
    boot_results['PP -> PT -> DC'].append(m1.params['PP_z'] * m2.params['PT_z'])
    boot_results['PC -> PT -> DC'].append(m1.params['PC_z'] * m2.params['PT_z'])
    boot_results['PC -> DC -> DCO'].append(m2.params['PC_z'] * m3.params['DC_z'])

for path, vals in boot_results.items():
    mean_val = np.mean(vals)
    ci_lower = np.percentile(vals, 2.5)
    ci_upper = np.percentile(vals, 97.5)
    p_val = 2 * min(np.mean(np.array(vals) <= 0), np.mean(np.array(vals) >= 0))
    print(f"{path}: Beta={mean_val:+.3f}, CI=[{ci_lower:+.3f}, {ci_upper:+.3f}], p={p_val:.4f}")
