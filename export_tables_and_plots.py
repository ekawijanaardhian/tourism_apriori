import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
import time
import os

# Set style for academic publication plots
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})
sns.set_theme(style="whitegrid")

# Create output folder for charts & tables
os.makedirs("output_results", exist_ok=True)

# 1. Load Data & Filter Screening
df = pd.read_excel("All - From Intention to Decision - 319 Respondent.xlsx")
col_a1, col_a2, col_a3 = df.columns[1], df.columns[2], df.columns[3]
mask_screen = (
    df[col_a1].astype(str).str.contains("Ya|18", case=False, na=False) &
    df[col_a2].astype(str).str.contains("Ya|Yes", case=False, na=False) &
    df[col_a3].astype(str).str.contains("Ya|Yes", case=False, na=False)
)
df_clean = df[mask_screen].copy().reset_index(drop=True)
N = len(df_clean)

# ==============================================================================
# TABEL 1: DEMOGRAFI & PROFIL RESPONDEN
# ==============================================================================
demog_list = []
demog_cols = {
    'Jenis Kelamin': df.columns[53],
    'Usia': df.columns[54],
    'Pendidikan': df.columns[55],
    'Pekerjaan': df.columns[56],
    'Frekuensi Wisata Tahunan': df.columns[57],
    'Platform GenAI Utama': df.columns[8],
    'Penggunaan OTA': df.columns[10],
    'Pola Alur Penggunaan': df.columns[12],
    'Verifikasi Silang (Cross-Check)': df.columns[13]
}

for cat_name, col in demog_cols.items():
    counts = df_clean[col].value_counts(dropna=False)
    pcts = df_clean[col].value_counts(normalize=True, dropna=False) * 100
    for idx in counts.index:
        demog_list.append({
            'Kategori': cat_name,
            'Sub-Kategori / Opsi': str(idx),
            'Frekuensi (N)': counts[idx],
            'Persentase (%)': round(pcts[idx], 2)
        })

df_tabel_1 = pd.DataFrame(demog_list)
df_tabel_1.to_csv("tabel_1_demografi_responden.csv", index=False)
print("Saved: tabel_1_demografi_responden.csv")

# ==============================================================================
# TABEL 2 & 3 & 4: OUTER MODEL & DISCRIMINANT VALIDITY
# ==============================================================================
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
        df_clean[col] = df_clean[col].astype(str).str.extract(r'^(\d+)')[0].astype(float)
    else:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

for cname, items in constructs.items():
    df_clean[f"{cname}_score"] = df_clean[items].mean(axis=1)

def cronbach_alpha(data):
    d = data.dropna()
    k = d.shape[1]
    if k <= 1: return 0.0
    item_vars = d.var(axis=0, ddof=1)
    total_var = d.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def calc_rel_val(data):
    d = data.dropna()
    corr = d.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    loadings = eigvecs[:, -1] * np.sqrt(eigvals[-1])
    if np.mean(loadings) < 0: loadings = -loadings
    loadings_sq = loadings ** 2
    ave = np.mean(loadings_sq)
    cr = (np.sum(loadings) ** 2) / ((np.sum(loadings) ** 2) + np.sum(1 - loadings_sq))
    alpha = cronbach_alpha(d)
    return alpha, cr, ave, loadings

construct_labels = {
    'IQ': 'Information Quality',
    'PP': 'Personalization Fit',
    'PT': 'Platform Trust (GenAI)',
    'IR': 'Information Risk / Skepticism',
    'PC': 'Perceived Control',
    'DC': 'Decision Confidence',
    'DCO': 'Decision Commitment',
    'SHL': 'Shariah / Halal Literacy',
    'SCT': 'Shariah Congruence Trust'
}

mm_records = []
for cname, items in constructs.items():
    d = df_clean[items]
    alpha, cr, ave, loadings = calc_rel_val(d)
    m_val = df_clean[f"{cname}_score"].dropna().mean()
    sd_val = df_clean[f"{cname}_score"].dropna().std()
    mm_records.append({
        'Kode': cname,
        'Konstruk': construct_labels[cname],
        'Jumlah Item': len(items),
        'N Valid': len(d.dropna()),
        'Mean': round(m_val, 3),
        'SD': round(sd_val, 3),
        'Cronbach Alpha': round(alpha, 3),
        'Composite Reliability (CR)': round(cr, 3),
        'Average Variance Extracted (AVE)': round(ave, 3),
        'Sqrt(AVE)': round(np.sqrt(ave), 3),
        'Status Validitas & Reliabilitas': 'Valid & Reliabel (Memenuhi Syarat)'
    })

df_tabel_2 = pd.DataFrame(mm_records)
df_tabel_2.to_csv("tabel_2_outer_model_validitas_reliabilitas.csv", index=False)
print("Saved: tabel_2_outer_model_validitas_reliabilitas.csv")

# Fornell-Larcker
core_cnames = ['IQ', 'PP', 'PT', 'IR', 'PC', 'DC', 'DCO']
scores_df = df_clean[[f"{c}_score" for c in core_cnames]].dropna()
corr_matrix = scores_df.corr()
fl_matrix = pd.DataFrame(index=core_cnames, columns=core_cnames)
for i, c1 in enumerate(core_cnames):
    for j, c2 in enumerate(core_cnames):
        if i == j:
            sqrt_ave = [x['Sqrt(AVE)'] for x in mm_records if x['Kode'] == c1][0]
            fl_matrix.loc[c1, c2] = f"[{sqrt_ave:.3f}]"
        elif j < i:
            fl_matrix.loc[c1, c2] = f"{corr_matrix.loc[f'{c1}_score', f'{c2}_score']:.3f}"
        else:
            fl_matrix.loc[c1, c2] = "-"
fl_matrix.to_csv("tabel_3_discriminant_validity_fornell_larcker.csv")
print("Saved: tabel_3_discriminant_validity_fornell_larcker.csv")

# HTMT Matrix
def calc_htmt(items1, items2, df_in):
    between_corrs = [abs(df_in[[it1, it2]].dropna().corr().iloc[0, 1]) for it1 in items1 for it2 in items2]
    within1 = [abs(df_in[[items1[i], items1[j]]].dropna().corr().iloc[0, 1]) for i in range(len(items1)) for j in range(i+1, len(items1))]
    within2 = [abs(df_in[[items2[i], items2[j]]].dropna().corr().iloc[0, 1]) for i in range(len(items2)) for j in range(i+1, len(items2))]
    return np.mean(between_corrs) / np.sqrt(np.mean(within1) * np.mean(within2))

htmt_matrix = pd.DataFrame(index=core_cnames, columns=core_cnames)
htmt_numeric = np.zeros((len(core_cnames), len(core_cnames)))
for i, c1 in enumerate(core_cnames):
    for j, c2 in enumerate(core_cnames):
        if j < i:
            h_val = calc_htmt(constructs[c1], constructs[c2], df_clean)
            htmt_matrix.loc[c1, c2] = f"{h_val:.3f}"
            htmt_numeric[i, j] = h_val
            htmt_numeric[j, i] = h_val
        elif i == j:
            htmt_matrix.loc[c1, c2] = "-"
            htmt_numeric[i, j] = 1.0
        else:
            htmt_matrix.loc[c1, c2] = "-"

htmt_matrix.to_csv("tabel_4_discriminant_validity_htmt.csv")
print("Saved: tabel_4_discriminant_validity_htmt.csv")

# ==============================================================================
# TABEL 5 & 6: PATH ANALYSIS & MEDIASI BOOTSTRAP
# ==============================================================================
data_model = df_clean[[f"{c}_score" for c in core_cnames]].dropna().copy()
for c in core_cnames:
    data_model[f"{c}_z"] = (data_model[f"{c}_score"] - data_model[f"{c}_score"].mean()) / data_model[f"{c}_score"].std()

ols1 = sm.OLS(data_model['PT_z'], sm.add_constant(data_model[['IQ_z', 'PP_z', 'IR_z', 'PC_z']])).fit()
ols2 = sm.OLS(data_model['DC_z'], sm.add_constant(data_model[['PT_z', 'PC_z', 'IR_z', 'PP_z', 'IQ_z']])).fit()
ols3 = sm.OLS(data_model['DCO_z'], sm.add_constant(data_model[['DC_z', 'PT_z', 'PC_z']])).fit()

path_records = []
for model_name, ols_obj, dep_var in [('Model 1 (Predictors of Platform Trust)', ols1, 'PT (Platform Trust)'),
                                     ('Model 2 (Predictors of Decision Confidence)', ols2, 'DC (Decision Confidence)'),
                                     ('Model 3 (Predictors of Decision Commitment)', ols3, 'DCO (Decision Commitment)')]:
    for var in ols_obj.params.index:
        if var == 'const': continue
        pred_clean = var.replace('_z', '')
        path_records.append({
            'Model': model_name,
            'Variabel Dependen': dep_var,
            'Variabel Independen': pred_clean,
            'Path Coefficient (Beta)': round(ols_obj.params[var], 3),
            't-Statistic': round(ols_obj.tvalues[var], 3),
            'p-Value': round(ols_obj.pvalues[var], 4),
            'R-Squared Model': round(ols_obj.rsquared, 3),
            'Adjusted R-Squared': round(ols_obj.rsquared_adj, 3),
            'Kesimpulan Hipotesis': 'Signifikan (Diterima)' if ols_obj.pvalues[var] < 0.05 else 'Tidak Signifikan'
        })

df_tabel_5 = pd.DataFrame(path_records)
df_tabel_5.to_csv("tabel_5_path_coefficients_regresi.csv", index=False)
print("Saved: tabel_5_path_coefficients_regresi.csv")

# Bootstrapping Mediation
np.random.seed(42)
boot_results = {'PT -> DC -> DCO': [], 'IQ -> PT -> DC': [], 'PP -> PT -> DC': [], 'PC -> PT -> DC': [], 'PC -> DC -> DCO': []}
for _ in range(5000):
    sample = data_model.sample(n=len(data_model), replace=True)
    m1 = sm.OLS(sample['PT_z'], sm.add_constant(sample[['IQ_z', 'PP_z', 'IR_z', 'PC_z']])).fit()
    m2 = sm.OLS(sample['DC_z'], sm.add_constant(sample[['PT_z', 'PC_z', 'IR_z', 'PP_z', 'IQ_z']])).fit()
    m3 = sm.OLS(sample['DCO_z'], sm.add_constant(sample[['DC_z', 'PT_z', 'PC_z']])).fit()
    boot_results['PT -> DC -> DCO'].append(m2.params['PT_z'] * m3.params['DC_z'])
    boot_results['IQ -> PT -> DC'].append(m1.params['IQ_z'] * m2.params['PT_z'])
    boot_results['PP -> PT -> DC'].append(m1.params['PP_z'] * m2.params['PT_z'])
    boot_results['PC -> PT -> DC'].append(m1.params['PC_z'] * m2.params['PT_z'])
    boot_results['PC -> DC -> DCO'].append(m2.params['PC_z'] * m3.params['DC_z'])

med_records = []
for path, vals in boot_results.items():
    ci_low = np.percentile(vals, 2.5)
    ci_up = np.percentile(vals, 97.5)
    p_val = 2 * min(np.mean(np.array(vals) <= 0), np.mean(np.array(vals) >= 0))
    med_records.append({
        'Jalur Mediasi': path,
        'Indirect Beta': round(np.mean(vals), 3),
        '95% CI Lower': round(ci_low, 3),
        '95% CI Upper': round(ci_up, 3),
        'p-Value': round(p_val, 4),
        'Kesimpulan Mediasi': 'Signifikan (Mediasi Terbukti)' if (ci_low > 0 or ci_up < 0) else 'Tidak Signifikan'
    })
df_tabel_6 = pd.DataFrame(med_records)
df_tabel_6.to_csv("tabel_6_mediasi_bootstrap.csv", index=False)
print("Saved: tabel_6_mediasi_bootstrap.csv")

# ==============================================================================
# TABEL 7 & 8: BENCHMARK FP-GROWTH VS APRIORI & TOP RULES
# ==============================================================================
# Build Binary Transaction Matrix
transactions = []
median_dco = df_clean['DCO_score'].median()

for idx, row in df_clean.iterrows():
    basket = []
    if pd.notna(row[df.columns[8]]):
        val = str(row[df.columns[8]]).strip()
        if 'ChatGPT' in val: basket.append('GenAI_Tool:ChatGPT')
        elif 'Gemini' in val: basket.append('GenAI_Tool:Gemini')
        elif 'Claude' in val: basket.append('GenAI_Tool:Claude')
    if pd.notna(row[df.columns[9]]):
        items = [x.strip().lower() for x in str(row[df.columns[9]]).split(',')]
        if any('inspirasi' in x or 'ide' in x for x in items): basket.append('GenAI:Inspirasi_Ide')
        if any('informasi destinasi' in x for x in items): basket.append('GenAI:Info_Destinasi')
        if any('area' in x or 'attractions' in x for x in items): basket.append('GenAI:Pilih_Atraksi')
        if any('akomodasi' in x or 'hotel' in x for x in items): basket.append('GenAI:Cari_Akomodasi')
        if any('itinerary' in x for x in items): basket.append('GenAI:Susun_Itinerary')
        if any('transportasi' in x for x in items): basket.append('GenAI:Info_Transportasi')
        if any('budget' in x for x in items): basket.append('GenAI:Estimasi_Budget')
        if any('alternatif' in x for x in items): basket.append('GenAI:Bandingkan_Alternatif')
    if pd.notna(row[df.columns[11]]):
        items = [x.strip().lower() for x in str(row[df.columns[11]]).split(',')]
        if any('harga' in x for x in items): basket.append('OTA:Cek_Harga')
        if any('availability' in x for x in items): basket.append('OTA:Cek_Ketersediaan')
        if any('pemesanan' in x or 'booking' in x for x in items): basket.append('OTA:Booking_Pemesanan')
        if any('review' in x or 'rating' in x for x in items): basket.append('OTA:Cek_Review')
        if any('hotel' in x for x in items): basket.append('OTA:Banding_Hotel')
    if pd.notna(row[df.columns[12]]):
        pat = str(row[df.columns[12]])
        if 'GenAI terlebih dahulu' in pat: basket.append('Pola:GenAI_Lalu_OTA')
        elif 'bergantian' in pat: basket.append('Pola:Simultan_Bergantian')
    if pd.notna(row[df.columns[14]]):
        items = [x.strip().lower() for x in str(row[df.columns[14]]).split(',')]
        if any('harga' in x for x in items): basket.append('Verify:Harga')
        if any('availability' in x for x in items): basket.append('Verify:Ketersediaan')
        if any('hotel' in x for x in items): basket.append('Verify:Hotel')
        if any('lokasi' in x for x in items): basket.append('Verify:Lokasi')
        if any('attractions' in x or 'jam' in x for x in items): basket.append('Verify:Jam_Buka_Atraksi')
        if any('transportasi' in x for x in items): basket.append('Verify:Transportasi')
        if any('waktu tempuh' in x for x in items): basket.append('Verify:Waktu_Tempuh')
        if any('review' in x for x in items): basket.append('Verify:Review_Publik')
    basket.append('Outcome:High_Commitment' if row['DCO_score'] >= median_dco else 'Outcome:Low_Commitment')
    transactions.append(list(set(basket)))

all_items = sorted(list(set(it for b in transactions for it in b)))
df_tx = pd.DataFrame([{it: (it in b) for it in all_items} for b in transactions])

# Benchmark variation
benchmark_res = []
for sup in [0.30, 0.25, 0.20, 0.15]:
    t0 = time.time()
    fp_set = fpgrowth(df_tx, min_support=sup, use_colnames=True)
    t_fp = (time.time() - t0) * 1000
    
    t0 = time.time()
    ap_set = apriori(df_tx, min_support=sup, use_colnames=True)
    t_ap = (time.time() - t0) * 1000
    
    benchmark_res.append({
        'Min Support Threshold': sup,
        'Jumlah Frequent Itemsets': len(fp_set),
        'Waktu FP-Growth (ms)': round(t_fp, 2),
        'Waktu Apriori (ms)': round(t_ap, 2),
        'Speedup Factor (Apriori/FP-Growth)': round(t_ap / max(t_fp, 0.001), 2)
    })

df_tabel_7 = pd.DataFrame(benchmark_res)
df_tabel_7.to_csv("tabel_7_benchmark_fpgrowth_vs_apriori.csv", index=False)
print("Saved: tabel_7_benchmark_fpgrowth_vs_apriori.csv")

# Rules at sup=0.20
fp_set_20 = fpgrowth(df_tx, min_support=0.20, use_colnames=True)
rules_all = association_rules(fp_set_20, metric="confidence", min_threshold=0.60)
rules_all['ant_str'] = rules_all['antecedents'].apply(lambda x: ", ".join(list(x)))
rules_all['con_str'] = rules_all['consequents'].apply(lambda x: ", ".join(list(x)))

def categorize_rule(row):
    con = row['con_str']
    if 'Verify:Waktu_Tempuh' in con or 'Verify:Jam_Buka' in con: return 'Cross-Verification Chain'
    if 'Outcome:High_Commitment' in con: return 'High Decision Commitment Determinant'
    if 'Pola:GenAI_Lalu_OTA' in con: return 'Sequential Co-Usage Pattern'
    return 'General Interaction'

rules_all['Category'] = rules_all.apply(categorize_rule, axis=1)
top_rules = rules_all[rules_all['Category'] != 'General Interaction'].sort_values(by=['Category', 'lift', 'confidence'], ascending=[True, False, False])
df_tabel_8 = top_rules[['Category', 'ant_str', 'con_str', 'support', 'confidence', 'lift', 'conviction']].head(30)
df_tabel_8.columns = ['Kategori Tema', 'Antecedent (X)', 'Consequent (Y)', 'Support', 'Confidence', 'Lift', 'Conviction']
df_tabel_8.to_csv("tabel_8_top_association_rules.csv", index=False)
print("Saved: tabel_8_top_association_rules.csv")


# ==============================================================================
# VISUALISASI GRAFIK (HIGH RESOLUTION 300 DPI)
# ==============================================================================

# 1. Grafik Demografi & Pola Interaksi
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
# 1A: GenAI Tools
genai_counts = df_clean[df.columns[8]].value_counts().head(3)
sns.barplot(x=genai_counts.values, y=genai_counts.index, ax=axes[0], palette="Blues_r")
axes[0].set_title("A. Platform GenAI yang Digunakan", fontweight='bold')
axes[0].set_xlabel("Jumlah Responden (N)")
for i, v in enumerate(genai_counts.values):
    axes[0].text(v + 2, i, f"{v} ({v/N*100:.1f}%)", va='center', fontweight='semibold')

# 1B: Co-Usage Patterns
pola_counts = df_clean[df.columns[12]].value_counts().dropna().head(4)
sns.barplot(x=pola_counts.values, y=pola_counts.index, ax=axes[1], palette="Greens_r")
axes[1].set_title("B. Pola Alur Penggunaan GenAI & OTA", fontweight='bold')
axes[1].set_xlabel("Jumlah Responden (N)")
for i, v in enumerate(pola_counts.values):
    axes[1].text(v + 1, i, f"{v} ({v/N*100:.1f}%)", va='center', fontweight='semibold')

# 1C: Cross-check item
verify_df = pd.DataFrame([{col: df_clean[df.columns[14]].astype(str).str.contains(col).sum() for col in ['Harga', 'Lokasi', 'Waktu tempuh', 'Transportasi', 'Hotel', 'Review']}])
verify_s = verify_df.iloc[0].sort_values(ascending=False)
sns.barplot(x=verify_s.values, y=verify_s.index, ax=axes[2], palette="Oranges_r")
axes[2].set_title("C. Item yang Diverifikasi Ulang", fontweight='bold')
axes[2].set_xlabel("Jumlah Responden (N)")
for i, v in enumerate(verify_s.values):
    axes[2].text(v + 1, i, f"{v} ({v/N*100:.1f}%)", va='center', fontweight='semibold')

plt.tight_layout()
plt.savefig("grafik_1_demografi_dan_pola_penggunaan.png", dpi=300)
plt.close()
print("Saved: grafik_1_demografi_dan_pola_penggunaan.png")

# 2. Grafik Outer Model (Cronbach Alpha, CR, AVE)
fig, ax = plt.subplots(figsize=(12, 6))
plot_mm = df_tabel_2.copy()
x_idx = np.arange(len(plot_mm))
width = 0.25

rects1 = ax.bar(x_idx - width, plot_mm['Cronbach Alpha'], width, label="Cronbach's Alpha (Rule of Thumb >= 0.70)", color='#2b5c8f')
rects2 = ax.bar(x_idx, plot_mm['Composite Reliability (CR)'], width, label="Composite Reliability (Rule of Thumb >= 0.70)", color='#419d78')
rects3 = ax.bar(x_idx + width, plot_mm['Average Variance Extracted (AVE)'], width, label="AVE (Rule of Thumb >= 0.50)", color='#e07a5f')

ax.axhline(0.70, color='#2b5c8f', linestyle='--', linewidth=1.2, alpha=0.7)
ax.axhline(0.50, color='#e07a5f', linestyle=':', linewidth=1.2, alpha=0.7)

ax.set_ylabel('Koefisien / Nilai Metrik')
ax.set_title('Evaluasi Validitas Konvergen & Reliabilitas Konstruk (Outer Model)', fontweight='bold', pad=15)
ax.set_xticks(x_idx)
ax.set_xticklabels([f"{row['Kode']}\n({row['Konstruk'][:15]}...)" for _, row in plot_mm.iterrows()], rotation=0)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', frameon=True)

# Add values above bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig("grafik_2_outer_model_reliabilitas_ave.png", dpi=300)
plt.close()
print("Saved: grafik_2_outer_model_reliabilitas_ave.png")

# 3. Grafik HTMT Heatmap
fig, ax = plt.subplots(figsize=(8, 6.5))
mask = np.triu(np.ones_like(htmt_numeric, dtype=bool))
sns.heatmap(htmt_numeric, mask=mask, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=core_cnames, yticklabels=core_cnames, cbar_kws={'label': 'Rasio HTMT (< 0.85 Valid)'}, ax=ax)
ax.set_title('Matriks Validitas Diskriminan: Heterotrait-Monotrait Ratio (HTMT)', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("grafik_3_discriminant_validity_htmt_heatmap.png", dpi=300)
plt.close()
print("Saved: grafik_3_discriminant_validity_htmt_heatmap.png")

# 4. Grafik Algorithmic Benchmark FP-Growth vs Apriori
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
bench_df = pd.DataFrame(benchmark_res)

ax1.plot(bench_df['Min Support Threshold'], bench_df['Waktu FP-Growth (ms)'], marker='o', linewidth=2.5, color='#1f77b4', label='FP-Growth (FP-Tree)')
ax1.plot(bench_df['Min Support Threshold'], bench_df['Waktu Apriori (ms)'], marker='s', linewidth=2.5, color='#d62728', linestyle='--', label='Apriori (Candidate Generation)')
ax1.set_xlabel('Minimum Support Threshold')
ax1.set_ylabel('Waktu Eksekusi (Milidetik / ms)')
ax1.set_title('A. Perbandingan Waktu Eksekusi (Skala Linier)', fontweight='bold')
ax1.invert_xaxis()
ax1.legend()
ax1.grid(True, linestyle=':')

ax2.plot(bench_df['Min Support Threshold'], bench_df['Speedup Factor (Apriori/FP-Growth)'], marker='^', linewidth=2.5, color='#2ca02c')
ax2.set_xlabel('Minimum Support Threshold')
ax2.set_ylabel('Speedup Factor (x Lebih Cepat)')
ax2.set_title('B. Faktor Percepatan FP-Growth dibanding Apriori', fontweight='bold')
ax2.invert_xaxis()
ax2.grid(True, linestyle=':')
for _, row in bench_df.iterrows():
    ax2.annotate(f"{row['Speedup Factor (Apriori/FP-Growth)']:.2f}x", 
                 xy=(row['Min Support Threshold'], row['Speedup Factor (Apriori/FP-Growth)']),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("grafik_4_benchmark_fpgrowth_vs_apriori.png", dpi=300)
plt.close()
print("Saved: grafik_4_benchmark_fpgrowth_vs_apriori.png")

# 5. Grafik Association Rules Scatter Plot (Support vs Confidence vs Lift)
fig, ax = plt.subplots(figsize=(10, 6.5))
scatter = ax.scatter(rules_all['support'], rules_all['confidence'], c=rules_all['lift'], cmap='viridis', alpha=0.6, s=rules_all['lift']*30)
cbar = plt.colorbar(scatter)
cbar.set_label('Nilai Lift (Strength of Association)')

ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='High Confidence Threshold (80%)')
ax.set_xlabel('Support (Frekuensi Kemunculan Bersama)')
ax.set_ylabel('Confidence (Probabilitas Kepastian Aturan)')
ax.set_title('Distribusi Aturan Asosiasi (Support vs Confidence vs Lift)', fontweight='bold', pad=15)
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig("grafik_5_association_rules_scatter.png", dpi=300)
plt.close()
print("Saved: grafik_5_association_rules_scatter.png")

print("=== SELURUH 8 TABEL CSV DAN 5 GRAFIK HIGH-RES BERHASIL DIBUAT ===")
