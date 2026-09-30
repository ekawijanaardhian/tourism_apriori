import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
import time

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
print(f"Total Transactions (N) = {N}")

# Column mapping
col_c1 = df.columns[8]   # Main GenAI tool
col_c2 = df.columns[9]   # GenAI purposes (multi-select)
col_d1 = df.columns[10]  # OTA used (Ya/Tidak)
col_d2 = df.columns[11]  # OTA purposes (multi-select)
col_d3 = df.columns[12]  # Co-usage pattern
col_d4 = df.columns[13]  # Cross check (Ya/Tidak)
col_d5 = df.columns[14]  # Cross-check items (multi-select)

# Construct score for DCO (Decision Commitment)
dco_cols = [df.columns[40], df.columns[41], df.columns[42], df.columns[43]]
df_clean['DCO_score'] = df_clean[dco_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
median_dco = df_clean['DCO_score'].median()

# Build Transaction List
transactions = []

for idx, row in df_clean.iterrows():
    basket = []
    
    # Main GenAI
    if pd.notna(row[col_c1]):
        val = str(row[col_c1]).strip()
        if 'ChatGPT' in val: basket.append('GenAI_Tool:ChatGPT')
        elif 'Gemini' in val: basket.append('GenAI_Tool:Gemini')
        elif 'Claude' in val: basket.append('GenAI_Tool:Claude')
        
    # GenAI Purposes (C2)
    if pd.notna(row[col_c2]):
        items = [x.strip() for x in str(row[col_c2]).split(',') if x.strip()]
        for it in items:
            # Clean and standardize names
            if 'inspirasi' in it.lower() or 'ide' in it.lower(): basket.append('GenAI:Inspirasi_Ide')
            elif 'informasi destinasi' in it.lower(): basket.append('GenAI:Info_Destinasi')
            elif 'area' in it.lower() or 'attractions' in it.lower(): basket.append('GenAI:Pilih_Atraksi')
            elif 'akomodasi' in it.lower() or 'hotel' in it.lower(): basket.append('GenAI:Cari_Akomodasi')
            elif 'itinerary' in it.lower(): basket.append('GenAI:Susun_Itinerary')
            elif 'transportasi' in it.lower(): basket.append('GenAI:Info_Transportasi')
            elif 'budget' in it.lower() or 'biaya' in it.lower(): basket.append('GenAI:Estimasi_Budget')
            elif 'keputusan akhir' in it.lower(): basket.append('GenAI:Bantu_Keputusan')
            elif 'alternatif' in it.lower(): basket.append('GenAI:Bandingkan_Alternatif')

    # OTA Purposes (D2)
    if pd.notna(row[col_d2]):
        items = [x.strip() for x in str(row[col_d2]).split(',') if x.strip()]
        for it in items:
            if 'harga' in it.lower(): basket.append('OTA:Cek_Harga')
            elif 'availability' in it.lower() or 'ketersediaan' in it.lower(): basket.append('OTA:Cek_Ketersediaan')
            elif 'pemesanan' in it.lower() or 'booking' in it.lower(): basket.append('OTA:Booking_Pemesanan')
            elif 'attractions' in it.lower() or 'atraksi' in it.lower(): basket.append('OTA:Cari_Atraksi')
            elif 'rating' in it.lower() or 'review' in it.lower() or 'ulasan' in it.lower(): basket.append('OTA:Cek_Review')
            elif 'hotel' in it.lower() or 'akomodasi' in it.lower(): basket.append('OTA:Banding_Hotel')
            elif 'transportasi' in it.lower(): basket.append('OTA:Cari_Transportasi')

    # Pattern (D3)
    if pd.notna(row[col_d3]):
        pat = str(row[col_d3]).strip()
        if 'GenAI terlebih dahulu' in pat: basket.append('Pola:GenAI_Lalu_OTA')
        elif 'bergantian' in pat: basket.append('Pola:Simultan_Bergantian')
        elif 'OTA terlebih dahulu' in pat: basket.append('Pola:OTA_Lalu_GenAI')

    # Cross Check Items (D5)
    if pd.notna(row[col_d5]):
        items = [x.strip() for x in str(row[col_d5]).split(',') if x.strip()]
        for it in items:
            if 'harga' in it.lower(): basket.append('Verify:Harga')
            elif 'availability' in it.lower() or 'ketersediaan' in it.lower(): basket.append('Verify:Ketersediaan')
            elif 'hotel' in it.lower() or 'accommodation' in it.lower(): basket.append('Verify:Hotel')
            elif 'lokasi' in it.lower() or 'location' in it.lower(): basket.append('Verify:Lokasi')
            elif 'attractions' in it.lower() or 'opening hours' in it.lower() or 'jam' in it.lower(): basket.append('Verify:Jam_Buka_Atraksi')
            elif 'transportasi' in it.lower(): basket.append('Verify:Transportasi')
            elif 'waktu tempuh' in it.lower(): basket.append('Verify:Waktu_Tempuh')
            elif 'review' in it.lower() or 'ulasan' in it.lower(): basket.append('Verify:Review_Publik')
            elif 'keakuratan' in it.lower(): basket.append('Verify:Akurasi_GenAI')
            elif 'visa' in it.lower() or 'regulasi' in it.lower(): basket.append('Verify:Regulasi_Visa')

    # Decision Commitment High/Low
    if pd.notna(row['DCO_score']):
        if row['DCO_score'] >= median_dco:
            basket.append('Outcome:High_Commitment')
        else:
            basket.append('Outcome:Low_Commitment')

    transactions.append(list(set(basket)))

# Convert to One-Hot Encoded DataFrame
all_unique_items = sorted(list(set(it for b in transactions for it in b)))
print(f"Total Unique Behavioral Items = {len(all_unique_items)}")

one_hot_data = []
for b in transactions:
    row_dict = {it: (it in b) for it in all_unique_items}
    one_hot_data.append(row_dict)

df_tx = pd.DataFrame(one_hot_data)

# Run FP-Growth vs Apriori Benchmarking
print("\n=== 2. ALGORITHMIC BENCHMARK: FP-GROWTH vs APRIORI ===")
t0 = time.time()
frequent_itemsets_fp = fpgrowth(df_tx, min_support=0.20, use_colnames=True)
t_fp = (time.time() - t0) * 1000

t0 = time.time()
frequent_itemsets_ap = apriori(df_tx, min_support=0.20, use_colnames=True)
t_ap = (time.time() - t0) * 1000

print(f"Frequent Itemsets found (min_sup = 0.20): {len(frequent_itemsets_fp)}")
print(f"Execution Time FP-Growth : {t_fp:.2f} ms")
print(f"Execution Time Apriori   : {t_ap:.2f} ms")
print(f"Speedup Factor           : {t_ap / max(t_fp, 0.001):.2f}x")

# Generate Association Rules
rules = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.60)
rules['lift'] = rules['lift'].round(3)
rules['support'] = rules['support'].round(3)
rules['confidence'] = rules['confidence'].round(3)
rules['conviction'] = rules['conviction'].round(3)

# Filter meaningful multi-domain cross rules (Antecedent has GenAI/OTA, Consequent has Verify/Outcome/Pattern)
def is_interesting_rule(row):
    ant = list(row['antecedents'])
    con = list(row['consequents'])
    ant_str = " ".join(ant)
    con_str = " ".join(con)
    # Check cross-domain interactions
    has_genai = any('GenAI' in x for x in ant)
    has_ota = any('OTA' in x for x in ant)
    has_target = any('Verify' in x or 'Outcome' in x or 'Pola' in x for x in con)
    return (has_genai or has_ota) and has_target and len(con) == 1

filtered_rules = rules[rules.apply(is_interesting_rule, axis=1)].sort_values(by='lift', ascending=False)

print(f"\nTotal Association Rules Generated: {len(rules)}")
print(f"Total High-Impact Cross-Platform Rules: {len(filtered_rules)}")

print("\n=== 3. TOP 15 ASSOCIATION RULES (SORTED BY LIFT & CONFIDENCE) ===")
for i, (_, r) in enumerate(filtered_rules.head(15).iterrows(), 1):
    ant = ", ".join(list(r['antecedents']))
    con = ", ".join(list(r['consequents']))
    print(f"Rule #{i:2d}: [{ant}]  ==>  [{con}]")
    print(f"         Support = {r['support']:.3f} | Confidence = {r['confidence']*100:.1f}% | Lift = {r['lift']:.3f} | Conviction = {r['conviction']:.3f}\n")

# Save rules to CSV
output_csv = "association_rules_tourism_genai.csv"
filtered_rules.to_csv(output_csv, index=False)
print(f"Rules exported to: {output_csv}")
