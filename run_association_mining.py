import time
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

df = pd.read_excel("All - From Intention to Decision - 319 Respondent.xlsx")
col_a1, col_a2, col_a3 = df.columns[1], df.columns[2], df.columns[3]
mask_screen = (
    df[col_a1].astype(str).str.contains("Ya|18", case=False, na=False) &
    df[col_a2].astype(str).str.contains("Ya|Yes", case=False, na=False) &
    df[col_a3].astype(str).str.contains("Ya|Yes", case=False, na=False)
)
df_clean = df[mask_screen].copy().reset_index(drop=True)

col_c1 = df.columns[8]
col_c2 = df.columns[9]
col_d1 = df.columns[10]
col_d2 = df.columns[11]
col_d3 = df.columns[12]
col_d4 = df.columns[13]
col_d5 = df.columns[14]

dco_cols = [df.columns[40], df.columns[41], df.columns[42], df.columns[43]]
df_clean['DCO_score'] = df_clean[dco_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
median_dco = df_clean['DCO_score'].median()

transactions = []
for idx, row in df_clean.iterrows():
    basket = []
    if pd.notna(row[col_c1]):
        val = str(row[col_c1]).strip()
        if 'ChatGPT' in val:
            basket.append('GenAI_Tool:ChatGPT')
        elif 'Gemini' in val:
            basket.append('GenAI_Tool:Gemini')
        elif 'Claude' in val:
            basket.append('GenAI_Tool:Claude')
    if pd.notna(row[col_c2]):
        items = [x.strip() for x in str(row[col_c2]).split(',') if x.strip()]
        for it in items:
            it_l = it.lower()
            if 'inspirasi' in it_l or 'ide' in it_l:
                basket.append('GenAI:Inspirasi_Ide')
            elif 'informasi destinasi' in it_l:
                basket.append('GenAI:Info_Destinasi')
            elif 'area' in it_l or 'attractions' in it_l:
                basket.append('GenAI:Pilih_Atraksi')
            elif 'akomodasi' in it_l or 'hotel' in it_l:
                basket.append('GenAI:Cari_Akomodasi')
            elif 'itinerary' in it_l:
                basket.append('GenAI:Susun_Itinerary')
            elif 'transportasi' in it_l:
                basket.append('GenAI:Info_Transportasi')
            elif 'budget' in it_l or 'biaya' in it_l:
                basket.append('GenAI:Estimasi_Budget')
            elif 'keputusan akhir' in it_l:
                basket.append('GenAI:Bantu_Keputusan')
            elif 'alternatif' in it_l:
                basket.append('GenAI:Bandingkan_Alternatif')
    if pd.notna(row[col_d2]):
        items = [x.strip() for x in str(row[col_d2]).split(',') if x.strip()]
        for it in items:
            it_l = it.lower()
            if 'harga' in it_l:
                basket.append('OTA:Cek_Harga')
            elif 'availability' in it_l or 'ketersediaan' in it_l:
                basket.append('OTA:Cek_Ketersediaan')
            elif 'pemesanan' in it_l or 'booking' in it_l:
                basket.append('OTA:Booking_Pemesanan')
            elif 'attractions' in it_l or 'atraksi' in it_l:
                basket.append('OTA:Cari_Atraksi')
            elif 'rating' in it_l or 'review' in it_l or 'ulasan' in it_l:
                basket.append('OTA:Cek_Review')
            elif 'hotel' in it_l or 'akomodasi' in it_l:
                basket.append('OTA:Banding_Hotel')
            elif 'transportasi' in it_l:
                basket.append('OTA:Cari_Transportasi')
    if pd.notna(row[col_d3]):
        pat = str(row[col_d3]).strip()
        if 'GenAI terlebih dahulu' in pat:
            basket.append('Pola:GenAI_Lalu_OTA')
        elif 'bergantian' in pat:
            basket.append('Pola:Simultan_Bergantian')
        elif 'OTA terlebih dahulu' in pat:
            basket.append('Pola:OTA_Lalu_GenAI')
    if pd.notna(row[col_d5]):
        items = [x.strip() for x in str(row[col_d5]).split(',') if x.strip()]
        for it in items:
            it_l = it.lower()
            if 'harga' in it_l:
                basket.append('Verify:Harga')
            elif 'availability' in it_l or 'ketersediaan' in it_l:
                basket.append('Verify:Ketersediaan')
            elif 'hotel' in it_l or 'accommodation' in it_l:
                basket.append('Verify:Hotel')
            elif 'lokasi' in it_l or 'location' in it_l:
                basket.append('Verify:Lokasi')
            elif 'attractions' in it_l or 'opening hours' in it_l or 'jam' in it_l:
                basket.append('Verify:Jam_Buka_Atraksi')
            elif 'transportasi' in it_l:
                basket.append('Verify:Transportasi')
            elif 'waktu tempuh' in it_l:
                basket.append('Verify:Waktu_Tempuh')
            elif 'review' in it_l or 'ulasan' in it_l:
                basket.append('Verify:Review_Publik')
            elif 'keakuratan' in it_l:
                basket.append('Verify:Akurasi_GenAI')
            elif 'visa' in it_l or 'regulasi' in it_l:
                basket.append('Verify:Regulasi_Visa')
    if pd.notna(row['DCO_score']):
        if row['DCO_score'] >= median_dco:
            basket.append('Outcome:High_Commitment')
        else:
            basket.append('Outcome:Low_Commitment')
    transactions.append(list(set(basket)))

all_unique_items = sorted(list(set(it for b in transactions for it in b)))
one_hot_data = [{it: (it in b) for it in all_unique_items} for b in transactions]
df_tx = pd.DataFrame(one_hot_data)

t0 = time.time()
frequent_itemsets_fp = fpgrowth(df_tx, min_support=0.20, use_colnames=True)
t_fp = (time.time() - t0) * 1000

t0 = time.time()
frequent_itemsets_ap = apriori(df_tx, min_support=0.20, use_colnames=True)
t_ap = (time.time() - t0) * 1000

print(f"FP-Growth: {t_fp:.2f} ms | Apriori: {t_ap:.2f} ms | Items: {len(frequent_itemsets_fp)}")

rules = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.60)
rules['lift'] = rules['lift'].round(3)
rules['support'] = rules['support'].round(3)
rules['confidence'] = rules['confidence'].round(3)
rules['conviction'] = rules['conviction'].round(3)

def is_interesting_rule(row):
    ant = list(row['antecedents'])
    con = list(row['consequents'])
    has_genai = any('GenAI' in x for x in ant)
    has_ota = any('OTA' in x for x in ant)
    has_target = any('Verify' in x or 'Outcome' in x or 'Pola' in x for x in con)
    return (has_genai or has_ota) and has_target and len(con) == 1

filtered_rules = rules[rules.apply(is_interesting_rule, axis=1)].sort_values(by='lift', ascending=False)
filtered_rules.to_csv("association_rules_tourism_genai.csv", index=False)

for i, (_, r) in enumerate(filtered_rules.head(15).iterrows(), 1):
    ant = ", ".join(list(r['antecedents']))
    con = ", ".join(list(r['consequents']))
    print(f"[{ant}] => [{con}] | Sup={r['support']:.3f}, Conf={r['confidence']*100:.1f}%, Lift={r['lift']:.3f}")
