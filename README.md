# ✈️ Tourism Decision Analytics: Human-GenAI & OTA Interaction

Repositori riset empiris dan analitika data untuk penelitian:
**"From Intention to Decision: Peran Generative AI dan Online Travel Agency dalam Pengambilan Keputusan Perjalanan Wisata Internasional"**

---

## 📁 Struktur Direktori & Berkas

```
├── All - From Intention to Decision - 319 Respondent.xlsx  # Dataset mentah survei eksperimen (N=318)
├── 1. Skenario Lab Experiment rev.docx                     # Dokumen skenario tugas lab eksperimen
│
├── rencana_eksperimen.md                                  # Protokol & rencana lengkap eksperimen
├── draft_paper_association_rules.md                       # Draf naskah paper Data Science (FP-Growth vs Apriori)
│
├── check_screening.py                                     # Pengecekan awal kriteria screening & data Likert
├── run_analysis.py                                        # Pipeline analisis statistik SEM-PLS, Regresi, & Mediasi
├── run_association_mining.py                              # Algoritma Association Rule Mining (FP-Growth vs Apriori)
├── export_tables_and_plots.py                             # Ekspor otomatis 8 tabel CSV & 5 grafik resolusi tinggi (300 DPI)
│
├── tabel_1_demografi_responden.csv                        # Tabel profil demografi responden (N=248)
├── tabel_2_outer_model_validitas_reliabilitas.csv         # Evaluasi Cronbach's Alpha, CR, AVE
├── tabel_3_discriminant_validity_fornell_larcker.csv      # Matriks Fornell-Larcker
├── tabel_4_discriminant_validity_htmt.csv                 # Matriks rasio HTMT
├── tabel_5_path_coefficients_regresi.csv                  # Hasil pengujian hipotesis regresi struktural
├── tabel_6_mediasi_bootstrap.csv                          # Uji pengaruh tidak langsung Bootstrap 5.000 sampel
├── tabel_7_benchmark_fpgrowth_vs_apriori.csv              # Perbandingan kinerja komputasi FP-Growth vs Apriori
├── tabel_8_top_association_rules.csv                      # Aturan asosiasi perilaku berbobot tinggi
├── association_rules_tourism_genai.csv                    # Dataset lengkap hasil ekstraksi aturan asosiasi
│
├── grafik_1_demografi_dan_pola_penggunaan.png             # Visualisasi platform GenAI, OTA, & cross-check
├── grafik_2_outer_model_reliabilitas_ave.png              # Visualisasi evaluasi outer model
├── grafik_3_discriminant_validity_htmt_heatmap.png        # Heatmap korelasi HTMT
├── grafik_4_benchmark_fpgrowth_vs_apriori.png             # Grafik benchmark FP-Growth vs Apriori
└── grafik_5_association_rules_scatter.png                 # Scatter plot Support vs Confidence vs Lift
```

---

## 🚀 Cara Menjalankan (*Quickstart*)

Proyek ini menggunakan Python dengan manajer paket `uv` atau `pip`:

### 1. Menjalankan Analisis Statistik Lengkap
```bash
uv run python run_analysis.py
```

### 2. Menjalankan Association Rule Mining (FP-Growth & Apriori)
```bash
uv run python run_association_mining.py
```

### 3. Ekspor Ulang Seluruh Tabel CSV & Grafik Publikasi
```bash
uv run python export_tables_and_plots.py
```

---

## 👥 Tim Peneliti
- **Jurusan Teknik Komputer dan Informatika (JTK)**
- **Politeknik Negeri Bandung (POLBAN)**
