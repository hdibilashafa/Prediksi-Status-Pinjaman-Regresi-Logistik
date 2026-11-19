# Klasifikasi Status Pinjaman dengan Regresi Logistik

![Python](https://img.shields.io/badge/Python-3-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![pandas](https://img.shields.io/badge/pandas-Preprocessing-150458)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic_Regression-F7931E)

Prediksi **status pinjaman (`loan_status`)** dari **45.000 data pemohon** menggunakan **Regresi Logistik**. Notebook ini mencakup pengkodean fitur kategorikal, normalisasi Min-Max, penilaian fitur dengan ANOVA F-test, dan evaluasi model.

**Hasil: akurasi 88,93%** pada 9.000 data uji.

---

## Dataset

File `loan_data.csv` berisi **45.000 baris dan 14 kolom**.

| Kelompok | Kolom |
|---|---|
| Profil pemohon | `person_age`, `person_gender`, `person_education`, `person_income`, `person_emp_exp`, `person_home_ownership` |
| Data pinjaman | `loan_amnt`, `loan_intent`, `loan_int_rate`, `loan_percent_income` |
| Riwayat kredit | `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file` |
| **Target** | `loan_status` (0 / 1) |

---

## Alur Pemrosesan

| Langkah | Detail |
|---|---|
| **1. Pengkodean kategorikal** | `LabelEncoder` pada 5 kolom: `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file` |
| **2. Normalisasi** | `MinMaxScaler` (rentang 0–1) pada 8 kolom numerik: `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`. Hasilnya disimpan ke `normalisasi_dataset.csv` |
| **3. Penilaian fitur** | `SelectKBest` dengan `f_classif` (ANOVA F-test), `k='all'`, sehingga seluruh 13 fitur tetap dipakai dan hanya skornya yang ditampilkan |
| **4. Split data** | 80% latih (36.000) dan 20% uji (9.000), `random_state=42` |
| **5. Model** | `LogisticRegression` (parameter default) |

### Skor fitur (ANOVA F-test)

| Peringkat | Fitur | Skor |
|---|---|---|
| 1 | `previous_loan_defaults_on_file` | 18.824,73 |
| 2 | `loan_percent_income` | 7.824,79 |
| 3 | `loan_int_rate` | 5.574,45 |
| 4 | `person_home_ownership` | 2.602,91 |
| 5 | `person_income` | 845,53 |
| 6 | `loan_amnt` | 528,21 |
| 7 | `loan_intent` | 235,52 |
| 8 | `person_age` | 20,76 |
| 9 | `person_emp_exp` | 18,88 |
| 10 | `cb_person_cred_hist_length` | 9,93 |
| 11 | `credit_score` | 2,63 |
| 12 | `person_education` | 0,14 |
| 13 | `person_gender` | 0,01 |

Fitur dengan skor tertinggi adalah **riwayat gagal bayar sebelumnya**, disusul **rasio pinjaman terhadap pendapatan** dan **suku bunga pinjaman**. Jenis kelamin dan pendidikan hampir tidak berpengaruh.

---

## Hasil Evaluasi

**Akurasi: 88,93%** (9.000 data uji)

| Kelas | Precision | Recall | F1-score | Jumlah data |
|---|---|---|---|---|
| 0 | 0,92 | 0,94 | 0,93 | 6.990 |
| 1 | 0,77 | 0,73 | 0,75 | 2.010 |
| Macro avg | 0,84 | 0,83 | 0,84 | 9.000 |
| Weighted avg | 0,89 | 0,89 | 0,89 | 9.000 |

---

## Cara Menjalankan

### Opsi 1: Google Colab

1. Buka file `.ipynb` di Google Colab
2. Upload `loan_data.csv` ke sesi Colab (ikon folder di sebelah kiri, lalu upload)
3. Klik **Runtime → Run all**

### Opsi 2: Komputer sendiri

```bash
git clone https://github.com/hdibilashafa/NAMA-REPO.git
cd NAMA-REPO
pip install pandas scikit-learn jupyter
jupyter notebook
```

Pastikan `loan_data.csv` berada di **folder yang sama** dengan notebook, lalu jalankan semua sel secara berurutan. File `normalisasi_dataset.csv` akan dibuat otomatis.

---

## Struktur Repository

```
.
├── loan_logistic_regression.ipynb
├── loan_data.csv
└── README.md
```

---

## Tech Stack

Python · pandas · scikit-learn (LabelEncoder, MinMaxScaler, SelectKBest, LogisticRegression) · Jupyter Notebook

---

## Keterbatasan

- **Kelas tidak seimbang.** Kelas 1 hanya sekitar 22% data uji (2.010 dari 9.000). Menebak kelas 0 untuk semua data sudah menghasilkan akurasi 77,7%, jadi precision dan recall kelas 1 (0,77 dan 0,73) lebih menggambarkan kemampuan model dibanding akurasi saja
- **Semua fitur tetap dipakai.** `SelectKBest` dijalankan dengan `k='all'`, sehingga tidak ada fitur yang dibuang meskipun beberapa skornya sangat rendah (`person_gender`, `person_education`, `credit_score`)
- **Normalisasi dan penilaian fitur dilakukan sebelum data dibagi**, sehingga informasi data uji ikut terpakai saat menghitung skala dan skor. Idealnya split dilakukan lebih dulu, lalu `MinMaxScaler` di-fit hanya pada data latih
- **`LabelEncoder` dipakai pada kolom nominal** (`person_gender`, `person_home_ownership`, `loan_intent`). Angka hasil encoding menyiratkan urutan yang sebenarnya tidak ada, padahal Regresi Logistik adalah model linear. One-hot encoding lebih tepat untuk kolom ini
- Hanya satu model dan satu kali split, tanpa cross-validation atau tuning parameter
