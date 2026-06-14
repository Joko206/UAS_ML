# Analisis Prediktif - Prediksi Penyakit Jantung

Proyek ini bertujuan untuk membangun model **Analisis Prediktif** (klasifikasi) menggunakan dataset terstruktur (`heart.csv`) untuk memprediksi apakah seorang pasien memiliki penyakit jantung atau tidak.

## Struktur Direktori

```text
analisis_prediktif/
├── data/           # Dataset mentah dan data hasil pemrosesan (berisi heart.csv)
├── notebooks/      # File Jupyter Notebook (.ipynb) untuk EDA dan eksperimen awal
├── src/            # Skrip kode sumber utama (.py) untuk pipeline produksi
│   ├── train.py    # Skrip untuk melatih model Random Forest
│   └── predict.py  # Skrip untuk melakukan inferensi/prediksi data baru
├── models/         # Berkas penyimpanan model terbaik (.pkl)
├── requirements.txt # Daftar pustaka dan versi yang digunakan agar lingkungan reproducible
└── README.md       # Panduan langkah demi langkah menjalankan proyek
```

## Persiapan Lingkungan (Environment Setup)

Berdasarkan pengaturan sistem OS Anda, hindari penggunaan virtual environment yang dapat memicu *Permission Denied*. Gunakan instalasi level-user sebagai gantinya:

1. Pastikan Anda berada di dalam folder proyek ini:
   ```bash
   cd "/home/crackzee/Unduhan/analisis prediktif"
   ```
2. Instal dependencies yang diperlukan langsung ke user environment (bypass blokir OS Arch Linux):
   ```bash
   pip install --user --break-system-packages -r requirements.txt
   ```

## Cara Menjalankan Proyek

### 1. Melatih Model (Training)

Jalankan skrip `train.py` untuk memproses data, melatih model klasifikasi (Random Forest), dan menyimpan model terbaik.

```bash
python src/train.py
```

Skrip ini akan menampilkan akurasi model dan Classification Report, lalu model akan disimpan di direktori `models/best_model.pkl`.

### 2. Melakukan Prediksi (Inference)

Setelah model berhasil dilatih, Anda bisa menguji prediksi dengan data baru menggunakan skrip `predict.py`.

```bash
python src/predict.py
```

Skrip ini akan mensimulasikan satu data pasien dan memprediksi kemungkinan terjadinya penyakit jantung.

## Jupyter Notebooks

Untuk melakukan Eksplorasi Data (EDA - Exploratory Data Analysis), Anda dapat membuat file `.ipynb` di dalam folder `notebooks/`. Gunakan Jupyter Notebook atau JupyterLab:

```bash
jupyter notebook
```
