# Teman Masak - Multilingual Recipe Classification & Recommendation

Sistem rekomendasi resep masakan multibahasa (Indonesia & Inggris) menggunakan Hybrid Model (TF-IDF + Fine-Tuned BERT) dan Neural Network untuk klasifikasi kategori resep serta rekomendasi resep berbasis kemiripan teks.

## Fitur Utama

* **Multilingual Dataset Processing**: Mendukung dataset resep Indonesia dan Inggris
* **Advanced Text Cleaning**: Normalisasi bahan, singkatan, takaran, emoji, dan format resep
* **Automatic Language Detection**: Deteksi otomatis Bahasa Indonesia dan Inggris
* **Fine-Tuned BERT Models**:

  * IndoBERT untuk resep Bahasa Indonesia
  * BERT Base Uncased untuk resep Bahasa Inggris
* **Hybrid Features**:

  * TF-IDF (pattern matching)
  * BERT Embeddings (semantic understanding)
* **Deep Neural Network Classifier**
* **Recipe Recommendation System** berbasis cosine similarity

## Struktur Project

```text
teman-masak-modeldata/
│
├── processing_v3.py              # Preprocessing dataset multilingual
├── model_multi.py                # Training multilingual model
├── inference_multi.py            # Inference & rekomendasi resep
│
├── requirements.txt
│
├── capstone/
│   ├── recipes_extended.csv
│   ├── Indonesian_Food_Recipes.csv
│   ├── dataset_indo.csv          # Output preprocessing
│   ├── dataset_en.csv            # Output preprocessing
│   └── dataset_merged.csv        # Output preprocessing
│
└── model2/
    ├── multilingual_model.pt
    ├── tfidf_multi.pkl
    ├── label_encoder_multi.pkl
    ├── scaler_tfidf_multi.pkl
    ├── scaler_bert_multi.pkl
    ├── recipes_multi.pkl
    ├── classes_multi.txt
    │
    ├── indobert_finetuned/
    └── bert_en_finetuned/
```

## Cara Pakai

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset

Taruh dataset berikut ke dalam folder `capstone/`

```text
capstone/
├── recipes_extended.csv
└── Indonesian_Food_Recipes.csv
```

### 3. Jalankan Preprocessing

python processing_v3.py

Output:

```text
capstone/dataset_indo.csv
capstone/dataset_en.csv
capstone/dataset_merged.csv
```

### 4. Training Model

python model_multi.py


Proses training meliputi:

* TF-IDF Feature Extraction
* Fine-Tuning IndoBERT
* Fine-Tuning English BERT
* Embedding Extraction
* Feature Scaling
* Neural Network Training
* Model Evaluation

Output:

```text
model2/
├── multilingual_model.pt
├── tfidf_multi.pkl
├── label_encoder_multi.pkl
├── scaler_tfidf_multi.pkl
├── scaler_bert_multi.pkl
├── recipes_multi.pkl
├── classes_multi.txt
├── indobert_finetuned/
└── bert_en_finetuned/
```

### 5. Jalankan Inference


python inference_multi.py


Contoh:

```text
Search: ayam goreng crispy

(confidence: 94.12%)

Recommendations:

1. Ayam Goreng Crispy
2. Ayam Kremes
3. Chicken Fried Steak
4. Ayam Tepung
5. Fried Chicken
```

## Detail File

| File               | Fungsi                                                      | Input                       | Output                           |
| ------------------ | ----------------------------------------------------------- | --------------------------- | -------------------------------- |
| processing_v3.py   | Cleaning, normalisasi, language detection, category mapping | Dataset mentah              | dataset_indo.csv, dataset_en.csv |
| model_multi.py     | Fine-tuning BERT dan training classifier                    | Dataset hasil preprocessing | Model dan artifacts              |
| inference_multi.py | Prediksi kategori dan rekomendasi resep                     | Query pengguna              | Kategori dan rekomendasi resep   |

## Arsitektur Model

### Text Features

TF-IDF

* Max Features: 3000
* N-Gram: (1,2)
* Sublinear TF
* L2 Normalization

### Semantic Features

Bahasa Indonesia:

* IndoBERT Base P1
* Fine-tuned pada dataset resep Indonesia

Bahasa Inggris:

* BERT Base Uncased
* Fine-tuned pada dataset resep Inggris

### Final Classifier

Neural Network:

```text
Input
  ↓
Linear (1024)
  ↓
BatchNorm
  ↓
Dropout
  ↓
Linear (512)
  ↓
BatchNorm
  ↓
Dropout
  ↓
Linear (256)
  ↓
BatchNorm
  ↓
Dropout
  ↓
Output Layer
```

## Pipeline

```text
Dataset
   ↓
Preprocessing
   ↓
Language Detection
   ↓
TF-IDF Features
   ↓
Fine-Tuned BERT Embeddings
   ↓
Feature Scaling
   ↓
Feature Concatenation
   ↓
Neural Network Classifier
   ↓
Category Prediction
   ↓
Recipe Recommendation
```

## Kategori yang Didukung

Model mendukung berbagai kategori resep seperti:

* Hidangan Utama
* Daging
* Unggas
* Seafood
* Nabati
* Telur
* Dessert
* Minuman
* Camilan
* Sarapan
* Sehat
* Roti
* Internasional

Kategori akhir bergantung pada dataset yang digunakan saat training.

## Requirements

* Python 3.11+
* PyTorch 2.6+
* CUDA 12.x (opsional)
* RAM 16GB+
* GPU NVIDIA direkomendasikan untuk proses training

## Notes

* Model otomatis mendeteksi bahasa query pengguna.
* Fine-tuned BERT akan disimpan setelah training selesai.
* Inference dapat berjalan di CPU maupun GPU.
* Folder `model2/` dibuat otomatis saat training.
