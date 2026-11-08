# Banknote-Authentication-
Banknote Authentication veri seti üzerinde geliştirilen doğrusal ve lojistik regresyon algoritmalarının karşılaştırılması.

## Veri Seti
1.372 örnek, 4 özellik ve 2 sınıftan oluşmaktadır.

[Banknote Authentication – Kaggle](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data/data)

## Model Sonuçları
| Metrik | Doğrusal Regresyon | Lojistik Regresyon |
|---|---:|---:|
| Accuracy | %97,82 | %99,27 |
| Precision | %95,31 | %98,39 |
| Recall | %100 | %100 |
| F1-score | %97,60 | %99,19 |

## Kullanılan Teknolojiler
Python, NumPy, pandas, Matplotlib, SciPy.

## Çalıştırma
```bash
pip install -r requirements.txt
python main.py
```
