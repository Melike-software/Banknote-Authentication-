import pandas as pd
from pathlib import Path

# CSV dosyasının yolu
dosya_yolu = Path(__file__).resolve().parent / "data" / "BankNote_Authentication.csv"

# Veri setini okuma
df = pd.read_csv(dosya_yolu)

print("VERİ SETİNİN İLK 5 SATIRI")
print(df.head())

print("\nVERİ SETİNİN BOYUTU")
print(df.shape)

print("\nSÜTUN İSİMLERİ")
print(df.columns.tolist())

print("\nEKSİK VERİ SAYISI")
print(df.isnull().sum())

print("\nSINIF DAĞILIMI")
print(df["class"].value_counts())

print("\nVERİ SETİNİN İSTATİSTİKLERİ")
print(df.describe())


import matplotlib.pyplot as plt

# Özelliklerin birbirleriyle ilişkisini incele
ozellikler = ["variance", "skewness", "curtosis", "entropy"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for i in range(len(ozellikler)):
    for j in range(i + 1, len(ozellikler)):
        ax = axes.flat[0]
        break
    break

ciftler = [
    ("variance", "skewness"),
    ("variance", "curtosis"),
    ("variance", "entropy"),
    ("skewness", "curtosis"),
    ("skewness", "entropy"),
    ("curtosis", "entropy")
]

for ax, (x_col, y_col) in zip(axes.flat, ciftler):
    for sinif, renk in [(0, "blue"), (1, "red")]:
        veri = df[df["class"] == sinif]
        ax.scatter(
            veri[x_col],
            veri[y_col],
            label=f"Sinif {sinif}",
            alpha=0.5,
            s=12,
            color=renk
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()

plt.suptitle("Banknot Veri Seti - Ozellik Dagilimlari")
plt.tight_layout()

plt.savefig("outputs/grafikler/ozellik_dagilimlari.png", dpi=300)
plt.show()


import numpy as np

# Dört sayısal özelliği al
X = df[["variance", "skewness", "curtosis", "entropy"]].to_numpy(dtype=float)
y = df["class"].to_numpy(dtype=int)

# Özellikleri standartlaştır
ortalama = X.mean(axis=0)
std = X.std(axis=0)
X = (X - ortalama) / np.where(std == 0, 1, std)

# Doğrusal sınıflandırıcı: Perceptron
# Bu kod yalnızca doğrusal ayrılabilirliği kontrol etmek için
X_bias = np.column_stack((np.ones(len(X)), X))
y_isaret = np.where(y == 1, 1, -1)

w = np.zeros(X_bias.shape[1])
hata_sayisi = 0

for epoch in range(1000):
    hata_sayisi = 0

    for i in range(len(X_bias)):
        if y_isaret[i] * np.dot(X_bias[i], w) <= 0:
            w += y_isaret[i] * X_bias[i]
            hata_sayisi += 1

    if hata_sayisi == 0:
        print("Veri doğrusal olarak ayrılabiliyor.")
        print("Bulunan epoch:", epoch + 1)
        break
else:
    print("1000 epoch içinde tam doğrusal ayrım bulunamadı.")
    print("Son epoch hata sayısı:", hata_sayisi)

from scipy.optimize import linprog

# X_bias ve y_isaret önceki Perceptron kodunda oluşturuldu.
# Her örnek için y_i * (w · x_i) >= 1 koşulunu arıyoruz.

A_ub = -(y_isaret[:, None] * X_bias)
b_ub = -np.ones(len(y_isaret))

sonuc = linprog(
    c=np.zeros(X_bias.shape[1]),
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(None, None)] * X_bias.shape[1],
    method="highs"
)

print("\nDOĞRUSAL AYRILABILIRLIK KONTROLÜ")

if sonuc.status == 0:
    print("Sonuc: Veri seti dogrusal olarak ayrilabiliyor.")
elif sonuc.status == 2:
    print("Sonuc: Veri seti dogrusal olarak tam ayrilamiyor.")
else:
    print("Sonuc: Test kesinlesmedi.")
    print("Aciklama:", sonuc.message)

#kontrol bitti

# EĞİTİM VE TEST VERİLERİNİN AYRILMASI

import numpy as np

# Özellikleri ve hedef değişkeni al
X = df[["variance", "skewness", "curtosis", "entropy"]].to_numpy(dtype=float)
y = df["class"].to_numpy(dtype=int)

# Sonuçların her çalıştırmada aynı olması için
rng = np.random.default_rng(42)

# Verilerin indekslerini karıştır
indeksler = rng.permutation(len(X))

# %80 eğitim, %20 test
egitim_sayisi = int(len(X) * 0.80)

egitim_indeksleri = indeksler[:egitim_sayisi]
test_indeksleri = indeksler[egitim_sayisi:]

X_train = X[egitim_indeksleri]
X_test = X[test_indeksleri]

y_train = y[egitim_indeksleri]
y_test = y[test_indeksleri]

# Standartlaştırma parametrelerini sadece
# eğitim verileri üzerinden hesapla
ortalama = X_train.mean(axis=0)
std = X_train.std(axis=0)
std = np.where(std == 0, 1, std)

X_train = (X_train - ortalama) / std
X_test = (X_test - ortalama) / std

print("\nEĞİTİM VE TEST AYRIMI")
print("Eğitim verisi:", X_train.shape)
print("Test verisi:", X_test.shape)

print("\nEğitim sınıf dağılımı:")
print(np.unique(y_train, return_counts=True))

print("\nTest sınıf dağılımı:")
print(np.unique(y_test, return_counts=True))

#--------------------
#linear_regression ekledik
#--------------------

# DOĞRUSAL REGRESYON MODELİ

from linear_regression import LinearRegression

# Modeli oluştur
linear_model = LinearRegression(
    learning_rate=0.01,
    epochs=2000
)

# Eğitim verileriyle modeli eğit
linear_model.fit(X_train, y_train)

# Test verilerinde tahmin yap
linear_predictions = linear_model.predict(X_test)

# Doğru tahmin sayısını hesapla
correct_predictions = np.sum(linear_predictions == y_test)

# Accuracy hesapla
accuracy = correct_predictions / len(y_test)

print("\nDOĞRUSAL REGRESYON SONUÇLARI")
print("Doğru tahmin sayısı:", correct_predictions)
print("Toplam test verisi:", len(y_test))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Son MSE değeri:", linear_model.loss_history[-1])
print("Ağırlıklar:", linear_model.weights)
print("Bias:", linear_model.bias)

#logistiği ekledik

#--------------------
# LOJISTIK REGRESYON MODELI
#--------------------

from logistic_regression import LogisticRegression

# Modeli oluştur
logistic_model = LogisticRegression(
    learning_rate=0.1,
    epochs=5000
)

# Eğitim
logistic_model.fit(X_train, y_train)

# Test verilerinde tahmin
logistic_predictions = logistic_model.predict(X_test)

# Accuracy hesaplama
logistic_correct = np.sum(logistic_predictions == y_test)
logistic_accuracy = logistic_correct / len(y_test)

print("\nLOJİSTİK REGRESYON SONUÇLARI")
print("Doğru tahmin sayısı:", logistic_correct)
print("Toplam test verisi:", len(y_test))
print("Accuracy:", round(logistic_accuracy * 100, 2), "%")
print("Son BCE değeri:", logistic_model.loss_history[-1])
print("Ağırlıklar:", logistic_model.weights)
print("Bias:", logistic_model.bias)

# İki modelin doğruluk karşılaştırması
print("\nİKİ MODELİN KARŞILAŞTIRILMASI")
print("Doğrusal regresyon:", round(accuracy * 100, 2), "%")
print("Lojistik regresyon:", round(logistic_accuracy * 100, 2), "%")

#-------------
#evaluation ekliyoruz
#-------------

# PERFORMANS DEĞERLENDİRMESI

from evaluation import evaluate_model

linear_results = evaluate_model(y_test, linear_predictions)
logistic_results = evaluate_model(y_test, logistic_predictions)

print("\n" + "=" * 50)
print("MODELLERİN PERFORMANS KARSILAŞTIRMASI")
print("=" * 50)

print(f"{'Metrik':<15} {'Dogrusal':>12} {'Lojistik':>12}")
print("-" * 42)

for metric in ["accuracy", "precision", "recall", "f1_score"]:
    print(
        f"{metric:<15} "
        f"{linear_results[metric]:>12.4f} "
        f"{logistic_results[metric]:>12.4f}"
    )

print("\nDOĞRUSAL REGRESYON CONFUSION MATRIX")
print(linear_results["confusion_matrix"])

print("\nLOJİSTİK REGRESYON CONFUSION MATRIX")
print(logistic_results["confusion_matrix"])

#tabloların görselleştirilmesi

# SONUCLARIN GRAFIKLE GOSTERILMESI

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs/grafikler", exist_ok=True)

# Grafiklerin genel görünümü
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

mavi = "#5981A4"
turuncu = "#B17E46"


# 1. PERFORMANS KARŞILAŞTIRMASI

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
metric_keys = ["accuracy", "precision", "recall", "f1_score"]

linear_scores = [
    linear_results[m] * 100 for m in metric_keys
]

logistic_scores = [
    logistic_results[m] * 100 for m in metric_keys
]

x = np.arange(len(metrics))
width = 0.34

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2,
    linear_scores,
    width,
    color=mavi,
    label="Doğrusal Regresyon"
)

bars2 = ax.bar(
    x + width / 2,
    logistic_scores,
    width,
    color=turuncu,
    label="Lojistik Regresyon"
)

# Sütunların üzerine değerleri yaz
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"%{height:.2f}".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_title(
    "Doğrusal ve Lojistik Regresyonun Performans Karşılaştırması",
    pad=18,
    fontweight="bold"
)

ax.set_ylabel("Başarı Oranı (%)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 112)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    "outputs/grafikler/performans_karsilastirmasi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 2. KARMAŞIKLIK MATRİSLERİ

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

models = [
    (
        "Doğrusal Regresyon",
        linear_results["confusion_matrix"]
    ),
    (
        "Lojistik Regresyon",
        logistic_results["confusion_matrix"]
    )
]

for ax, (name, cm) in zip(axes, models):

    ax.imshow(cm, cmap="Blues", vmin=0, vmax=153)

    ax.set_title(name, fontweight="bold", pad=12)
    ax.set_xlabel("Tahmin Edilen Sınıf")
    ax.set_ylabel("Gerçek Sınıf")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):

            value = cm[i, j]

            # Koyu hücrelerde beyaz yazı kullan
            text_color = (
                "white" if value > 76 else "black"
            )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=text_color
            )

fig.suptitle(
    "Modellerin Karmaşıklık Matrisleri",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/grafikler/karisiklik_matrisleri.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 3. EĞİTİM KAYIP GRAFİKLERİ

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    linear_model.loss_history,
    color=mavi,
    linewidth=2
)

axes[0].set_title(
    "Doğrusal Regresyon - MSE",
    fontweight="bold"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Ortalama Kare Hata (MSE)")
axes[0].grid(alpha=0.2)

axes[1].plot(
    logistic_model.loss_history,
    color=turuncu,
    linewidth=2
)

axes[1].set_title(
    "Lojistik Regresyon - BCE",
    fontweight="bold"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("İkili Çapraz Entropi (BCE)")
axes[1].grid(alpha=0.2)

fig.suptitle(
    "Modellerin Eğitim Sürecindeki Kayıp Değişimleri",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/grafikler/kayip_grafikleri.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRapor grafikleri başarıyla oluşturuldu!")
