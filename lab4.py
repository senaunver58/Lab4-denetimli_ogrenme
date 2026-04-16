import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
from sklearn.exceptions import ConvergenceWarning

# Uyarıları gizleyelim (özellikle raw data ve kötü LR denemelerinde çok fazla uyarı alacağız)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Veriyi yükleme ve bölme
wine = load_wine()
X = wine.data
y = wine.target

# %80 Eğitim, %20 Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Modüler grafik çizdirme fonksiyonumuz
def plot_loss(loss_curve, title):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve, label='Training Loss', color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch (İterasyon)')
    plt.ylabel('Loss (Kayıp)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# Ham veri ile MLP modeli (1 adet 16 nöronlu gizli katman)
mlp_raw = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
mlp_raw.fit(X_train, y_train)

# Tahmin ve Başarı
y_pred_raw = mlp_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

print(f"Ham Veri Model Accuracy: {acc_raw:.4f}")
plot_loss(mlp_raw.loss_curve_, "Ham Veri ile Eğitim - Loss Eğrisi")

# Standardizasyon işlemi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Test setini sadece dönüştürüyoruz, fit etmiyoruz (Data Leakage'i önlemek için)

# Aynı hiperparametrelerle yeni model
mlp_scaled = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
mlp_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = mlp_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, y_pred_scaled)

print(f"Standardize Veri Model Accuracy: {acc_scaled:.4f}")
plot_loss(mlp_scaled.loss_curve_, "Standardize Veri ile Eğitim - Loss Eğrisi")

# 1. Sigmoid (Logistic)
mlp_sigmoid = MLPClassifier(hidden_layer_sizes=(16, 16), activation='logistic', max_iter=500, random_state=42)
mlp_sigmoid.fit(X_train_scaled, y_train)
acc_sigmoid = accuracy_score(y_test, mlp_sigmoid.predict(X_test_scaled))

# 2. ReLU
mlp_relu = MLPClassifier(hidden_layer_sizes=(16, 16), activation='relu', max_iter=500, random_state=42)
mlp_relu.fit(X_train_scaled, y_train)
acc_relu = accuracy_score(y_test, mlp_relu.predict(X_test_scaled))

print(f"Sigmoid Model Accuracy: {acc_sigmoid:.4f}")
print(f"ReLU Model Accuracy: {acc_relu:.4f}")

# Eğrileri karşılaştırma
plt.figure(figsize=(8, 5))
plt.plot(mlp_sigmoid.loss_curve_, label='Sigmoid Loss', color='blue')
plt.plot(mlp_relu.loss_curve_, label='ReLU Loss', color='green')
plt.title("Sigmoid vs ReLU Loss Karşılaştırması")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()