# Module 16 Lab - Perceptron ile kedi / kedi değil sınıflandırması
# En basit sinir ağını, yani tek nöronlu perceptron'u sıfırdan yazıyorum.
# Lab'ın istediği gibi: aktivasyon = sigmoid, kayıp = MSE, optimizasyon = SGD.
# Hazır model kütüphanesi (scikit-learn, keras vb.) kullanmıyorum, her şeyi numpy ile yazıyorum.

import os
import numpy as np                 # bütün matematik
import h5py                        # veri setini .h5 dosyasından okumak için
from PIL import Image              # dışarıdan gelen resmi açmak için
from scipy import ndimage          # o resmi 64x64'e küçültmek için


# ---------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------
def veriyi_yukle(klasor="datasets"):
    # Kaggle'daki catvnoncat verisini okuyorum: 209 eğitim, 50 test resmi (64x64 renkli).
    egitim = h5py.File(os.path.join(klasor, "train_catvnoncat.h5"), "r")
    test = h5py.File(os.path.join(klasor, "test_catvnoncat.h5"), "r")
    X_egitim = np.array(egitim["train_set_x"])
    y_egitim = np.array(egitim["train_set_y"]).reshape(1, -1)   # 1 = kedi, 0 = kedi değil
    X_test = np.array(test["test_set_x"])
    y_test = np.array(test["test_set_y"]).reshape(1, -1)
    return on_isleme(X_egitim), y_egitim, on_isleme(X_test), y_test


def on_isleme(X):
    # Perceptron resim değil vektör bekliyor. Her resmi 64*64*3 = 12288 sayılık
    # bir sütuna düzleştiriyorum ve 255'e bölerek değerleri 0-1 arasına çekiyorum.
    return X.reshape(X.shape[0], -1).T / 255.0


# ---------------------------------------------------------------
# PERCEPTRON
# ---------------------------------------------------------------
def sigmoid(z):
    # Aktivasyon fonksiyonum. Her sayıyı 0 ile 1 arasına sıkıştırıyor,
    # ben de çıkan sayıyı "kedi olma olasılığı" olarak yorumluyorum.
    return 1 / (1 + np.exp(-z))


def tahmin_olasiligi(w, b, X):
    # Nöronun yaptığı iş bu kadar: girdileri ağırlıklarla çarp, topla, sigmoid'den geçir.
    return sigmoid(np.dot(w.T, X) + b)


# 1) Parametreleri başlatma
def parametreleri_baslat(boyut):
    # Her piksel için bir ağırlık (w) ve bir tane bias (b) var. Hepsini sıfırla başlatıyorum.
    w = np.zeros((boyut, 1))
    b = 0.0
    return w, b


# 2) Kaybı hesaplama (Mean Square Error)
def kayip_hesapla(y_hat, y):
    # Tahmin ile gerçek etiketin farkının karesini alıp ortalıyorum.
    # 0'a ne kadar yakınsa model o kadar iyi.
    return np.mean((y_hat - y) ** 2)


# 3) SGD ile parametreleri güncelleme
def sgd_guncelle(w, b, x, y, ogrenme_orani):
    # SGD'de her adımda TEK bir resim kullanıyorum (x: tek resim, y: onun etiketi).
    y_hat = tahmin_olasiligi(w, b, x)

    # Kaybın w'ye göre türevini zincir kuralıyla buluyorum:
    #   kayıp = (y_hat - y)^2          -> türevi 2 (y_hat - y)
    #   y_hat = sigmoid(z)             -> türevi y_hat (1 - y_hat)
    #   z = w.x + b                    -> w'ye göre türevi x, b'ye göre türevi 1
    ortak = 2 * (y_hat - y) * y_hat * (1 - y_hat)
    dw = x * ortak
    db = ortak.item()

    # Kaybı azaltmak için türevin ters yönünde küçük bir adım atıyorum.
    w = w - ogrenme_orani * dw
    b = b - ogrenme_orani * db
    return w, b


# Ana model: yukarıdaki üç fonksiyonu burada birleştiriyorum
def model(X, y, epoch_sayisi=50, ogrenme_orani=0.001):
    w, b = parametreleri_baslat(X.shape[0])
    rng = np.random.default_rng(0)      # sonuç her çalıştırmada aynı çıksın diye

    for epoch in range(epoch_sayisi):
        # Bir epoch = bütün eğitim resimlerinin üzerinden bir kez geçmek.
        # Her epoch'ta resimlerin sırasını karıştırıyorum (SGD'nin "stokastik" kısmı).
        for i in rng.permutation(X.shape[1]):
            w, b = sgd_guncelle(w, b, X[:, i:i + 1], y[:, i:i + 1], ogrenme_orani)

        kayip = kayip_hesapla(tahmin_olasiligi(w, b, X), y)
        if epoch % 10 == 0 or epoch == epoch_sayisi - 1:
            print(f"Epoch {epoch + 1:3d} | MSE kaybı: {kayip:.4f} -> {kayip_yorumla(kayip)}")

    return w, b, kayip


def dogruluk(w, b, X, y):
    # Olasılık 0.5'ten büyükse "kedi" diyorum, sonra kaç tanesini bildiğime bakıyorum.
    tahmin = tahmin_olasiligi(w, b, X) > 0.5
    return float(np.mean(tahmin == y) * 100)


# ---------------------------------------------------------------
# SAYILARI SÖZLE YORUMLAMA
# ---------------------------------------------------------------
def kayip_yorumla(kayip):
    # Karşılaştırma noktam 0.25: model hiçbir şey bilmeyip her resme 0.5 dese
    # hata hep 0.5 olur, karesi 0.25 eder. Yani 0.25 "yazı tura" seviyesi.
    if kayip >= 0.25:
        return "henüz öğrenmedi (her resme 0.5 demekle aynı ya da daha kötü)"
    if kayip >= 0.15:
        return "öğrenmeye başladı ama tahminleri hâlâ kararsız"
    if kayip >= 0.05:
        return "eğitim resimlerinin çoğunu doğru tarafa yerleştiriyor"
    return "eğitim resimlerini neredeyse ezbere biliyor"


def olasilik_yorumla(p):
    if p >= 0.8:
        return "Model bunun kedi olduğundan oldukça emin."
    if p > 0.5:
        return "Model kedi diyor ama emin değil, sınıra yakın bir karar."
    if p > 0.2:
        return "Model kedi değil diyor ama emin değil, sınıra yakın bir karar."
    return "Model bunun kedi olmadığından oldukça emin."


def sonuclari_yorumla(egitim_dog, test_dog):
    fark = egitim_dog - test_dog
    satirlar = [
        f"Eğitim doğruluğu %{egitim_dog:.1f}: model daha önce gördüğü 209 resmin "
        f"yaklaşık {round(egitim_dog / 100 * 209)} tanesini doğru biliyor.",
        f"Test doğruluğu %{test_dog:.1f}: hiç görmediği 50 resmin {round(test_dog / 100 * 50)} tanesini doğru biliyor.",
    ]
    if fark > 15:
        satirlar.append(f"Aradaki {fark:.0f} puanlık fark, modelin eğitim resimlerini kısmen ezberlediğini "
                        "gösteriyor (overfitting). Tek nöron ve 209 resim için bu beklenen bir durum.")
    else:
        satirlar.append("Eğitim ve test doğruluğu birbirine yakın, model ezberlemeden öğrenmiş.")
    return satirlar


# ---------------------------------------------------------------
# DIŞARIDAN GELEN RESMİ MODELE HAZIRLAMA (PIL + scipy)
# ---------------------------------------------------------------
def resmi_hazirla(dosya):
    # Resmi PIL ile açıyorum ve scipy ile 64x64'e küçültüyorum, çünkü model 64x64 resimle eğitildi.
    resim = np.array(Image.open(dosya).convert("RGB"))
    oran = (64 / resim.shape[0], 64 / resim.shape[1], 1)     # renk kanalına dokunmuyorum
    kucuk = np.clip(ndimage.zoom(resim, oran, order=1), 0, 255).astype(np.uint8)
    # Eğitimdeki ön işlemin aynısı: düzleştir ve 255'e böl.
    return kucuk, on_isleme(kucuk[np.newaxis])


def resmi_siniflandir(dosya, w, b):
    kucuk, x = resmi_hazirla(dosya)
    p = float(tahmin_olasiligi(w, b, x)[0, 0])
    etiket = "KEDİ" if p > 0.5 else "KEDİ DEĞİL"
    return etiket, p, kucuk


# ---------------------------------------------------------------
# EĞİTİM (hem bu dosya hem de web uygulaması burayı çağırıyor)
# ---------------------------------------------------------------
def egit():
    X_egitim, y_egitim, X_test, y_test = veriyi_yukle()
    print(f"Eğitim: {X_egitim.shape[1]} resim | Test: {X_test.shape[1]} resim | "
          f"Her resim {X_egitim.shape[0]} sayı\n")

    w, b, son_kayip = model(X_egitim, y_egitim, epoch_sayisi=50, ogrenme_orani=0.001)

    egitim_dog = dogruluk(w, b, X_egitim, y_egitim)
    test_dog = dogruluk(w, b, X_test, y_test)
    yorumlar = [f"Son MSE kaybı {son_kayip:.4f}: {kayip_yorumla(son_kayip)}."]
    yorumlar += sonuclari_yorumla(egitim_dog, test_dog)
    print()
    for satir in yorumlar:
        print("-", satir)
    return w, b, yorumlar


if __name__ == "__main__":
    # Bu dosyayı tek başına çalıştırırsam modeli eğitip test_gorselleri klasörünü deniyorum.
    w, b, _ = egit()
    print("\ntest_gorselleri klasöründeki resimler:")
    for ad in sorted(os.listdir("test_gorselleri")):
        etiket, p, _ = resmi_siniflandir(os.path.join("test_gorselleri", ad), w, b)
        print(f"- {ad:18s} -> {etiket:10s} (kedi olasılığı %{p * 100:.0f}) {olasilik_yorumla(p)}")
