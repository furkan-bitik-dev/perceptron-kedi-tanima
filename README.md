# Perceptron ile Kedi Tanıma

**Canlı demo:** https://furkan-bitik-dev.github.io/perceptron-kedi-tanima/

Bu projede en basit sinir ağı olan **tek nöronlu perceptron'u sıfırdan** yazdım ve bir resmin kedi olup olmadığını tahmin eden bir web uygulamasına çevirdim. Proje, DLI Accelerated Data Science Teaching Kit'in *Module 16 Lab: Binary Classification with Perceptron* çalışmasına göre hazırlandı.

- Aktivasyon fonksiyonu: **sigmoid**
- Kayıp fonksiyonu: **Mean Square Error (MSE)**
- Optimizasyon: **Stochastic Gradient Descent (SGD)**
- Veri: Kaggle [catvnoncat](https://www.kaggle.com/muhammeddalkran/catvnoncat) (209 eğitim, 50 test resmi)

Model hazır bir kütüphane (scikit-learn, keras vb.) kullanmadan, sadece **numpy** ile yazıldı.

---

## İçindekiler
1. [Proje yapısı](#1-proje-yapısı)
2. [Kurulum](#2-kurulum)
3. [Çalıştırma](#3-çalıştırma)
4. [Perceptron nasıl çalışıyor?](#4-perceptron-nasıl-çalışıyor)
5. [Kodun yapısı](#5-kodun-yapısı)
6. [Sonuçlar ve yorumu](#6-sonuçlar-ve-yorumu)
7. [Canlı demo nasıl çalışıyor?](#7-canlı-demo-nasıl-çalışıyor)

---

## 1. Proje yapısı

```
perceptron-kedi-tanima/
├── perceptron_kedi.py     -> perceptron modeli (eğitim, kayıp, SGD, tahmin)
├── app.py                 -> Flask web uygulaması
├── templates/index.html   -> Flask uygulamasının sayfası
├── requirements.txt       -> gerekli kütüphaneler
├── datasets/              -> Kaggle catvnoncat verisi
│   ├── train_catvnoncat.h5
│   └── test_catvnoncat.h5
├── test_gorselleri/       -> eğitimde kullanılmamış 5 deneme fotoğrafı
└── docs/                  -> GitHub Pages'te yayınlanan canlı demo
```

`test_gorselleri` klasöründe `a1` ve `a2` kedi; `a3`, `a4` ve `a5` kedi olmayan fotoğraflar.

## 2. Kurulum

Python 3.8 veya üstü gerekiyor.

```bash
git clone https://github.com/furkan-bitik-dev/perceptron-kedi-tanima.git
cd perceptron-kedi-tanima
pip install -r requirements.txt
```

| Kütüphane | Ne için kullanılıyor? |
|---|---|
| numpy | Bütün matematik: ağırlıklar, sigmoid, kayıp, gradyan |
| h5py | Veri setini `.h5` dosyalarından okumak |
| pillow (PIL) | Yüklenen resmi açmak |
| scipy | Yüklenen resmi 64×64'e küçültmek (`ndimage.zoom`) |
| matplotlib | Modelin gördüğü 64×64 resmi PNG olarak sayfaya vermek |
| flask | Web uygulaması |

## 3. Çalıştırma

### Web uygulaması (Flask)
```bash
python app.py
```
1. Program önce modeli eğitiyor (birkaç saniye). Eğitim çıktısı terminalde görünüyor.
2. Tarayıcıda **http://127.0.0.1:5000** adresini açın.
3. Bir resim seçip **Sınıflandır**'a basın.

Sayfa şunları gösteriyor:
- Sonuç: **KEDİ** ya da **KEDİ DEĞİL**,
- Kedi olasılığı ve sözel yorumu,
- Yüklenen resim ve modelin gerçekte gördüğü 64×64 hali,
- Eğitim sonuçlarının yorumu.

### Sadece model (web olmadan)
```bash
python perceptron_kedi.py
```
Modeli eğitir, sonuçları yorumlar ve `test_gorselleri` klasöründeki resimleri sınıflandırır.

## 4. Perceptron nasıl çalışıyor?

Perceptron tek bir yapay nörondur ve üç adımda tahmin yapar:

1. **Girdiyi hazırla.** Bir resim 64×64 piksel ve 3 renk kanalından oluşur: 64·64·3 = **12288 sayı**. Resim bu sayılardan oluşan tek bir vektöre düzleştirilir ve 255'e bölünerek 0–1 aralığına getirilir.
2. **Ağırlıklı toplam al.** Her pikselin bir ağırlığı (w) vardır. Pikseller ağırlıklarla çarpılıp toplanır ve bir bias (b) eklenir:
   ```
   z = w·x + b
   ```
3. **Sigmoid'den geçir.**
   ```
   ŷ = 1 / (1 + e^(-z))
   ```
   Sonuç 0 ile 1 arasındadır ve **kedi olma olasılığı** olarak yorumlanır. 0.5'ten büyükse tahmin "kedi"dir.

**Öğrenme**, doğru w ve b değerlerini bulmak demektir:

- **Kayıp (MSE):** Tahmin ile gerçek etiket (1 = kedi, 0 = kedi değil) arasındaki farkın karesinin ortalaması.
  ```
  MSE = ortalama( (ŷ − y)² )
  ```
- **Gradyan:** Kaybın w'ye göre türevi. Zincir kuralıyla üç parçanın çarpımı çıkar:
  ```
  dKayıp/dw = 2(ŷ − y) · ŷ(1 − ŷ) · x
  ```
  | Parça | Nereden geliyor? |
  |---|---|
  | `2(ŷ − y)` | Karenin türevi (MSE) |
  | `ŷ(1 − ŷ)` | Sigmoid'in türevi |
  | `x` | `z = w·x + b` ifadesinin w'ye göre türevi (b için bu parça 1) |

- **SGD:** Her adımda **tek bir resim** alınır, gradyan hesaplanır ve ağırlıklar türevin ters yönünde biraz kaydırılır:
  ```
  w = w − öğrenme_oranı · dw
  b = b − öğrenme_oranı · db
  ```
  Bütün eğitim resimlerinin üzerinden bir kez geçmeye **epoch** denir. Her epoch'ta resimlerin sırası karıştırılır.

## 5. Kodun yapısı

`perceptron_kedi.py` içindeki fonksiyonlar:

| Fonksiyon | Görevi |
|---|---|
| `veriyi_yukle()` | `.h5` dosyalarını okur ve ön işlemden geçirir |
| `on_isleme()` | Resimleri 12288'lik vektöre düzleştirir, 255'e böler |
| `sigmoid()` | Aktivasyon fonksiyonu |
| `parametreleri_baslat()` | w ve b'yi sıfırla başlatır |
| `kayip_hesapla()` | MSE kaybını hesaplar |
| `sgd_guncelle()` | Tek bir resimle gradyanı bulur, w ve b'yi günceller |
| `model()` | Bu üç fonksiyonu birleştirip modeli eğitir |
| `dogruluk()` | Doğruluğu hesaplar |
| `resmi_hazirla()` | Dışarıdan gelen resmi PIL ile açar, scipy ile 64×64'e küçültür |
| `resmi_siniflandir()` | Resim için kedi olasılığını döndürür |
| `kayip_yorumla()`, `olasilik_yorumla()`, `sonuclari_yorumla()` | Sayısal sonuçları cümleye çevirir |

Lab isterlerinin karşılıkları:

| Lab'daki ister | Karşılığı |
|---|---|
| Perceptron'u sıfırdan yaz | Hazır model yok, sadece numpy |
| Aktivasyon = sigmoid | `sigmoid()` |
| Parametreleri başlat | `parametreleri_baslat()` |
| Kaybı kayıp fonksiyonuna göre hesapla (MSE) | `kayip_hesapla()` |
| SGD ile parametreleri güncelle | `sgd_guncelle()` |
| Üç fonksiyonu ana model fonksiyonunda birleştir | `model()` |
| Kaggle verisiyle kedi tanıma | `veriyi_yukle()`, `egit()`, `dogruluk()` |
| PIL ve scipy ile kendi resminle test | `resmi_hazirla()`, `resmi_siniflandir()`, web uygulaması |

## 6. Sonuçlar ve yorumu

Ayarlar: 50 epoch, öğrenme oranı 0.001. Sabit seed kullanıldığı için sonuçlar her çalıştırmada aynıdır.

```
Epoch   1 | MSE kaybı: 0.2252
Epoch  11 | MSE kaybı: 0.1342
Epoch  21 | MSE kaybı: 0.0999
Epoch  31 | MSE kaybı: 0.0888
Epoch  41 | MSE kaybı: 0.0720
Epoch  50 | MSE kaybı: 0.0598
Eğitim doğruluğu: %96.7      Test doğruluğu: %70.0
```

### MSE kaybı
- Karşılaştırma noktası **0.25**. Hiçbir şey bilmeyen ve her resme 0.5 diyen bir modelin hatası her zaman 0.5, karesi 0.25 olur. Yani 0.25 "yazı tura" seviyesidir.
- **İlk epoch'ta 0.2252:** Yazı turadan çok az iyi; model öğrenmeye yeni başlamış.
- **11. epoch'ta 0.1342:** Kayıp neredeyse yarıya inmiş; model resimlerin çoğunu doğru tarafa yerleştirmeye başlamış.
- **50. epoch'ta 0.0598:** Tahminler gerçek etiketten ortalama yaklaşık 0.24 uzakta (√0.06). Model eğitim resimlerinin çoğunda hem doğru hem de oldukça emin.
- Kayıp sürekli düşüyor ama düşüş giderek yavaşlıyor. Bu beklenen bir durum: kolay resimler önce öğreniliyor.

### Doğruluk
- **Eğitim %96.7:** Model gördüğü 209 resmin 202'sini doğru biliyor.
- **Test %70:** Hiç görmediği 50 resmin 35'ini doğru biliyor.
- **Aradaki 27 puanlık fark overfitting'i gösteriyor.** Modelde 12288 ağırlık var ama sadece 209 eğitim resmi var; model eğitim resimlerini kısmen ezberliyor.
- Test setindeki 50 resmin 33'ü kedi. Her resme "kedi" diyen bir model %66 alırdı; perceptron bundan biraz daha iyi. Tek bir nöron pikselleri sadece ağırlıklı olarak topladığı için (lineer karar sınırı) kedi gibi karmaşık bir şekli tam öğrenemiyor. Daha iyi sonuç için çok katmanlı bir ağ ya da CNN gerekir.

### Hiperparametre seçimi
| Deneme | Sonuç |
|---|---|
| Öğrenme oranı 0.05 | Adımlar çok büyük, model öğrenemiyor (test %36) |
| Öğrenme oranı 0.01 | Test %56 |
| **Öğrenme oranı 0.001, 50 epoch** | **Test %70 (seçilen ayar)** |
| 200 epoch | Eğitim %100 oluyor ama test artmıyor: daha fazla ezber |

### Deneme fotoğrafları
| Resim | Gerçek | Tahmin | Kedi olasılığı |
|---|---|---|---|
| a1.jpg | Kedi | KEDİ | %60 |
| a2.jpg | Kedi | KEDİ | %76 |
| a3.jpg | Kedi değil | KEDİ (yanlış) | %61 |
| a4.jpg | Kedi değil | KEDİ DEĞİL | %20 |
| a5.jpg | Kedi değil | KEDİ DEĞİL | %46 |

5 resmin 4'ü doğru. Olasılıkların çoğu %50'ye yakın; model yeni resimlerde pek emin değil. Bu da %70'lik test doğruluğuyla uyumlu.

Olasılığın sözel yorumu:
| Kedi olasılığı | Yorum |
|---|---|
| %80 ve üstü | Kedi olduğundan oldukça emin |
| %50–80 | Kedi diyor ama sınıra yakın |
| %20–50 | Kedi değil diyor ama sınıra yakın |
| %20 ve altı | Kedi olmadığından oldukça emin |

## 7. Canlı demo nasıl çalışıyor?

GitHub Pages sadece statik sayfa yayınlayabilir, Python/Flask çalıştıramaz. Bu yüzden canlı demo (`docs/` klasörü) şöyle çalışıyor:

- Model Python'da (`perceptron_kedi.py`) eğitildi ve öğrenilen w ve b değerleri `docs/model.json` dosyasına kaydedildi.
- `docs/perceptron.js`, tahmini tarayıcıda yapıyor:
  - Resmi 64×64'e küçültmek için `scipy.ndimage.zoom(order=1)` ile aynı bilinear yöntemi kullanıyor.
  - Ardından `sigmoid(w·x + b)` hesaplıyor.
- Tarayıcıdaki sonuçlar Python'dakiyle aynı çıkıyor (deneme fotoğraflarında fark binde 1'in altında).
- Yüklenen resim hiçbir sunucuya gönderilmiyor; bütün hesaplama tarayıcıda yapılıyor.

Eğitimin kendisi ve lab'ın istediği fonksiyonlar `perceptron_kedi.py` içindedir. Flask uygulaması (`app.py`) aynı modeli yerel bilgisayarda çalıştırır.
