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
6. [Sonuçları nasıl iyileştirdim?](#6-sonuçları-nasıl-iyileştirdim)
7. [Sonuçlar ve yorumu](#7-sonuçlar-ve-yorumu)
8. [Canlı demo nasıl çalışıyor?](#8-canlı-demo-nasıl-çalışıyor)

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
| `veriyi_yukle()` | `.h5` dosyalarını okur |
| `veriyi_cogalt()` | Eğitim resimlerini yatay çevirir ve kedi resimlerini çoğaltarak sınıfları dengeler |
| `on_isleme()` | Resimleri 12288'lik vektöre düzleştirir, 255'e böler |
| `sigmoid()` | Aktivasyon fonksiyonu |
| `parametreleri_baslat()` | w ve b'yi sıfırla başlatır |
| `kayip_hesapla()` | MSE kaybını hesaplar |
| `sgd_guncelle()` | Tek bir resimle gradyanı bulur, w ve b'yi günceller |
| `model()` | Bu üç fonksiyonu birleştirip modeli eğitir (azalan öğrenme oranıyla) |
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

## 6. Sonuçları nasıl iyileştirdim?

İlk sürüm test setinde %70 doğruluk veriyordu ve yanlışlarının çoğu, kedi resimlerine "kedi değil" demesiydi. Modelin yapısına (tek nöron, sigmoid, MSE, SGD) dokunmadan üç değişiklik yaptım:

| Değişiklik | Neden? |
|---|---|
| **Yatay çevirme** | Aynadaki kedi de kedidir. Her eğitim resminin aynadaki halini ekleyince 209 resim 418 oluyor ve model kedinin hangi tarafa baktığını ezberleyemiyor. |
| **Sınıf dengeleme** | Eğitim verisinin üçte ikisi "kedi değil" (72 kedi, 137 kedi değil). Model kararsız kaldığında "kedi değil" demeye alışıyordu. Kedi resimlerinden rastgele kopyalar ekleyerek SGD'nin iki sınıfı eşit sıklıkta görmesini sağladım (548 resim, 274 kedi). |
| **Azalan öğrenme oranı** | `adım = 0.001 / (1 + 0.3 · epoch)`. Başta büyük adımlarla hızlı öğreniyor, sonlarda küçük adımlarla oturuyor. Sabit oranda SGD son epoch'larda zıplayıp iyi bulduğu noktadan uzaklaşabiliyordu. |

Bir değişikliğin gerçekten işe yarayıp yaramadığını tek bir çalıştırmayla anlamak zor; SGD'nin rastgeleliği yüzünden sonuç her seferinde birkaç puan oynuyor. Bu yüzden her ayarı 6 farklı rastgele başlangıçla çalıştırıp ortalamaya baktım:

| Sürüm | Test doğruluğu (6 çalıştırma ortalaması) | Kedileri bilme | Kedi olmayanları bilme |
|---|---|---|---|
| İlk sürüm | ≈ %72 (±6) | ≈ %71 | ≈ %74 |
| + çevirme + dengeleme + azalan öğrenme oranı | **≈ %77 (±3)** | **≈ %78** | **≈ %75** |

Ortalama doğruluk yaklaşık 5 puan arttı ve kazancın çoğu kedileri bilmekten geliyor. Sonuçların çalıştırmadan çalıştırmaya oynaması da yarıya indi (±6 puandan ±3 puana); yani model daha kararlı.

Denediğim ama işe yaramayan şeyler: piksel bazında standardizasyon (z-score), gri tonlamaya çevirme ve resmi 32×32'ye küçültme. Hepsi doğruluğu düşürdü.

### Ayarlar neden böyle seçildi?

Bütün ayarlar `perceptron_kedi.py` dosyasında. Kullanılan değerler 202. satırda: `model(X_cok, y_cok, epoch_sayisi=40, ogrenme_orani=0.001, azalma=0.3)`.

| Ayar | Değer | Kodda |
|---|---|---|
| Ağırlıklar | 12288 tane w + 1 bias b, başlangıçta hepsi 0 | satır 68–69 |
| Başlangıç öğrenme oranı | 0.001 | satır 202 |
| Öğrenme oranının azalması | `adım = 0.001 / (1 + 0.3 · epoch)` | satır 108 |
| Epoch sayısı | 40 | satır 104, 202 |
| Karar eşiği | 0.5 | satır 123, 186 |

**Neden ağırlıklar sıfırdan başlıyor?** Başlangıçta z = 0 ve sigmoid 0.5 veriyor: model her resme "yarı yarıya" diyor, yani hiçbir ön yargısı yok. Çok katmanlı ağlarda sıfırla başlatmak bütün nöronların aynı şeyi öğrenmesine yol açar; tek nöronda böyle bir sorun yok.

**Neden 0.001?** Bir güncellemenin ne kadar büyük olduğunu hesaplayabiliyoruz. Bir resmin 12288 sayısının karelerinin toplamı ortalama **2851**. SGD tek bir resimle güncelleme yaptığında o resmin z değeri yaklaşık `öğrenme oranı × ortak × 2851` kadar değişiyor. Başlangıçta `ortak` yaklaşık 0.25:

| Öğrenme oranı | İlk güncellemede z'nin değişimi | Sonuç (ilk sürümde denendi) |
|---|---|---|
| 0.05 | ≈ 35.6 | Sigmoid hemen 0'a ya da 1'e yapışıyor, `ŷ(1 − ŷ)` sıfırlanıyor, türev ölüyor. Model öğrenemiyor (test %36). |
| 0.01 | ≈ 7.1 | sigmoid(7) = 0.999. Model tek bir resme aşırı tepki veriyor (test %56). |
| **0.001** | **≈ 0.7** | sigmoid 0.5'ten 0.67'ye çıkıyor: anlamlı ama aşırı olmayan bir adım (test %70–74). |

**Neden öğrenme oranını aşama aşama düşürüyoruz?**
SGD her adımda tek bir resme bakıyor. Model iyi bir noktaya yaklaştığında her resim ağırlıkları kendi yönüne çekiyor: kedi resmi bir yöne, kedi olmayan başka bir yöne. Adım sabit kalırsa ağırlıklar iyi noktanın etrafında zıplayıp durur. Adımlar küçüldükçe bu zıplamalar da küçülür ve model oturur.

| Epoch | 1 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| Adım | 0.001 | 0.00045 | 0.00027 | 0.00015 | 0.00008 |

Başta büyük adımlarla hızlı öğreniyor; 40. epoch'ta adım başlangıcın yaklaşık %8'ine iniyor.

Aynı veri ve aynı başlangıçla, sabit ve azalan oranı karşılaştırdım (40 epoch):

| | Sabit oran (0.001) | Azalan oran |
|---|---|---|
| Kaybın yükseldiği epoch sayısı | 22 / 39 | 11 / 39 |
| En büyük kayıp sıçraması | 0.06 (30. epoch'ta 0.0765 → 0.1248) | 0.01 |
| Eğitim doğruluğu | %96.2 | %86.6 |
| Test doğruluğu | %68 | **%74** |
| Ağırlık vektörünün büyüklüğü ‖w‖ | 1.69 | 0.68 |

Bu tablo iki faydayı gösteriyor:
1. **Kararlılık:** Kayıp daha az zıplıyor ve model iyi bulduğu noktadan uzaklaşmıyor.
2. **Daha az ezber:** Adımlar küçüldüğü için ağırlıklar aşırı büyüyemiyor (‖w‖ 1.69'dan 0.68'e indi). Sabit oranda model eğitim resimlerini %96 doğrulukla ezberliyor ama testte %68'de kalıyor. Azalan oranda eğitim ile test birbirine yaklaşıyor. Bu, erken durdurma ya da düzenlileştirmeye benzer bir etki.

Teorik olarak SGD'nin bir noktaya oturması için adımların iki şartı sağlaması gerekir: toplamları sonsuz olmalı ki model gereken yere kadar gidebilsin, karelerinin toplamı sonlu olmalı ki gürültü sönsün (Robbins–Monro şartları). `1 / (1 + 0.3 · epoch)` biçimindeki adımlar ikisini de sağlıyor.

**Neden 40 epoch ve neden 0.3 azalma?** Her ayarı 6 farklı rastgele başlangıçla denedim:

| Ayar | Ortalama test doğruluğu | Çalıştırmalar arası oynama |
|---|---|---|
| Azalma 0.3, 20 epoch | %78 | ±3.1 |
| Azalma 0.3, 30 epoch | %75 | ±5.1 |
| **Azalma 0.3, 40 epoch** | **%77** | **±2.8** |
| Azalma 0.1, 40 epoch | %77 | ±3.8 |
| Azalma yok, 40 epoch | %77 | ±5.6 |

Doğruluklar birbirine yakın; 40 epoch ve 0.3 azalmayı, sonuçlar en az oynadığı için seçtim. 40. epoch'ta adım o kadar küçük ki daha fazla epoch neredeyse hiçbir şeyi değiştirmiyor.

**Neden eşik 0.5?** Sigmoid'in çıktısı kedi olma olasılığı; 0.5 "yarı yarıya" demek. Sınıfları dengelediğimiz için iki hataya da eşit davranan doğal eşik bu.

## 7. Sonuçlar ve yorumu

Ayarlar: 40 epoch, başlangıç öğrenme oranı 0.001, azalma 0.3. Sabit seed kullanıldığı için sonuçlar her çalıştırmada aynıdır.

```
Epoch   1 | MSE kaybı: 0.2411
Epoch  11 | MSE kaybı: 0.1545
Epoch  21 | MSE kaybı: 0.1359
Epoch  31 | MSE kaybı: 0.1326
Epoch  40 | MSE kaybı: 0.1248
Eğitim doğruluğu: %86.6      Test doğruluğu: %74.0
Test: 33 kediden 24'ü, 17 kedi olmayandan 13'ü doğru
```

Bu çalıştırmanın %74'ü, 6 çalıştırmanın ortalaması olan %77'nin biraz altında; SGD'nin rastgeleliğinden kaynaklanan normal bir oynama.

### MSE kaybı
- Karşılaştırma noktası **0.25**. Hiçbir şey bilmeyen ve her resme 0.5 diyen bir modelin hatası her zaman 0.5, karesi 0.25 olur. Yani 0.25 "yazı tura" seviyesidir.
- **İlk epoch'ta 0.2411:** Yazı turanın hemen altında; model öğrenmeye yeni başlamış.
- **11. epoch'ta 0.1545:** Kayıp üçte bir azalmış; model resimleri doğru tarafa ayırmaya başlamış.
- **40. epoch'ta 0.1248:** Tahminler gerçek etiketten ortalama yaklaşık 0.35 uzakta (√0.125). Model çoğu resimde doğru tarafta ama çok emin değil.
- İlk sürümde son kayıp 0.06'ydı, yani daha düşüktü. Ama o düşük kayıp ezberin işaretiydi: model eğitim resimlerinde çok emindi, yeni resimlerde ise yanılıyordu. Çevrilmiş ve dengelenmiş veriyi ezberlemek daha zor olduğu için kayıp daha yüksek kalıyor, test doğruluğu ise artıyor.

### Doğruluk
- **Eğitim %86.6:** Model orijinal 209 resmin 181'ini doğru biliyor.
- **Test %74:** Hiç görmediği 50 resmin 37'sini doğru biliyor.
- **Aradaki 13 puanlık fark** ilk sürümdeki 27 puanın yarısı. Model hâlâ biraz ezberliyor ama çoğunlukla genelleyebiliyor.
- **Tek nöronun sınırı:** Perceptron pikselleri sadece ağırlıklı olarak toplar (lineer karar sınırı). Kedinin şekli, kulakları, gözleri gibi karmaşık örüntüleri öğrenemez; daha çok renk ve parlaklık dağılımına bakar. Bu yüzden %75–80'in üstüne çıkmak için çok katmanlı bir ağ ya da CNN gerekir.

### Deneme fotoğrafları
| Resim | Gerçek | Tahmin | Kedi olasılığı |
|---|---|---|---|
| a1.jpg | Kedi | KEDİ | %63 |
| a2.jpg | Kedi | KEDİ | %78 |
| a3.jpg | Kedi değil | KEDİ (yanlış) | %71 |
| a4.jpg | Kedi değil | KEDİ DEĞİL | %43 |
| a5.jpg | Kedi değil | KEDİ (yanlış) | %61 |

Bu fotoğraflar eğitim verisinden çok farklı: yüksek çözünürlüklü, farklı ışıkta ve farklı kadrajla çekilmiş. Model bu tür resimlerde zorlanıyor ve olasılıklar %50'ye yakın çıkıyor. Kırmızı ve mor tonlu Paris fotoğrafları (a3, a5) renk dağılımı yüzünden kedi sanılıyor.

Olasılığın sözel yorumu:
| Kedi olasılığı | Yorum |
|---|---|
| %75 ve üstü | Kedi olduğundan oldukça emin |
| %50–75 | Kedi diyor ama sınıra yakın |
| %25–50 | Kedi değil diyor ama sınıra yakın |
| %25 ve altı | Kedi olmadığından oldukça emin |

## 8. Canlı demo nasıl çalışıyor?

GitHub Pages sadece statik sayfa yayınlayabilir, Python/Flask çalıştıramaz. Bu yüzden canlı demo (`docs/` klasörü) şöyle çalışıyor:

- Model Python'da (`perceptron_kedi.py`) eğitildi ve öğrenilen w ve b değerleri `docs/model.json` dosyasına kaydedildi.
- `docs/perceptron.js`, tahmini tarayıcıda yapıyor:
  - Resmi 64×64'e küçültmek için `scipy.ndimage.zoom(order=1)` ile aynı bilinear yöntemi kullanıyor.
  - Ardından `sigmoid(w·x + b)` hesaplıyor.
- Tarayıcıdaki sonuçlar Python'dakiyle aynı çıkıyor (deneme fotoğraflarında fark binde 1'in altında).
- Yüklenen resim hiçbir sunucuya gönderilmiyor; bütün hesaplama tarayıcıda yapılıyor.

Eğitimin kendisi ve lab'ın istediği fonksiyonlar `perceptron_kedi.py` içindedir. Flask uygulaması (`app.py`) aynı modeli yerel bilgisayarda çalıştırır.
