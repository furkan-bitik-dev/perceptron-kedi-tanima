# Perceptron'umu bir web uygulamasına çeviriyorum.
# Sayfadan bir resim yüklüyorum, model "kedi" ya da "kedi değil" diye cevap veriyor.

import base64
import io
from flask import Flask, render_template, request
import matplotlib.image as mpimg       # modelin gördüğü 64x64 resmi PNG'ye çevirmek için (grafik çizmiyorum)
import perceptron_kedi as pk

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024    # en fazla 10 MB resim kabul ediyorum

# Uygulama açılırken modeli bir kez eğitiyorum (birkaç saniye sürüyor).
# Sonra gelen her resim için sadece tahmin yapıyorum.
w, b, egitim_yorumlari = pk.egit()


def sayfaya_gom(veri, tur):
    # Resmi diske kaydetmeden doğrudan sayfanın içine gömmek için base64'e çeviriyorum.
    return f"data:image/{tur};base64," + base64.b64encode(veri).decode()


@app.route("/", methods=["GET", "POST"])
def ana_sayfa():
    sonuc = None
    hata = None

    if request.method == "POST":
        dosya = request.files.get("resim")
        if not dosya or dosya.filename == "":
            hata = "Önce bir resim seçmem gerekiyor."
        else:
            try:
                veri = dosya.read()
                etiket, p, kucuk = pk.resmi_siniflandir(io.BytesIO(veri), w, b)

                # Modelin gerçekte gördüğü 64x64 hali de sayfada göstermek istiyorum.
                tampon = io.BytesIO()
                mpimg.imsave(tampon, kucuk, format="png")

                sonuc = {
                    "etiket": etiket,
                    "olasilik": round(p * 100),
                    "yorum": pk.olasilik_yorumla(p),
                    "orijinal": sayfaya_gom(veri, "jpeg"),
                    "kucuk": sayfaya_gom(tampon.getvalue(), "png"),
                }
            except Exception:
                hata = "Bu dosyayı resim olarak açamadım. JPG ya da PNG yüklemeliyim."

    return render_template("index.html", sonuc=sonuc, hata=hata, egitim_yorumlari=egitim_yorumlari)


if __name__ == "__main__":
    app.run(port=5000)
