// Python'da eğittiğim perceptron'un tarayıcıdaki karşılığı.
// Eğitim burada yapılmıyor: Python'da öğrendiğim w ve b değerlerini model.json'dan okuyorum
// ve sadece tahmin yapıyorum. Ön işleme Python'dakiyle (PIL + scipy.ndimage.zoom) birebir aynı.

// scipy.ndimage.zoom(resim, (64/yükseklik, 64/genişlik, 1), order=1) ile aynı işi yapıyorum:
// çıkıştaki her piksel, girişte i*(H-1)/63 konumuna denk geliyor ve 4 komşu pikselden
// doğrusal (bilinear) ara değer alınıyor.
function resmi64eIndir(piksel, H, W) {
  const cikis = new Uint8Array(64 * 64 * 3);
  for (let i = 0; i < 64; i++) {
    const y = i * (H - 1) / 63, y0 = Math.floor(y), y1 = Math.min(y0 + 1, H - 1), dy = y - y0;
    for (let j = 0; j < 64; j++) {
      const x = j * (W - 1) / 63, x0 = Math.floor(x), x1 = Math.min(x0 + 1, W - 1), dx = x - x0;
      for (let c = 0; c < 3; c++) {
        const p = (yy, xx) => piksel[(yy * W + xx) * 3 + c];
        const deger = p(y0, x0) * (1 - dy) * (1 - dx) + p(y0, x1) * (1 - dy) * dx
                    + p(y1, x0) * dy * (1 - dx) + p(y1, x1) * dy * dx;
        // Python'daki np.clip(...).astype(np.uint8) gibi: sınırla ve aşağı yuvarla
        cikis[(i * 64 + j) * 3 + c] = Math.floor(Math.min(255, Math.max(0, deger + 1e-9)));
      }
    }
  }
  return cikis;
}

// Nöronun yaptığı iş: z = w.x + b, sonra sigmoid. x değerleri 255'e bölünüyor.
function kediOlasiligi(model, kucuk) {
  let z = model.b;
  for (let k = 0; k < kucuk.length; k++) z += model.w[k] * (kucuk[k] / 255);
  return 1 / (1 + Math.exp(-z));
}

function olasilikYorumla(p) {
  if (p >= 0.8) return "Model bunun kedi olduğundan oldukça emin.";
  if (p > 0.5) return "Model kedi diyor ama emin değil, sınıra yakın bir karar.";
  if (p > 0.2) return "Model kedi değil diyor ama emin değil, sınıra yakın bir karar.";
  return "Model bunun kedi olmadığından oldukça emin.";
}

if (typeof module !== "undefined") module.exports = { resmi64eIndir, kediOlasiligi, olasilikYorumla };
