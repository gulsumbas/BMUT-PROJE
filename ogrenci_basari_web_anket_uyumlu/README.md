# Öğrenci Başarısı Tahmin Sistemi - Anket Uyumlu Web Uygulaması

Bu sürümde web formu, Google Forms'taki anket sorularına göre düzenlenmiştir.

## Akış

1. Kullanıcı cinsiyet, sınıf ve fakülte bilgilerini seçer.
2. Not sistemini seçer.
3. 4'lük sistem seçilirse yalnızca 4'lük ortalama soruları açılır.
4. 100'lük sistem seçilirse yalnızca 100'lük ortalama soruları açılır.
5. Daha sonra tüm kullanıcılar ortak akademik ve yaşam bilgileri sorularını cevaplar.
6. Sistem makine öğrenmesi modeliyle tahmini başarı puanını üretir.

## Çalıştırma

1. `anket_verisi.csv` dosyanı bu klasöre koy.
2. Gerekli paketleri yükle:

```bash
pip install -r requirements.txt
```

3. Modeli eğit:

```bash
python train_model.py
```

4. Siteyi çalıştır:

```bash
python app.py
```

5. Tarayıcıda aç:

```text
http://127.0.0.1:5000
```