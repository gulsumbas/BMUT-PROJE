from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
import csv
from datetime import datetime
import math

app = Flask(__name__)

MODEL_PATH = Path("model_package.joblib")
RESPONSES_PATH = Path("web_anket_cevaplari.csv")


def float_form(name, default=None):
    value = request.form.get(name, "").strip().replace(",", ".")
    if value == "":
        return default
    return float(value)


def int_form(name, default=0):
    value = request.form.get(name, "").strip()
    if value == "":
        return default
    return int(float(value.replace(",", ".")))


def normalize_4luk(value):
    """
    4'lük sistemde 289 gibi yazılan değerleri 2.89'a çevirir.
    2.89 gibi girilen değerleri değiştirmez.
    """
    if value is None:
        return None

    if value > 4 and value <= 400:
        value = value / 100

    return value


def build_input_dataframe():
    not_sistemi = request.form.get("Not_Sistemi")

    gecen_4luk = None
    genel_4luk = None
    gecen_100luk = None
    genel_100luk = None

    if not_sistemi == "4'lük sistem":
        gecen_4luk = normalize_4luk(float_form("Gecen_4luk"))
        genel_4luk = normalize_4luk(float_form("Genel_4luk"))

        ortak_gecen_100 = gecen_4luk * 25
        ortak_basari_100 = genel_4luk * 25

    else:
        gecen_100luk = float_form("Gecen_100luk")
        genel_100luk = float_form("Genel_100luk")

        ortak_gecen_100 = gecen_100luk
        ortak_basari_100 = genel_100luk

    ders_sayisi = int_form("Ders_Sayisi")
    basarisiz_ders = int_form("Basarisiz_Ders")

    anne_egitim = int_form("Anne_Egitim")
    baba_egitim = int_form("Baba_Egitim")
    haftalik_calisma = int_form("Haftalik_Calisma")
    sinav_calisma = int_form("Sinav_Haftasi_Calisma")
    sinif = int_form("Sinif")
    aylik_gelir = int_form("Aylik_Gelir")
    ekran_suresi = int_form("Ekran_Suresi")
    uyku_suresi = int_form("Uyku_Suresi")

    basarisiz_orani = basarisiz_ders / (ders_sayisi + 1)
    calisma_artisi = sinav_calisma - haftalik_calisma
    ekran_uyku_orani = ekran_suresi / (uyku_suresi + 1)

    data = {
        "Ortak_Gecen_100": ortak_gecen_100,
        "Ders_Sayisi": ders_sayisi,
        "Basarisiz_Ders": basarisiz_ders,
        "Anne_Egitim": anne_egitim,
        "Baba_Egitim": baba_egitim,
        "Haftalik_Calisma": haftalik_calisma,
        "Sinav_Haftasi_Calisma": sinav_calisma,
        "Sinif": sinif,
        "Aylik_Gelir": aylik_gelir,
        "Ekran_Suresi": ekran_suresi,
        "Uyku_Suresi": uyku_suresi,
        "Basarisiz_Orani": basarisiz_orani,
        "Calisma_Artisi": calisma_artisi,
        "Ekran_Uyku_Orani": ekran_uyku_orani,
        "Cinsiyet": request.form.get("Cinsiyet"),
        "Fakulte": request.form.get("Fakulte"),
        "Calisma_Durumu": request.form.get("Calisma_Durumu"),
        "Ikamet": request.form.get("Ikamet"),
    }

    raw_data = {
        "Not_Sistemi": not_sistemi,
        "Gecen_4luk": gecen_4luk,
        "Genel_4luk": genel_4luk,
        "Gecen_100luk": gecen_100luk,
        "Genel_100luk": genel_100luk,
        "Ortak_Basari_100_Kullanici_Girdisi": ortak_basari_100,
    }

    return pd.DataFrame([data]), raw_data


def save_response(input_df, raw_data, prediction):
    row = input_df.iloc[0].to_dict()
    row.update(raw_data)
    row["Tahmini_Basari_100"] = round(float(prediction), 2)
    row["Kayit_Zamani"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_header = not RESPONSES_PATH.exists()
    with RESPONSES_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def yorum_uret(tahmin):
    if tahmin >= 85:
        return "Başarı durumu çok yüksek görünüyor."
    if tahmin >= 70:
        return "Başarı durumu iyi görünüyor."
    if tahmin >= 55:
        return "Başarı durumu orta seviyede görünüyor."
    return "Başarı riski bulunuyor, akademik destek önerilebilir."


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    yorum = None
    error = None

    if request.method == "POST":
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Model dosyası bulunamadı. Önce terminalden 'python train_model.py' komutunu çalıştırmalısınız."
                )

            input_df, raw_data = build_input_dataframe()

            if input_df["Basarisiz_Ders"].iloc[0] > input_df["Ders_Sayisi"].iloc[0]:
                raise ValueError("Başarısız ders sayısı, alınan ders sayısından fazla olamaz.")

            model = joblib.load(MODEL_PATH)
            tahmin = float(model.predict(input_df)[0])
            tahmin = max(0, min(100, tahmin))

            save_response(input_df, raw_data, tahmin)

            prediction = round(tahmin, 2)
            yorum = yorum_uret(tahmin)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, yorum=yorum, error=error)


if __name__ == "__main__":
    app.run(debug=True)