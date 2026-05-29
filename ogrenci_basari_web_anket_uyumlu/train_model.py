import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = Path("anket_verisi.csv")
MODEL_PATH = Path("model_package.joblib")

rename_map = {
    "Zaman damgası": "Zaman",
    "Cinsiyetiniz nedir?": "Cinsiyet",
    "Kaçıncı sınıfta eğitim görüyorsunuz?": "Sinif",
    "Hangi fakültede öğrenim görüyorsunuz?": "Fakulte",
    "Not ortalamanız hangi sistem üzerinden hesaplanıyor?": "Not_Sistemi",
    "4'lük sistemde geçen dönemki ortalamanızı giriniz (Küsüratlı değerleri virgül yerine noktayla ayırınız. Örn: 2.15):": "Gecen_4luk",
    "4'lük sistemde genel ortalamanızı giriniz (Küsüratlı değerleri virgül yerine noktayla ayırınız. Örn: 2.15):": "Genel_4luk",
    "100'lük sistemde geçen dönemki ortalamanızı giriniz (Küsüratlı değerleri virgül yerine noktayla ayırınız. Örn: 93.5):": "Gecen_100luk",
    "100'lük sistemde genel ortalamanızı giriniz (Küsüratlı değerleri virgül yerine noktayla ayırınız. Örn: 93.5):": "Genel_100luk",
    "Önceki dönem aldığınız ders sayısını giriniz:": "Ders_Sayisi",
    "Önceki dönem başarısız olduğunuz ders sayısını giriniz:": "Basarisiz_Ders",
    "Haftada kaç saat ders çalışıyorsunuz?": "Haftalik_Calisma",
    "Sınav haftaları kaç saat ders çalışıyorsunuz?": "Sinav_Haftasi_Calisma",
    "Annenizin eğitim durumu nedir?": "Anne_Egitim",
    "Babanızın eğitim durumu nedir?": "Baba_Egitim",
    "Evinizin aylık toplam geliri ne kadardır?": "Aylik_Gelir",
    "Bir işte çalışıyor musunuz?": "Calisma_Durumu",
    "Eğitim-öğretim dönemi içerisinde nerede ikamet ediyorsunuz?": "Ikamet",
    "Günlük ortalama ekran sürenizi seçiniz:": "Ekran_Suresi",
    "Günlük ortalama uyku sürenizi seçiniz:": "Uyku_Suresi"
}


def temizle_sayi(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    try:
        deger = float(s)

        # 4'lük sistemde 289 gibi gelen değerleri 2.89'a çevirir.
        if deger > 4 and deger <= 400:
            deger = deger / 100

        return deger
    except ValueError:
        return np.nan


def nadir_kategorileri_birlestir(df, col, min_count=10):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other="Diğer")
    return df


def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("anket_verisi.csv bulunamadı. Bu dosyayı proje klasörüne koymalısınız.")

    df = pd.read_csv(DATA_PATH)
    df.rename(columns=rename_map, inplace=True)

    sayisal_sutunlar = [
        "Gecen_4luk", "Genel_4luk", "Gecen_100luk", "Genel_100luk",
        "Ders_Sayisi", "Basarisiz_Ders"
    ]

    for col in sayisal_sutunlar:
        df[col] = df[col].apply(temizle_sayi)
        if col in ["Gecen_4luk", "Genel_4luk"]:
            df[col] = df[col].apply(lambda x: x / 100 if pd.notna(x) and x > 4 else x)

    df["Ortak_Basari_100"] = np.where(
        df["Genel_100luk"].notna(), df["Genel_100luk"], df["Genel_4luk"] * 25
    )

    df["Ortak_Gecen_100"] = np.where(
        df["Gecen_100luk"].notna(), df["Gecen_100luk"], df["Gecen_4luk"] * 25
    )

    df.loc[(df["Ortak_Basari_100"] < 0) | (df["Ortak_Basari_100"] > 100), "Ortak_Basari_100"] = np.nan
    df.loc[(df["Ortak_Gecen_100"] < 0) | (df["Ortak_Gecen_100"] > 100), "Ortak_Gecen_100"] = np.nan

    df = df[df["Ortak_Basari_100"].notna()].copy()

    df.loc[df["Basarisiz_Ders"] > df["Ders_Sayisi"], "Basarisiz_Ders"] = np.nan

    for col in ["Fakulte", "Sinif", "Aylik_Gelir", "Ikamet"]:
        df = nadir_kategorileri_birlestir(df, col, min_count=10)

    egitim_map = {
        "Hiçbiri": 0,
        "İlkokul": 1,
        "Ortaokul": 2,
        "Lise": 3,
        "Üniversite": 4
    }

    calisma_map = {
        "0-5 saat": 1,
        "5-10 saat": 2,
        "10-20 saat": 3,
        "20-30 saat": 4,
        "30-50 saat": 5,
        "50+ saat": 6
    }

    sinif_map = {
        "Hazırlık": 0,
        "1.sınıf": 1,
        "2.sınıf": 2,
        "3.sınıf": 3,
        "4.sınıf": 4,
        "5.sınıf": 5,
        "6.sınıf": 6,
        "Diğer": 3
    }

    gelir_map = {
        "0 - 28.075 TL": 1,
        "28.075 - 60.000 TL": 2,
        "60.000 - 90.000 TL": 3,
        "90.000+ TL": 4,
        "Diğer": np.nan
    }

    ekran_map = {
        "0 - 3 saat": 1,
        "3 - 6 saat": 2,
        "7 - 10 saat": 3,
        "10 - 15 saat": 4,
        "15 - 24 saat": 5
    }

    uyku_map = {
        "0 - 3 saat": 1,
        "3 - 6 saat": 2,
        "7 - 10 saat": 3,
        "10 - 15 saat": 4,
        "15 - 24 saat": 5
    }

    df["Anne_Egitim"] = df["Anne_Egitim"].map(egitim_map)
    df["Baba_Egitim"] = df["Baba_Egitim"].map(egitim_map)
    df["Haftalik_Calisma"] = df["Haftalik_Calisma"].map(calisma_map)
    df["Sinav_Haftasi_Calisma"] = df["Sinav_Haftasi_Calisma"].map(calisma_map)
    df["Sinif"] = df["Sinif"].map(sinif_map)
    df["Aylik_Gelir"] = df["Aylik_Gelir"].map(gelir_map)
    df["Ekran_Suresi"] = df["Ekran_Suresi"].map(ekran_map)
    df["Uyku_Suresi"] = df["Uyku_Suresi"].map(uyku_map)

    df["Basarisiz_Orani"] = df["Basarisiz_Ders"] / (df["Ders_Sayisi"] + 1)
    df["Calisma_Artisi"] = df["Sinav_Haftasi_Calisma"] - df["Haftalik_Calisma"]
    df["Ekran_Uyku_Orani"] = df["Ekran_Suresi"] / (df["Uyku_Suresi"] + 1)

    return df


def main():
    df = prepare_data()

    sayisal_ozellikler = [
        "Ortak_Gecen_100",
        "Ders_Sayisi",
        "Basarisiz_Ders",
        "Anne_Egitim",
        "Baba_Egitim",
        "Haftalik_Calisma",
        "Sinav_Haftasi_Calisma",
        "Sinif",
        "Aylik_Gelir",
        "Ekran_Suresi",
        "Uyku_Suresi",
        "Basarisiz_Orani",
        "Calisma_Artisi",
        "Ekran_Uyku_Orani",
    ]

    kategorik_ozellikler = [
        "Cinsiyet",
        "Fakulte",
        "Calisma_Durumu",
        "Ikamet",
    ]

    X = df[sayisal_ozellikler + kategorik_ozellikler].copy()
    y = df["Ortak_Basari_100"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, sayisal_ozellikler),
            ("cat", cat_transformer, kategorik_ozellikler)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(pipeline, MODEL_PATH)

    print("Model başarıyla kaydedildi:", MODEL_PATH)
    print("Test RMSE:", round(rmse, 3))
    print("Test MAE :", round(mae, 3))
    print("Test R2  :", round(r2, 3))


if __name__ == "__main__":
    main()