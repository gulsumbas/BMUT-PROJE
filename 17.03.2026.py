import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("anket_verisi.csv")


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

df.rename(columns=rename_map, inplace=True)


def temizle_sayi(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    try:
        return float(s)
    except:
        return np.nan

sayisal_sutunlar = [
    "Gecen_4luk",
    "Genel_4luk",
    "Gecen_100luk",
    "Genel_100luk",
    "Ders_Sayisi",
    "Basarisiz_Ders"
]

for col in sayisal_sutunlar:
    if col in df.columns:
        df[col] = df[col].apply(temizle_sayi)


df["Ortak_Basari_100"] = np.where(
    df["Genel_100luk"].notna(),
    df["Genel_100luk"],
    df["Genel_4luk"] * 25
)

df["Ortak_Gecen_100"] = np.where(
    df["Gecen_100luk"].notna(),
    df["Gecen_100luk"],
    df["Gecen_4luk"] * 25
)

df["Ortak_Basari_4"] = np.where(
    df["Genel_4luk"].notna(),
    df["Genel_4luk"],
    df["Genel_100luk"] / 25
)

df.loc[(df["Ortak_Basari_100"] < 0) | (df["Ortak_Basari_100"] > 100), "Ortak_Basari_100"] = np.nan
df.loc[(df["Ortak_Gecen_100"] < 0) | (df["Ortak_Gecen_100"] > 100), "Ortak_Gecen_100"] = np.nan
df.loc[(df["Ortak_Basari_4"] < 0) | (df["Ortak_Basari_4"] > 4), "Ortak_Basari_4"] = np.nan


df = df[df["Ortak_Basari_100"].notna()].copy()

print("\n--- VERİ SETİ GENEL BİLGİ ---")
print(df.info())

print("\n--- TOPLAM GEÇERLİ GÖZLEM SAYISI ---")
print(len(df))

print("\n--- NOT SİSTEMİ DAĞILIMI ---")
print(df["Not_Sistemi"].value_counts(dropna=False))


analiz_sayisal = [
    "Ortak_Basari_100",
    "Ortak_Gecen_100",
    "Ortak_Basari_4",
    "Ders_Sayisi",
    "Basarisiz_Ders"
]

analiz_sayisal = [c for c in analiz_sayisal if df[c].notna().sum() >= 2]

print("\n--- SAYISAL DEĞİŞKENLERİN TANIMSAL İSTATİSTİKLERİ ---")
print(df[analiz_sayisal].describe())

kategorik_sutunlar = [
    "Cinsiyet",
    "Sinif",
    "Fakulte",
    "Not_Sistemi",
    "Haftalik_Calisma",
    "Sinav_Haftasi_Calisma",
    "Anne_Egitim",
    "Baba_Egitim",
    "Aylik_Gelir",
    "Calisma_Durumu",
    "Ikamet",
    "Ekran_Suresi",
    "Uyku_Suresi"
]

print("\n--- KATEGORİK DEĞİŞKENLERİN FREKANSLARI ---")
for col in kategorik_sutunlar:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))


print("\n--- KATEGORİK DEĞİŞKENLERE GÖRE ORTALAMA BAŞARI ---")
for col in kategorik_sutunlar:
    print(f"\n{col}")
    print(df.groupby(col)["Ortak_Basari_100"].agg(["count", "mean"]).sort_values("mean", ascending=False))

print("\n--- SAYISAL DEĞİŞKENLERİN BAŞARIYLA KORELASYONU ---")
corr_target = df[analiz_sayisal].corr(numeric_only=True)["Ortak_Basari_100"].sort_values(ascending=False)
print(corr_target)


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

for column in df.columns:
    plt.figure()

    if pd.api.types.is_numeric_dtype(df[column]):
        temiz_veri = df[column].dropna()

        if len(temiz_veri) > 0:
            plt.hist(temiz_veri, bins=12, edgecolor="black")
            ortalama = temiz_veri.mean()
            medyan = temiz_veri.median()

            plt.axvline(ortalama, linestyle="--", linewidth=2, label=f"Ortalama: {ortalama:.2f}")
            plt.axvline(medyan, linestyle="-.", linewidth=2, label=f"Medyan: {medyan:.2f}")

            plt.title(f"{column} - Sayısal Dağılım", fontsize=13, fontweight="bold")
            plt.xlabel(column)
            plt.ylabel("Frekans")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Gösterilecek veri yok", ha="center", va="center")
            plt.title(f"{column} - Sayısal Dağılım")

    else:
        sayim = df[column].fillna("Boş").astype(str).value_counts()
        kategori_sayisi = len(sayim)

        if kategori_sayisi > 0:
            if kategori_sayisi <= 3:
                plt.pie(
                    sayim.values,
                    labels=sayim.index,
                    autopct="%1.1f%%",
                    startangle=90
                )
                plt.title(f"{column} - Oransal Dağılım", fontsize=13, fontweight="bold")
                plt.axis("equal")
            else:
                if kategori_sayisi > 12:
                    sayim = sayim.head(12)

                ax = sayim.plot(kind="bar", edgecolor="black")

                plt.title(f"{column} - Kategorik Dağılım", fontsize=13, fontweight="bold")
                plt.xlabel(column)
                plt.ylabel("Frekans")
                plt.xticks(rotation=35, ha="right")
                plt.grid(axis="y", linestyle="--", alpha=0.5)

                for p in ax.patches:
                    yukseklik = p.get_height()
                    ax.annotate(
                        str(int(yukseklik)),
                        (p.get_x() + p.get_width() / 2, yukseklik),
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )
        else:
            plt.text(0.5, 0.5, "Gösterilecek veri yok", ha="center", va="center")
            plt.title(f"{column} - Grafik")

    plt.tight_layout()
    plt.show()


corr_matrix = df[analiz_sayisal].corr(numeric_only=True)

print("\n--- KORELASYON MATRİSİ ---")
print(corr_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df["Ortak_Basari_100"].hist(bins=10)
plt.title("Genel Başarı Dağılımı")
plt.xlabel("Başarı Notu (100'lük)")
plt.ylabel("Frekans")
plt.tight_layout()
plt.show()


sayisal_ozellikler = [
    "Ortak_Gecen_100",
    "Ders_Sayisi",
    "Basarisiz_Ders"
]

kategorik_ozellikler = [
    "Cinsiyet",
    "Sinif",
    "Fakulte",
    "Haftalik_Calisma",
    "Sinav_Haftasi_Calisma",
    "Anne_Egitim",
    "Baba_Egitim",
    "Aylik_Gelir",
    "Calisma_Durumu",
    "Ikamet",
    "Ekran_Suresi",
    "Uyku_Suresi"
]

X = df[sayisal_ozellikler + kategorik_ozellikler].copy()
y = df["Ortak_Basari_100"].copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n--- EĞİTİM / TEST BOYUTLARI ---")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# =========================================================
# 12) PREPROCESSOR
# =========================================================
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


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
}

results = []
trained_models = {}


for model_name, regressor in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    results.append({
        "Model": model_name,
        "Egitim_RMSE": train_rmse,
        "Test_RMSE": test_rmse,
        "Egitim_MAE": train_mae,
        "Test_MAE": test_mae,
        "Egitim_R2": train_r2,
        "Test_R2": test_r2
    })

    trained_models[model_name] = {
        "pipeline": pipeline,
        "y_pred_test": y_pred_test
    }


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Test_R2", ascending=False).reset_index(drop=True)

print("\n--- MODEL KARŞILAŞTIRMA TABLOSU ---")
print(results_df.round(3))


best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]["pipeline"]
best_y_pred_test = trained_models[best_model_name]["y_pred_test"]

print("\n--- EN İYİ MODEL ---")
print(f"En iyi model: {best_model_name}")

print("\n--- EN İYİ MODEL YORUMU ---")
best_train_r2 = results_df.loc[0, "Egitim_R2"]
best_test_r2 = results_df.loc[0, "Test_R2"]

if abs(best_train_r2 - best_test_r2) < 0.10:
    print("Seçilen model eğitim ve test verisinde benzer performans göstermektedir. Genelleme başarısı kabul edilebilir.")
elif best_train_r2 > best_test_r2:
    print("Seçilen model eğitim verisinde daha iyi performans göstermektedir. Aşırı öğrenme olabilir.")
else:
    print("Seçilen model test verisinde beklenmedik şekilde farklı sonuç vermektedir. Veri yapısı kontrol edilmelidir.")


for model_name, model_data in trained_models.items():

    y_pred = model_data["y_pred_test"]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred)

    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())

    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("Gerçek Başarı")
    plt.ylabel("Tahmin Edilen Başarı")
    plt.title(f"Gerçek - Tahmin Karşılaştırması ({model_name})")

    plt.tight_layout()
    plt.show()
