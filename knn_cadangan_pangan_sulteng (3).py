import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────
df_raw = pd.read_excel("dataset_cadanganPangan_dan_kepadatanPenduduk_sulteng.xlsx")

# Agregasi per kabupaten: ambil rata-rata CPPD (karena ada data bulanan)
df = df_raw.groupby("Kabupaten/Kota").agg(
    Kepadatan_Jiwa_per_km2=("Kepadatan_Jiwa_per_km2", "first"),
    Jumlah_Bencana=("Jumlah_Bencana", "first"),
    CPPD=("CPPD (Ton)", "mean")
).reset_index()

print("=" * 55)
print("  DATASET CADANGAN PANGAN SULAWESI TENGAH")
print("=" * 55)
print(f"  Jumlah Kab/Kota : {len(df)}")
print(f"  Rata-rata CPPD  : {df['CPPD'].mean():.2f} ton")
print()
print(df[["Kabupaten/Kota", "Kepadatan_Jiwa_per_km2",
          "Jumlah_Bencana", "CPPD"]].to_string(index=False))
print()

# ─────────────────────────────────────────────
# Statistik Deskriptif
# ─────────────────────────────────────────────
cppd = df["CPPD"].values

print("=" * 55)
print("  STATISTIK DESKRIPTIF CPPD")
print("=" * 55)
print(f"  Mean     : {np.mean(cppd):.4f} ton")
print(f"  Median   : {np.median(cppd):.4f} ton")
print(f"  Std Dev  : {np.std(cppd):.4f} ton")
print(f"  Min      : {np.min(cppd):.4f} ton")
print(f"  Max      : {np.max(cppd):.4f} ton")
print(f"  Skewness : {stats.skew(cppd):.4f}")
print(f"  Kurtosis : {stats.kurtosis(cppd):.4f}")

stat_sw, p_sw = stats.shapiro(cppd)
print(f"  Shapiro-Wilk : stat={stat_sw:.4f}, p={p_sw:.4f}", end="  →  ")
print("Normal" if p_sw > 0.05 else "Tidak Normal")
print()

print("KORELASI PEARSON FITUR vs CPPD")
for fitur in ["Kepadatan_Jiwa_per_km2", "Jumlah_Bencana"]:
    r, p = stats.pearsonr(df[fitur], df["CPPD"])
    print(f"  {fitur:<25} r={r:+.4f}  p={p:.4f}", end="  →  ")
    print("Signifikan" if p < 0.05 else "Tidak Signifikan")
print()

# ─────────────────────────────────────────────
# Fitur dan Target
# ─────────────────────────────────────────────
FITUR  = ["Kepadatan_Jiwa_per_km2", "Jumlah_Bencana"]
TARGET = "CPPD"

X = df[FITUR].values
y = df[TARGET].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# Evaluasi Leave-One-Out CV
# (LOO lebih tepat untuk n kecil seperti 13 kab/kota)
# ─────────────────────────────────────────────
loo = LeaveOneOut()

models = {
    "SVR":           SVR(kernel="rbf", C=100, gamma="scale"),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

print("=" * 55)
print("  EVALUASI MODEL — Leave-One-Out Cross Validation")
print("  (LOO dipilih karena n=13, lebih andal dari K-Fold)")
print("  Catatan: R² tidak dihitung di LOO karena setiap")
print("  fold hanya memiliki 1 sampel uji (tidak terdefinisi)")
print("=" * 55)
print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8}")
print("  " + "-" * 40)

results  = {}
best_rmse = 999
best_model_name = None

import warnings
from sklearn.exceptions import UndefinedMetricWarning

for nama, model in models.items():
    X_input = X_scaled if nama == "SVR" else X

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        mae_cv = -cross_val_score(model, X_input, y, cv=loo,
                                   scoring="neg_mean_absolute_error").mean()
        mse_cv = -cross_val_score(model, X_input, y, cv=loo,
                                   scoring="neg_mean_squared_error").mean()
    rmse_cv = np.sqrt(mse_cv)

    results[nama] = {"MAE": mae_cv, "RMSE": rmse_cv}
    print(f"  {nama:<20} {mae_cv:>8.4f} {rmse_cv:>8.4f}")

    if rmse_cv < best_rmse:
        best_rmse       = rmse_cv
        best_model_name = nama

print()
print(f"  ✔ Model Terbaik : {best_model_name}")
print(f"    RMSE LOO-CV   : {best_rmse:.4f} ton")
print()

# ─────────────────────────────────────────────
# Training final dengan seluruh data
# lalu hasilkan prediksi kebutuhan cadangan ideal
# ─────────────────────────────────────────────
best_model = models[best_model_name]
X_input    = X_scaled if best_model_name == "SVR" else X
best_model.fit(X_input, y)
y_pred_raw = best_model.predict(X_input)

# Clip nilai negatif — CPPD tidak mungkin negatif
y_pred = np.maximum(y_pred_raw, 0.0)

mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_score(y, y_pred)

# ─────────────────────────────────────────────
# Tabel hasil prediksi
# ─────────────────────────────────────────────
df_hasil = df[["Kabupaten/Kota",
               "Kepadatan_Jiwa_per_km2",
               "Jumlah_Bencana"]].copy()
df_hasil["Prediksi_CPPD (Ton)"] = np.round(y_pred, 4)

# Klasifikasi berdasarkan nilai bermakna (bukan kuartil mekanis):
# Tinggi  : > 10 ton  (di atas rata-rata kebutuhan)
# Sedang  : 3 – 10 ton
# Rendah  : < 3 ton
def klasifikasi(v):
    if v > 10:  return "Tinggi"
    elif v >= 3: return "Sedang"
    else:        return "Rendah"

df_hasil["Kategori Kebutuhan"] = df_hasil["Prediksi_CPPD (Ton)"].apply(klasifikasi)
df_hasil = df_hasil.sort_values("Prediksi_CPPD (Ton)", ascending=False).reset_index(drop=True)
df_hasil.index += 1  # Ranking mulai dari 1

print("=" * 65)
print("  ESTIMASI KEBUTUHAN CADANGAN PANGAN PER KABUPATEN/KOTA")
print(f"  Model: {best_model_name}  |  MAE={mae:.4f}  |  RMSE={rmse:.4f}  |  R²={r2:.4f}")
print("=" * 65)
print(df_hasil.to_string())
print()

# Feature Importance (Random Forest)
if best_model_name == "Random Forest":
    print("TINGKAT PENGARUH FITUR TERHADAP PREDIKSI CPPD")
    importance = sorted(
        zip(FITUR, best_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for fitur, nilai in importance:
        print(f"  {fitur:<25} {nilai:.4f}  ({nilai*100:.1f}%)")
    print()

# ─────────────────────────────────────────────
# VISUALISASI 1 — Ranking Estimasi Kebutuhan Cadangan
# ─────────────────────────────────────────────
warna_map = {"Tinggi": "red", "Sedang": "orange", "Rendah": "green"}
warna_bar  = [warna_map[k] for k in df_hasil["Kategori Kebutuhan"]]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    df_hasil["Kabupaten/Kota"],
    df_hasil["Prediksi_CPPD (Ton)"],
    color=warna_bar, edgecolor="white", linewidth=0.5
)

# Label nilai di ujung bar
for bar, val in zip(bars, df_hasil["Prediksi_CPPD (Ton)"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} ton", va="center", fontsize=8)

ax.invert_yaxis()
ax.set_xlabel("Estimasi CPPD (Ton)", fontsize=10)
ax.set_title(
    f"Estimasi Kebutuhan Cadangan Pangan per Kab/Kota — Sulawesi Tengah\n"
    f"Model: {best_model_name}  |  R² = {r2:.4f}  |  RMSE = {rmse:.4f} ton",
    fontsize=11, fontweight="bold"
)

# Legend kategori
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=v, label=k) for k, v in warna_map.items()]
ax.legend(handles=legend_els, title="Kategori Kebutuhan",
          loc="lower right", fontsize=9)

ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.set_xlim(left=0)
plt.tight_layout()
plt.show()
