"""Wizualizacja RMSE i R^2 na przykladzie z model.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 200
moc_grzalki_kw = np.random.uniform(5, 20, n)
czas_min = np.random.uniform(10, 120, n)
szum = np.random.normal(0, 1.5, n)
temperatura_c = 20 + 1.8 * moc_grzalki_kw + 0.15 * czas_min + szum

X = pd.DataFrame({"moc_grzalki_kw": moc_grzalki_kw, "czas_min": czas_min})
y = temperatura_c

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Wykres 1: przewidziane vs prawdziwe
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.7, color="#2563eb", edgecolor="white", s=60)
lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
ax.plot(lims, lims, "--", color="gray", label="idealna predykcja (y=x)")
ax.fill_between(lims, [l - rmse for l in lims], [l + rmse for l in lims],
                 color="#2563eb", alpha=0.1, label=f"pasmo ±RMSE ({rmse:.2f}°C)")
ax.set_xlabel("Prawdziwa temperatura (°C)")
ax.set_ylabel("Przewidziana temperatura (°C)")
ax.set_title(f"Predykcja vs rzeczywistość\nR² = {r2:.3f}")
ax.legend()
ax.set_xlim(lims); ax.set_ylim(lims)

# Wykres 2: rozklad bledow (residuals)
ax = axes[1]
ax.hist(residuals, bins=15, color="#2563eb", alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(rmse, color="red", linestyle=":", label=f"+RMSE = {rmse:.2f}")
ax.axvline(-rmse, color="red", linestyle=":", label=f"-RMSE = {-rmse:.2f}")
ax.set_xlabel("Błąd predykcji (°C) = prawdziwa − przewidziana")
ax.set_ylabel("Liczba obserwacji")
ax.set_title("Rozkład błędów (residuals)")
ax.legend()

plt.tight_layout()
plt.savefig("krok1_regresja/rmse_r2_wizualizacja.png", dpi=130)
print("Zapisano: krok1_regresja/rmse_r2_wizualizacja.png")
print(f"RMSE={rmse:.3f}, R2={r2:.3f}")
