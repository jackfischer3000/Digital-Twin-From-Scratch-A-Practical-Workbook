"""Wizualizacja: jak korekta ML usuwa systematyczny blad modelu fizycznego."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200
moc_grzalki_kw = np.random.uniform(5, 20, n)
czas_min = np.random.uniform(10, 120, n)
temp_otoczenia_c = np.random.uniform(15, 30, n)
szum = np.random.normal(0, 1.5, n)
temperatura_c = (20 + 15 * np.tanh(moc_grzalki_kw / 8) + 0.15 * czas_min
                  - 0.3 * (25 - temp_otoczenia_c) + szum)

dane = pd.DataFrame({"moc_grzalki_kw": moc_grzalki_kw, "czas_min": czas_min,
                      "temp_otoczenia_c": temp_otoczenia_c})
y = temperatura_c
X_train, X_test, y_train, y_test = train_test_split(dane, y, test_size=0.2, random_state=42)

def model_fizyczny(moc, czas):
    return 20 + 15 * np.tanh(moc / 8) + 0.15 * czas

pred_fizyka_train = model_fizyczny(X_train["moc_grzalki_kw"], X_train["czas_min"])
pred_fizyka_test = model_fizyczny(X_test["moc_grzalki_kw"], X_test["czas_min"])
rezydua_train = y_train - pred_fizyka_train
rezydua_test_przed = y_test - pred_fizyka_test

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)
model_korekta = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                              max_iter=5000, random_state=42).fit(Xtr_s, rezydua_train)
korekta_test = model_korekta.predict(Xte_s)
rezydua_test_po = rezydua_test_przed - korekta_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax = axes[0]
ax.scatter(X_test["temp_otoczenia_c"], rezydua_test_przed, alpha=0.7, color="#dc2626", edgecolor="white", s=60)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
z = np.polyfit(X_test["temp_otoczenia_c"], rezydua_test_przed, 1)
xs = np.linspace(15, 30, 50)
ax.plot(xs, np.polyval(z, xs), color="#dc2626", linewidth=2, alpha=0.5, label="trend")
ax.set_xlabel("Temperatura otoczenia (°C)")
ax.set_ylabel("Rezyduum = prawdziwa − przewidziana (°C)")
ax.set_title("PRZED korektą (sama fizyka)\nwyraźny trend = efekt, którego fizyka nie zna")
ax.legend()

ax = axes[1]
ax.scatter(X_test["temp_otoczenia_c"], rezydua_test_po, alpha=0.7, color="#16a34a", edgecolor="white", s=60)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Temperatura otoczenia (°C)")
ax.set_title("PO korekcie ML\ntrend zniknął — zostaje tylko szum")

plt.suptitle("Rezydua fizyki w funkcji temperatury otoczenia — przed i po korekcie ML", y=1.03)
plt.tight_layout()
plt.savefig("krok3_hybrydowy/korekta_rezyduow.png", dpi=130, bbox_inches="tight")
print("Zapisano: krok3_hybrydowy/korekta_rezyduow.png")
