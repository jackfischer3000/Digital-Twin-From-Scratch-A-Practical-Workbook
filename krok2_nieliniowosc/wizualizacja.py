"""Wizualizacja: gdzie regresja liniowa systematycznie sie myli, a siec NN - nie."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200
moc_grzalki_kw = np.random.uniform(5, 20, n)
czas_min = np.random.uniform(10, 120, n)
szum = np.random.normal(0, 1.5, n)
temperatura_c = 20 + 15 * np.tanh(moc_grzalki_kw / 8) + 0.15 * czas_min + szum

X = pd.DataFrame({"moc_grzalki_kw": moc_grzalki_kw, "czas_min": czas_min})
y = temperatura_c
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_liniowy = LinearRegression().fit(X_train, y_train)
pred_liniowy = model_liniowy.predict(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model_nn = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                         max_iter=5000, random_state=42).fit(X_train_s, y_train)
pred_nn = model_nn.predict(X_test_s)

resid_lin = y_test - pred_liniowy
resid_nn = y_test - pred_nn
moc_test = X_test["moc_grzalki_kw"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, resid, tytul in [(axes[0], resid_lin, "Regresja liniowa"), (axes[1], resid_nn, "Sieć neuronowa (MLP)")]:
    ax.scatter(moc_test, resid, alpha=0.7, color="#2563eb", edgecolor="white", s=60)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Moc grzałki (kW)")
    ax.set_title(tytul)
axes[0].set_ylabel("Błąd (residual) = prawdziwa − przewidziana (°C)")

plt.suptitle("Błąd predykcji w funkcji mocy grzałki — szukamy WZORU w punktach, nie chmury losowej", y=1.02)
plt.tight_layout()
plt.savefig("krok2_nieliniowosc/residua_porownanie.png", dpi=130, bbox_inches="tight")
print("Zapisano: krok2_nieliniowosc/residua_porownanie.png")
