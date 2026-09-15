"""
Krok 2: nieliniowosc - gdzie regresja liniowa zawodzi, i pierwsza siec neuronowa.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 200
moc_grzalki_kw = np.random.uniform(5, 20, n)
czas_min = np.random.uniform(10, 120, n)
szum = np.random.normal(0, 1.5, n)

# Tym razem: NASYCENIE przy wysokiej mocy (tanh) zamiast prostej zaleznosci liniowej.
# Fizyczna interpretacja: powyzej pewnej mocy przyrost temperatury spowalnia (straty rosna, chlodzenie wchodzi w gre).
temperatura_c = 20 + 15 * np.tanh(moc_grzalki_kw / 8) + 0.15 * czas_min + szum

X = pd.DataFrame({"moc_grzalki_kw": moc_grzalki_kw, "czas_min": czas_min})
y = temperatura_c

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL A: regresja liniowa (jak w kroku 1) ---
model_liniowy = LinearRegression().fit(X_train, y_train)
pred_liniowy = model_liniowy.predict(X_test)
rmse_liniowy = np.sqrt(mean_squared_error(y_test, pred_liniowy))
r2_liniowy = r2_score(y_test, pred_liniowy)

print("=== MODEL A: regresja liniowa ===")
print(f"RMSE: {rmse_liniowy:.3f} C")
print(f"R^2:  {r2_liniowy:.3f}")

# --- MODEL B: mala siec neuronowa (1 warstwa ukryta, 8 neuronow) ---
# MLP = Multi-Layer Perceptron. Kazdy neuron liczy: wazona suma wejsc -> funkcja aktywacji (tu: relu/tanh).
# To ta funkcja aktywacji wprowadza NIELINIOWOSC - bez niej siec zlozona z samych warstw liniowych
# nadal bylaby tylko regresja liniowa, niezaleznie od liczby warstw.
#
# WAZNE: siec neuronowa wymaga PRZESKALOWANYCH cech (srednia=0, odchylenie=1).
# Bez tego, przy roznych zakresach wejsc (moc: 5-20, czas: 10-120), trening gradientowy
# jest niestabilny i moze "eksplodowac" (overflow). Regresja liniowa nie ma tego problemu.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit TYLKO na treningowych - test nie moze "wyciekac" do skalowania
X_test_scaled = scaler.transform(X_test)

# solver="lbfgs" (nie domyslny "adam"): adam/sgd to metody stochastyczne zaprojektowane
# pod duze zbiory danych (uczenie na mini-batchach). Dla malego zbioru (160 obserwacji)
# lbfgs (metoda quasi-Newtona, pelny batch) zbiega szybciej i stabilniej - to oficjalna
# rekomendacja z dokumentacji scikit-learn dla malych danych.
model_nn = MLPRegressor(
    hidden_layer_sizes=(8,),   # jedna warstwa ukryta, 8 neuronow
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=42,
)
model_nn.fit(X_train_scaled, y_train)
pred_nn = model_nn.predict(X_test_scaled)
rmse_nn = np.sqrt(mean_squared_error(y_test, pred_nn))
r2_nn = r2_score(y_test, pred_nn)

print("\n=== MODEL B: siec neuronowa (MLP, 1 warstwa, 8 neuronow) ===")
print(f"RMSE: {rmse_nn:.3f} C")
print(f"R^2:  {r2_nn:.3f}")

print(f"\n=== PORONANIE ===")
print(f"Regresja liniowa: RMSE={rmse_liniowy:.3f}, R^2={r2_liniowy:.3f}")
print(f"Siec neuronowa:   RMSE={rmse_nn:.3f}, R^2={r2_nn:.3f}")
print(f"Poprawa RMSE: {(1 - rmse_nn/rmse_liniowy)*100:.1f}%")
