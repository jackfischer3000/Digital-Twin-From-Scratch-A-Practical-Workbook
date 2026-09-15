"""
Krok 3: model hybrydowy (fizyka + korekta ML) - jak w case study bioreaktora (Rozdzial 6, Aneks D.1).

Architektura: predykcja_hybrydowa = predykcja_fizyki + korekta_ML(rezydua)
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
temp_otoczenia_c = np.random.uniform(15, 30, n)   # NOWA zmienna - fizyka jej nie zna
szum = np.random.normal(0, 1.5, n)

# "PRAWDA" (rzeczywisty proces): nasycenie mocy + czas + STRATY CIEPLA do otoczenia
# (im chlodniejsze otoczenie wzgledem 25C referencyjnych, tym wieksza strata -> nizsza temperatura)
temperatura_c = (
    20 + 15 * np.tanh(moc_grzalki_kw / 8) + 0.15 * czas_min
    - 0.3 * (25 - temp_otoczenia_c)   # ten czlon jest "niewidoczny" dla naszej fizyki
    + szum
)

dane = pd.DataFrame({
    "moc_grzalki_kw": moc_grzalki_kw,
    "czas_min": czas_min,
    "temp_otoczenia_c": temp_otoczenia_c,
})
y = temperatura_c

X_train, X_test, y_train, y_test = train_test_split(dane, y, test_size=0.2, random_state=42)

# ============================================================
# PODEJSCIE A: TYLKO FIZYKA (uproszczony model mechanistyczny, NIE zna temp_otoczenia)
# ============================================================
def model_fizyczny(moc, czas):
    return 20 + 15 * np.tanh(moc / 8) + 0.15 * czas

pred_fizyka_test = model_fizyczny(X_test["moc_grzalki_kw"], X_test["czas_min"])
rmse_fizyka = np.sqrt(mean_squared_error(y_test, pred_fizyka_test))
r2_fizyka = r2_score(y_test, pred_fizyka_test)

# ============================================================
# PODEJSCIE B: TYLKO ML - "czarna skrzynka" trenowana na surowym sygnale, ze wszystkimi cechami
# ============================================================
scaler_b = StandardScaler()
Xtr_b = scaler_b.fit_transform(X_train)
Xte_b = scaler_b.transform(X_test)
model_ml_only = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                              max_iter=5000, random_state=42).fit(Xtr_b, y_train)
pred_ml_only = model_ml_only.predict(Xte_b)
rmse_ml_only = np.sqrt(mean_squared_error(y_test, pred_ml_only))
r2_ml_only = r2_score(y_test, pred_ml_only)

# ============================================================
# PODEJSCIE C: HYBRYDA = fizyka + ML na rezyduach
# ============================================================
# Krok 1: policz predykcje fizyki na zbiorze treningowym
pred_fizyka_train = model_fizyczny(X_train["moc_grzalki_kw"], X_train["czas_min"])

# Krok 2: policz rezydua = to, czego fizyka NIE wyjasnia
rezydua_train = y_train - pred_fizyka_train

# Krok 3: wytrenuj MALY model ML, zeby przewidywal rezydua (nie surowa temperature!)
# Uzywamy WSZYSTKICH cech, w tym temp_otoczenia - ML ma szanse odkryc brakujacy efekt
scaler_c = StandardScaler()
Xtr_c = scaler_c.fit_transform(X_train)
Xte_c = scaler_c.transform(X_test)
model_korekta = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                              max_iter=5000, random_state=42).fit(Xtr_c, rezydua_train)

# Krok 4: polacz - hybryda = fizyka + korekta ML
korekta_test = model_korekta.predict(Xte_c)
pred_hybryda = pred_fizyka_test + korekta_test
rmse_hybryda = np.sqrt(mean_squared_error(y_test, pred_hybryda))
r2_hybryda = r2_score(y_test, pred_hybryda)

# ============================================================
# PORONANIE (jak tabela benchmarkowa w Aneksie D.1 ksiazki)
# ============================================================
print("=== PORONANIE TRZECH PODEJSC (na zbiorze TESTOWYM) ===\n")
print(f"{'Podejscie':<30} {'RMSE (C)':<12} {'R^2':<10}")
print(f"{'-'*52}")
print(f"{'A: tylko fizyka':<30} {rmse_fizyka:<12.3f} {r2_fizyka:<10.3f}")
print(f"{'B: tylko ML (czarna skrzynka)':<30} {rmse_ml_only:<12.3f} {r2_ml_only:<10.3f}")
print(f"{'C: hybryda (fizyka+korekta ML)':<30} {rmse_hybryda:<12.3f} {r2_hybryda:<10.3f}")

print(f"\nHybryda vs sama fizyka: {(1 - rmse_hybryda/rmse_fizyka)*100:.1f}% mniejszy RMSE")
print(f"Hybryda vs czysty ML:   {(1 - rmse_hybryda/rmse_ml_only)*100:.1f}% mniejszy RMSE")
