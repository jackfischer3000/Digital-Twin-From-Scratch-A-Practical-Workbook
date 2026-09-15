"""
Krok 5: dlaczego proces z pamiecia (bezwladnoscia) wymaga modelu z pamiecia.

Symulujemy reaktor z bezwladnoscia cieplna (uklad pierwszego rzedu):
    T[t+1] = T[t] + dt * k * (T_docelowa(moc[t]) - T[t]) + szum

Kluczowa wlasnosc: ta sama moc chwilowa moze odpowiadac roznym temperaturom,
w zaleznosci od HISTORII (jak dlugo grzalka dziala na danym poziomie).
Model bez pamieci (widzi tylko biezaca moc) nie ma jak tego odroznic.
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# --- Symulacja procesu z bezwladnoscia (kroki co 1 minute, 400 minut) ---
n_kroki = 400
dt = 1.0
k = 0.08   # stala czasowa ukladu - jak szybko T "dogania" wartosc docelowa

def t_docelowa(moc):
    return 20 + 15 * np.tanh(moc / 8)   # ta sama fizyka nasycenia co wczesniej

# Moc zmienia sie skokowo co ok. 40 minut (typowe dla operatora zmieniajacego nastawe)
moc_szereg = np.zeros(n_kroki)
poziom = np.random.uniform(5, 20)
for t in range(n_kroki):
    if t % 40 == 0:
        poziom = np.random.uniform(5, 20)
    moc_szereg[t] = poziom

T = np.zeros(n_kroki)
T[0] = 20.0
szum = np.random.normal(0, 0.3, n_kroki)
for t in range(1, n_kroki):
    T[t] = T[t - 1] + dt * k * (t_docelowa(moc_szereg[t - 1]) - T[t - 1]) + szum[t]

dane = pd.DataFrame({"czas_krok": np.arange(n_kroki), "moc": moc_szereg, "T_prawdziwa": T})

# Podzial: pierwsze 300 krokow trening, ostatnie 100 test (typowe dla szeregow czasowych -
# NIE losowy split, bo to zafalszowaloby ocene: model nie moze "widziec przyszlosci")
podzial = 300
train = dane.iloc[:podzial]
test = dane.iloc[podzial:]

# ============================================================
# MODEL A: BEZ PAMIECI - widzi tylko biezaca moc, przewiduje biezaca T
# ============================================================
scaler_a = StandardScaler()
Xtr_a = scaler_a.fit_transform(train[["moc"]])
Xte_a = scaler_a.transform(test[["moc"]])
model_a = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                        max_iter=5000, random_state=42).fit(Xtr_a, train["T_prawdziwa"])
pred_a = model_a.predict(Xte_a)
rmse_a = np.sqrt(mean_squared_error(test["T_prawdziwa"], pred_a))

# ============================================================
# MODEL B: Z PAMIECIA - widzi biezaca moc ORAZ T z poprzedniego kroku
# To jest jadro tego, co robi kazda siec rekurencyjna (RNN/GRU): stan z poprzedniego
# kroku wraca jako wejscie do nastepnego. GRU dodaje do tego "bramki" (gates), ktore
# ucza sie, ile z poprzedniego stanu zachowac a ile nadpisac - ale mechanizm bazowy
# jest dokladnie taki jak tutaj.
# ============================================================
dane["T_poprzednia"] = dane["T_prawdziwa"].shift(1)
dane_b = dane.dropna()
train_b = dane_b.iloc[:podzial - 1]
test_b = dane_b.iloc[podzial - 1:]

scaler_b = StandardScaler()
Xtr_b = scaler_b.fit_transform(train_b[["moc", "T_poprzednia"]])
Xte_b = scaler_b.transform(test_b[["moc", "T_poprzednia"]])
model_b = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                        max_iter=5000, random_state=42).fit(Xtr_b, train_b["T_prawdziwa"])

# WAZNE: w prawdziwej predykcji "na zywo" nie masz prawdziwej T_poprzednia z przyszlosci -
# model musi uzywac WLASNEJ poprzedniej predykcji (predykcja rekurencyjna, krok po kroku).
pred_b = []
T_poprzednia_symulowana = test_b["T_poprzednia"].iloc[0]
for _, wiersz in test_b.iterrows():
    x_df = pd.DataFrame([[wiersz["moc"], T_poprzednia_symulowana]], columns=["moc", "T_poprzednia"])
    x = scaler_b.transform(x_df)
    p = model_b.predict(x)[0]
    pred_b.append(p)
    T_poprzednia_symulowana = p   # wlasna predykcja staje sie "pamiecia" na kolejny krok
pred_b = np.array(pred_b)
rmse_b = np.sqrt(mean_squared_error(test_b["T_prawdziwa"], pred_b))

print("=== PORONANIE: model bez pamieci vs model z pamiecia ===\n")
print(f"Model A (bez pamieci, tylko biezaca moc):  RMSE = {rmse_a:.3f} C")
print(f"Model B (z pamiecia, wlasna T_poprzednia):  RMSE = {rmse_b:.3f} C")
print(f"\nPoprawa: {(1 - rmse_b/rmse_a)*100:.1f}% mniejszy RMSE dzieki dodaniu pamieci")

# zapisz do pliku dla wizualizacji
wyniki = test.copy()
wyniki["pred_A_bez_pamieci"] = pred_a
wyniki_b = test_b.copy()
wyniki_b["pred_B_z_pamiecia"] = pred_b
wyniki.to_csv("krok5_pamiec/wyniki_A.csv", index=False)
wyniki_b.to_csv("krok5_pamiec/wyniki_B.csv", index=False)
