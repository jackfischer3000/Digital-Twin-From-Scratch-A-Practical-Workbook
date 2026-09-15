"""
Krok 1: najprostszy kompletny model ML — regresja liniowa.
Cel: przejsc caly cykl (dane -> trening -> walidacja -> interpretacja) raz, na prostym przykladzie.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. DANE ---
# W realnym projekcie dane pochodza z historiana/SCADA (patrz Rozdzial 5 ksiazki: "Dane w srodowisku GxP").
# Tutaj: generujemy syntetycznie, zeby miec kontrole nad "prawdziwa" zaleznoscia i sprawdzic,
# czy model jest w stanie ja odtworzyc.
np.random.seed(42)
n = 200

moc_grzalki_kw = np.random.uniform(5, 20, n)      # kW
czas_min = np.random.uniform(10, 120, n)          # minuty

# "Prawdziwa" fizyka (uproszczona): temperatura rosnie z moca i czasem, plus szum pomiarowy
szum = np.random.normal(0, 1.5, n)
temperatura_c = 20 + 1.8 * moc_grzalki_kw + 0.15 * czas_min + szum

dane = pd.DataFrame({
    "moc_grzalki_kw": moc_grzalki_kw,
    "czas_min": czas_min,
    "temperatura_c": temperatura_c,
})

print("Pierwsze 5 wierszy danych:")
print(dane.head())
print(f"\nLiczba obserwacji: {len(dane)}")

# --- 2. PODZIAL TRAIN/TEST ---
# Nigdy nie oceniamy modelu na danych, na ktorych sie uczyl - to podstawowa zasada (odpowiednik "OQ" w ksiazce).
X = dane[["moc_grzalki_kw", "czas_min"]]
y = dane["temperatura_c"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nZbior treningowy: {len(X_train)} obserwacji, testowy: {len(X_test)} obserwacji")

# --- 3. TRENING (odpowiednik Rozdzialu 9 - "budowa i trening modelu") ---
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nWspolczynniki modelu:")
print(f"  moc_grzalki_kw: {model.coef_[0]:.3f}  (prawdziwa wartosc: 1.8)")
print(f"  czas_min:       {model.coef_[1]:.3f}  (prawdziwa wartosc: 0.15)")
print(f"  wyraz wolny:    {model.intercept_:.3f}  (prawdziwa wartosc: 20)")

# --- 4. WALIDACJA (odpowiednik Rozdzialu 10 - "walidacja modelu / OQ") ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Wyniki na zbiorze TESTOWYM (dane, ktorych model nie widzial) ---")
print(f"RMSE: {rmse:.3f} C")
print(f"R^2:  {r2:.3f}  (1.0 = idealne dopasowanie)")

# --- 5. INTERPRETACJA ---
print("\n--- Przyklad predykcji ---")
przyklad = X_test.iloc[[0]]
prawdziwa = y_test.iloc[0]
predykcja = model.predict(przyklad)[0]
print(f"Wejscie: moc={przyklad['moc_grzalki_kw'].values[0]:.1f} kW, czas={przyklad['czas_min'].values[0]:.1f} min")
print(f"Prawdziwa temperatura: {prawdziwa:.2f} C")
print(f"Predykcja modelu:      {predykcja:.2f} C")
