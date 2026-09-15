"""
Krok 4: przeuczenie (overfitting) i walidacja krzyzowa.

Klasyczny przyklad: dopasowujemy wielomiany rosnacego stopnia do MALEGO, zaszumionego
zbioru danych. Niski stopien = zbyt prosty (underfitting). Wysoki stopien = model
"zapamietuje" szum zamiast uczyc sie prawdziwej zaleznosci (overfitting).
"""
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(7)
n = 25   # CELOWO malo danych - overfitting jest dramatyczniejszy i latwiejszy do zobaczenia
moc = np.random.uniform(5, 20, n)
szum = np.random.normal(0, 1.2, n)
temperatura = 20 + 15 * np.tanh(moc / 8) + szum

X = moc.reshape(-1, 1)
y = temperatura

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

stopnie = [1, 2, 3, 5, 9, 15]

print(f"{'Stopien wielomianu':<20} {'RMSE (train)':<15} {'RMSE (test)':<15}")
print("-" * 50)

wyniki = []
for stopien in stopnie:
    model = make_pipeline(
        PolynomialFeatures(degree=stopien),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(X_train, y_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    wyniki.append((stopien, rmse_train, rmse_test))
    print(f"{stopien:<20} {rmse_train:<15.3f} {rmse_test:<15.3f}")

print("\nObserwacja: RMSE na TRENINGU spada niemal do zera przy wysokim stopniu -")
print("model idealnie 'zapamietuje' 17 punktow treningowych. RMSE na TESCIE rosnie -")
print("model nie generalizuje sie na nowe dane. To jest overfitting.")

# ============================================================
# PROBLEM z pojedynczym podzialem train/test: dla 25 punktow, wynik testowy
# mocno zalezy od tego, KTORE punkty trafily do testu (losowosc podzialu).
# Sprawdzmy to - zmieniamy tylko random_state podzialu.
# ============================================================
print("\n=== Jak niestabilny jest wynik pojedynczego train/test split? ===")
print("(ten sam model, stopien=3, rozne losowe podzialy)\n")
for rs in [1, 2, 3, 4, 5]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=rs)
    model = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
    model.fit(Xtr, ytr)
    rmse = np.sqrt(mean_squared_error(yte, model.predict(Xte)))
    print(f"random_state={rs}: RMSE testowe = {rmse:.3f}")

# ============================================================
# WALIDACJA KRZYZOWA (k-fold cross-validation): zamiast JEDNEGO podzialu,
# dzielimy dane na k czesci, kazda po kolei jest zbiorem testowym, reszta treningowym.
# Usredniamy k wynikow -> stabilniejsza, wiarygodniejsza ocena.
# ============================================================
print("\n=== Walidacja krzyzowa (5-fold) dla kazdego stopnia wielomianu ===\n")
print(f"{'Stopien':<10} {'CV RMSE (srednia)':<20} {'CV RMSE (odch. std.)':<20}")
print("-" * 50)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_wyniki = []
for stopien in stopnie:
    model = make_pipeline(PolynomialFeatures(degree=stopien), StandardScaler(), LinearRegression())
    # scikit-learn liczy scoring jako "neg_mean_squared_error" (wieksze=lepsze), stad minus
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_rmse = -scores
    cv_wyniki.append((stopien, cv_rmse.mean(), cv_rmse.std()))
    print(f"{stopien:<10} {cv_rmse.mean():<20.3f} {cv_rmse.std():<20.3f}")

najlepszy = min(cv_wyniki, key=lambda w: w[1])
print(f"\nNajlepszy stopien wg CV: {najlepszy[0]} (CV RMSE = {najlepszy[1]:.3f})")
print("To ta liczba - nie wynik na pojedynczym tescie - powinna decydowac o zlozonosci modelu.")
