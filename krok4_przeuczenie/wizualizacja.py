"""Wizualizacja: underfitting vs dobre dopasowanie vs overfitting + krzywa train/test RMSE."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(7)
n = 25
moc = np.random.uniform(5, 20, n)
szum = np.random.normal(0, 1.2, n)
temperatura = 20 + 15 * np.tanh(moc / 8) + szum
X = moc.reshape(-1, 1)
y = temperatura
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel 1: dopasowane krzywe dla roznych stopni ---
ax = axes[0]
moc_siatka = np.linspace(4, 21, 300).reshape(-1, 1)
kolory = {1: "#f59e0b", 2: "#16a34a", 15: "#dc2626"}
etykiety = {1: "stopień 1 (underfitting)", 2: "stopień 2 (dobre dopasowanie)", 15: "stopień 15 (overfitting)"}

for stopien in [1, 2, 15]:
    model = make_pipeline(PolynomialFeatures(degree=stopien), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    pred_siatka = model.predict(moc_siatka)
    ax.plot(moc_siatka, pred_siatka, color=kolory[stopien], linewidth=2, label=etykiety[stopien])

ax.scatter(X_train, y_train, color="#1e293b", s=50, zorder=5, label="dane treningowe")
ax.scatter(X_test, y_test, color="#1e293b", s=50, marker="x", zorder=5, label="dane testowe")
ax.set_xlabel("Moc grzałki (kW)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Ta sama chmura punktów, trzy stopnie dopasowania")
ax.legend(fontsize=8)
ax.set_ylim(15, 45)

# --- Panel 2: RMSE train/test/CV vs stopien ---
ax = axes[1]
stopnie = [1, 2, 3, 5, 9, 15]
rmse_train, rmse_test, cv_mean, cv_std = [], [], [], []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for stopien in stopnie:
    model = make_pipeline(PolynomialFeatures(degree=stopien), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    rmse_train.append(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    rmse_test.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    model_cv = make_pipeline(PolynomialFeatures(degree=stopien), StandardScaler(), LinearRegression())
    scores = -cross_val_score(model_cv, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_mean.append(scores.mean())
    cv_std.append(scores.std())

ax.plot(stopnie, rmse_train, "o-", color="#2563eb", label="RMSE (trening)")
ax.plot(stopnie, rmse_test, "o-", color="#dc2626", label="RMSE (pojedynczy test)")
ax.errorbar(stopnie, cv_mean, yerr=cv_std, fmt="o-", color="#16a34a", capsize=4,
            label="RMSE (5-fold CV, ±std)")
ax.set_xlabel("Stopień wielomianu (złożoność modelu)")
ax.set_ylabel("RMSE (°C)")
ax.set_title("Trening vs test vs walidacja krzyżowa")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("krok4_przeuczenie/overfitting.png", dpi=130, bbox_inches="tight")
print("Zapisano: krok4_przeuczenie/overfitting.png")
