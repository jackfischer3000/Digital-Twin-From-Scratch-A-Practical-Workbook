"""
Krok 6: symulacja bioreaktora - replika case study D.1 z ksiazki (predykcja DO).

Fizyka: bilans tlenu rozpuszczonego
    dDO/dt = OTR - OUR
    OTR (oxygen transfer rate) = kLa * (DO_sat - DO)          [korelacja kLa a la Van't Riet]
    OUR (oxygen uptake rate)   = qO2 * X * DO/(DO+K_do)        [zuzycie tlenu przez biomase]

"Ukryty" efekt (ktorego prosty model fizyczny NIE zna): metaboliczny tryb overflow -
gdy skumulowana kombinacja gestosci biomasy i historii karmienia przekroczy prog,
komorki zwiekszaja zuzycie tlenu (qO2 rosnie). To dokladnie mechanizm opisany we
Wstepie ksiazki: "kombinacja gestosci inokulum i historii karmienia" -> spadek DO
z opoznieniem. Wykrycie tego wymaga PAMIECI (stad GRU), nie tylko biezacych wartosci.
"""
import numpy as np
import pandas as pd

DT = 1.0          # min
DO_SAT = 100.0    # % saturacji
XMAX = 40.0       # g/L, pojemnosc nosna
MU_MAX = 0.018    # 1/min, tempo wzrostu biomasy
QO2_BASE = 0.04   # bazowe zuzycie tlenu na gram biomasy
K_DO = 5.0        # stala Monoda dla limitacji tlenowej
C_KLA = 0.08      # wspolczynnik korelacji kLa (a la Van't Riet: kLa ~ moc^a * przeplyw^b)


def kla(stirring_rpm, aeration_vvm):
    return C_KLA * (stirring_rpm / 500.0) ** 2.2 * (aeration_vvm / 1.0) ** 0.5


def krokowy_szereg(n_kroki, poziomy, co_ile):
    """Generuje szereg skokowy (operator co jakis czas zmienia nastawe)."""
    szereg = np.zeros(n_kroki)
    poziom = np.random.uniform(*poziomy)
    for t in range(n_kroki):
        if t % co_ile == 0:
            poziom = np.random.uniform(*poziomy)
        szereg[t] = poziom
    return szereg


def symuluj_batch(n_kroki=480, seed=0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    stirring = krokowy_szereg(n_kroki, (250, 750), co_ile=60)
    aeration = krokowy_szereg(n_kroki, (0.5, 2.0), co_ile=60)
    feed_rate = krokowy_szereg(n_kroki, (0.0, 1.0), co_ile=45)

    X = np.zeros(n_kroki)
    X[0] = np.random.uniform(0.5, 3.0)   # gestosc inokulum - rozna dla kazdego batcha
    DO = np.zeros(n_kroki)
    DO[0] = 90.0

    metabolic_stress = np.zeros(n_kroki)
    metabolic_mode = np.zeros(n_kroki)
    prog = 55.0

    szum = rng.normal(0, 0.15, n_kroki)

    for t in range(1, n_kroki):
        # wzrost biomasy (logistyczny)
        dX = MU_MAX * X[t - 1] * (1 - X[t - 1] / XMAX) * DT
        X[t] = X[t - 1] + dX

        # akumulacja "stresu metabolicznego" - kombinacja biomasy i karmienia
        metabolic_stress[t] = metabolic_stress[t - 1] * 0.985 + X[t] * feed_rate[t] * 0.06
        metabolic_mode[t] = 1.0 if metabolic_stress[t] > prog else 0.0

        qo2_eff = QO2_BASE * (1.7 if metabolic_mode[t] else 1.0)

        otr = kla(stirring[t - 1], aeration[t - 1]) * (DO_SAT - DO[t - 1])
        our = qo2_eff * X[t] * DO[t - 1] / (DO[t - 1] + K_DO)
        DO[t] = np.clip(DO[t - 1] + DT * (otr - our) + szum[t], 0.0, 100.0)

    return pd.DataFrame({
        "t": np.arange(n_kroki),
        "stirring_rpm": stirring,
        "aeration_vvm": aeration,
        "feed_rate": feed_rate,
        "X_biomasa": X,
        "DO": DO,
        "metabolic_mode": metabolic_mode,   # tylko do wizualizacji/diagnostyki - NIE jest wejsciem modeli
    })


if __name__ == "__main__":
    # szybki test kalibracyjny
    df = symuluj_batch(seed=1)
    print(df.describe())
    print(f"\nDO min={df['DO'].min():.1f}, max={df['DO'].max():.1f}")
    print(f"Ulamek czasu w trybie metabolicznym: {df['metabolic_mode'].mean()*100:.1f}%")
