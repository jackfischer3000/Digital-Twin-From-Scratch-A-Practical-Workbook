"""
Krok 6: replika case study D.1 - predykcja DO w bioreaktorze, architektura hybrydowa
(bilans masy + GRU), PyTorch. Benchmark: fizyka vs GRU (czarna skrzynka) vs hybryda,
na horyzoncie 30 minut - jak w Rozdziale 10/11 i Aneksie D.1 ksiazki.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from symulacja import symuluj_batch, kla, QO2_BASE, K_DO, DO_SAT, DT

torch.manual_seed(0)

W = 60   # okno historii (minuty) podawane do GRU
H = 30   # horyzont predykcji (minuty) - jak w benchmarku ksiazki

CECHY = ["stirring_rpm", "aeration_vvm", "feed_rate", "X_biomasa", "DO"]

# ============================================================
# 1. DANE: 16 batchy (symulowane), 12 trening / 4 test
# ============================================================
batche = [symuluj_batch(seed=s) for s in range(16)]
batche_train = batche[:12]
batche_test = batche[12:]

# normalizacja - liczona TYLKO na treningowych (jak w kroku 1-2, zeby nie "wyciekac" danych testowych)
wszystkie_train = pd.concat(batche_train)
srednia = wszystkie_train[CECHY].mean()
odchylenie = wszystkie_train[CECHY].std()


def normalizuj(df):
    return (df[CECHY] - srednia) / odchylenie


# DO (target) tez trzeba wystandaryzowac dla modelu "surowego" - bez tego trening
# jest niestabilny (te same przyczyny co w kroku 2: skala wejsc/wyjsc dla sieci).
# Rezydua sa naturalnie mniejsze/blizej zera, wiec dla modelu hybrydowego to mniej krytyczne,
# ale robimy to konsekwentnie dla obu.
DO_SREDNIA = wszystkie_train["DO"].mean()
DO_ODCHYLENIE = wszystkie_train["DO"].std()


# ============================================================
# 2. MODEL FIZYCZNY (niekompletny - NIE zna metabolicznego trybu overflow)
#    Rollout na H krokow do przodu, zakladajac znany harmonogram sterowania (stirring/aeration/feed)
#    - to standardowe zalozenie: harmonogram jest czescia receptury procesowej, wiec jest znany z gory.
# ============================================================
def fizyka_rollout(df, t0, horyzont=H):
    do = df["DO"].iloc[t0]
    for t in range(t0 + 1, t0 + horyzont + 1):
        x = df["X_biomasa"].iloc[t]
        otr = kla(df["stirring_rpm"].iloc[t - 1], df["aeration_vvm"].iloc[t - 1]) * (DO_SAT - do)
        our = QO2_BASE * x * do / (do + K_DO)   # QO2_BASE stale - fizyka NIE wie o trybie metabolicznym
        do = np.clip(do + DT * (otr - our), 0.0, 100.0)
    return do


# ============================================================
# 3. DATASET DLA GRU: okno [t0-W+1, t0] -> cel w t0+H
# ============================================================
class OknaDataset(Dataset):
    def __init__(self, batche, tryb):
        okna, cele_surowe = [], []
        for df in batche:
            df_norm = normalizuj(df)
            fizyka_cache = {}
            for t0 in range(W - 1, len(df) - H - 1):
                okno = df_norm.iloc[t0 - W + 1: t0 + 1].values.astype(np.float32)
                prawdziwa_do = df["DO"].iloc[t0 + H]
                if tryb == "resztkowy":
                    if t0 not in fizyka_cache:
                        fizyka_cache[t0] = fizyka_rollout(df, t0)
                    cel = prawdziwa_do - fizyka_cache[t0]      # GRU uczy sie REZYDUUM
                else:
                    cel = prawdziwa_do                          # GRU uczy sie surowej wartosci (czarna skrzynka)
                okna.append(okno)
                cele_surowe.append(cel)

        cele_surowe = np.array(cele_surowe, dtype=np.float32)
        # standaryzacja celu - ta sama logika co w kroku 2: bez tego trening sieci jest niestabilny
        self.cel_srednia = float(cele_surowe.mean())
        self.cel_odchylenie = float(cele_surowe.std()) + 1e-6
        self.okna = okna
        self.cele = (cele_surowe - self.cel_srednia) / self.cel_odchylenie

    def __len__(self):
        return len(self.okna)

    def __getitem__(self, idx):
        return torch.from_numpy(self.okna[idx]), torch.tensor(self.cele[idx])


# ============================================================
# 4. MODEL GRU (PyTorch) - male, jedna warstwa, do regresji jednej wartosci
# ============================================================
class GRUPredyktor(nn.Module):
    def __init__(self, n_cech=len(CECHY), hidden=24):
        super().__init__()
        self.gru = nn.GRU(input_size=n_cech, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h_ostatni = self.gru(x)         # h_ostatni: stan ukryty po calej sekwencji
        return self.head(h_ostatni.squeeze(0)).squeeze(-1)


def trenuj(model, dataset, epoki=15, batch_size=64, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    strata_fn = nn.MSELoss()
    for epoka in range(epoki):
        model.train()
        strata_suma = 0.0
        for okno, cel in loader:
            opt.zero_grad()
            pred = model(okno)
            strata = strata_fn(pred, cel)
            strata.backward()
            opt.step()
            strata_suma += strata.item() * len(cel)
        if (epoka + 1) % 5 == 0:
            print(f"  epoka {epoka+1}/{epoki}: MSE treningowe = {strata_suma/len(dataset):.3f}")
    return model


print("=== Trening modelu B: GRU czarna skrzynka (surowa wartosc DO) ===")
ds_train_bb = OknaDataset(batche_train, tryb="surowy")
model_bb = trenuj(GRUPredyktor(), ds_train_bb)

print("\n=== Trening modelu C: GRU na rezyduach (hybryda) ===")
ds_train_res = OknaDataset(batche_train, tryb="resztkowy")
model_res = trenuj(GRUPredyktor(), ds_train_res)


def odstandaryzuj(wartosc_znorm, dataset):
    return wartosc_znorm * dataset.cel_odchylenie + dataset.cel_srednia


# ============================================================
# 5. EWALUACJA na zbiorze TESTOWYM (batche, ktorych zaden model nie widzial)
# ============================================================
def ewaluuj(batche_test):
    bledy_fizyka, bledy_bb, bledy_hybryda = [], [], []
    model_bb.eval()
    model_res.eval()
    with torch.no_grad():
        for df in batche_test:
            df_norm = normalizuj(df)
            for t0 in range(W - 1, len(df) - H - 1):
                okno = torch.from_numpy(df_norm.iloc[t0 - W + 1: t0 + 1].values.astype(np.float32)).unsqueeze(0)
                prawdziwa = df["DO"].iloc[t0 + H]

                pred_fizyka = fizyka_rollout(df, t0)
                pred_bb = odstandaryzuj(model_bb(okno).item(), ds_train_bb)
                korekta = odstandaryzuj(model_res(okno).item(), ds_train_res)
                pred_hybryda = pred_fizyka + korekta

                bledy_fizyka.append(prawdziwa - pred_fizyka)
                bledy_bb.append(prawdziwa - pred_bb)
                bledy_hybryda.append(prawdziwa - pred_hybryda)

    rmse = lambda b: float(np.sqrt(np.mean(np.square(b))))
    return rmse(bledy_fizyka), rmse(bledy_bb), rmse(bledy_hybryda)


rmse_fizyka, rmse_bb, rmse_hybryda = ewaluuj(batche_test)

print(f"\n=== BENCHMARK (horyzont {H} min, zbior testowy: {len(batche_test)} batchy) ===\n")
print(f"{'Podejscie':<35} {'RMSE (%DO sat.)':<18}")
print("-" * 53)
print(f"{'A: tylko fizyka (bez efektu metabolicznego)':<35} {rmse_fizyka:<18.3f}")
print(f"{'B: tylko GRU (czarna skrzynka)':<35} {rmse_bb:<18.3f}")
print(f"{'C: hybryda (fizyka + GRU na rezyduach)':<35} {rmse_hybryda:<18.3f}")
print(f"\nHybryda vs fizyka: {(1-rmse_hybryda/rmse_fizyka)*100:.1f}% mniejszy RMSE")
print(f"Hybryda vs czysty GRU: {(1-rmse_hybryda/rmse_bb)*100:.1f}% mniejszy RMSE")

torch.save(model_bb.state_dict(), "krok6_bioreaktor_gru/model_bb.pt")
torch.save(model_res.state_dict(), "krok6_bioreaktor_gru/model_res.pt")
