"""
Krok 7: trenujemy model (dokladnie jak w kroku 6) i zapisujemy WSZYSTKIE artefakty
potrzebne do serwowania: wagi GRU, statystyki skalowania wejsc, statystyki normalizacji
celu, kwantyl conformal prediction, statystyki Mahalanobisa (do flagi OOD).

To jest odpowiednik "IQ infrastruktury inferencji" z Rozdzialu 13 - jeden skrypt,
ktory produkuje kompletny, wersjonowany pakiet modelu.
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys, json, hashlib
sys.path.insert(0, '../krok6_bioreaktor_gru')
from symulacja import symuluj_batch, kla, QO2_BASE, K_DO, DO_SAT, DT

torch.manual_seed(0)
CECHY = ['stirring_rpm', 'aeration_vvm', 'feed_rate', 'X_biomasa', 'DO']
W, H = 60, 30
WERSJA_MODELU = "1.0.0"


def fizyka_rollout(df, t0, horyzont=H):
    do = df['DO'].iloc[t0]
    for t in range(t0 + 1, t0 + horyzont + 1):
        x = df['X_biomasa'].iloc[t]
        otr = kla(df['stirring_rpm'].iloc[t - 1], df['aeration_vvm'].iloc[t - 1]) * (DO_SAT - do)
        our = QO2_BASE * x * do / (do + K_DO)
        do = np.clip(do + DT * (otr - our), 0.0, 100.0)
    return do


class OknaDataset(Dataset):
    def __init__(self, batche):
        self.probki, cele = [], []
        for df in batche:
            for t0 in range(W - 1, len(df) - H - 1):
                okno = df[CECHY].iloc[t0 - W + 1: t0 + 1].values
                cel = df['DO'].iloc[t0 + H] - fizyka_rollout(df, t0)
                self.probki.append(okno)
                cele.append(cel)
        cele = np.array(cele, dtype=np.float32)
        self.m, self.s = float(cele.mean()), float(cele.std()) + 1e-6
        self.cele = (cele - self.m) / self.s

    def __len__(self):
        return len(self.probki)

    def __getitem__(self, i):
        return torch.tensor(self.probki[i], dtype=torch.float32), torch.tensor(self.cele[i], dtype=torch.float32)


class GRUPredyktor(nn.Module):
    def __init__(self, n=5, h=16):
        super().__init__()
        self.gru = nn.GRU(n, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn.squeeze(0)).squeeze(-1)


if __name__ == "__main__":
    batche_train = [symuluj_batch(seed=s) for s in range(12)]
    batche_kalibracja = [symuluj_batch(seed=s) for s in range(12, 16)]

    ds = OknaDataset(batche_train)
    model = GRUPredyktor()
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(20):
        for okna, cele in loader:
            opt.zero_grad()
            nn.MSELoss()(model(okna), cele).backward()
            opt.step()
    model.eval()
    print("Model wytrenowany.")

    # --- kwantyl conformal prediction (Rozdzial 10) ---
    def predykcja_hybrydowa(df, t0i):
        x = torch.tensor(df[CECHY].iloc[t0i - W + 1: t0i + 1].values, dtype=torch.float32).unsqueeze(0)
        pf = fizyka_rollout(df, t0i)
        with torch.no_grad():
            kor = model(x).item() * ds.s + ds.m
        return pf, kor

    bledy_kal = []
    for df in batche_kalibracja:
        for t0i in range(W - 1, len(df) - H - 1):
            pf, kor = predykcja_hybrydowa(df, t0i)
            bledy_kal.append(abs(df['DO'].iloc[t0i + H] - (pf + kor)))
    bledy_kal = np.array(bledy_kal)
    kwantyl = float(np.sort(bledy_kal)[int(np.ceil((len(bledy_kal) + 1) * 0.90)) - 1])
    print(f"Kwantyl conformal (90%): {kwantyl:.3f}")

    # --- statystyki Mahalanobisa (flaga OOD) ---
    wszystkie_train = pd.concat(batche_train)[CECHY]
    srednia_m = wszystkie_train.mean().values
    kow_odw = np.linalg.inv(np.cov(wszystkie_train.values.T))
    odl_train = np.array([np.sqrt((x - srednia_m) @ kow_odw @ (x - srednia_m)) for x in wszystkie_train.values])
    prog_ood = float(np.percentile(odl_train, 99))
    print(f"Prog OOD (99 percentyl): {prog_ood:.3f}")

    # --- hash danych treningowych (data freeze, Rozdzial 8) ---
    wszystkie = pd.concat(batche_train, keys=range(12))
    hash_danych = hashlib.sha256(pd.util.hash_pandas_object(wszystkie).values.tobytes()).hexdigest()[:16]

    # --- zapis wszystkich artefaktow ---
    torch.save(model.state_dict(), "model_gru.pt")
    artefakty = {
        "wersja_modelu": WERSJA_MODELU,
        "hash_danych_treningowych": hash_danych,
        "cechy": CECHY,
        "okno_W": W,
        "horyzont_H": H,
        "cel_srednia": ds.m,
        "cel_odchylenie": ds.s,
        "kwantyl_conformal_90": kwantyl,
        "mahalanobis_srednia": srednia_m.tolist(),
        "mahalanobis_kowariancja_odwrotna": kow_odw.tolist(),
        "prog_ood_99pct": prog_ood,
    }
    with open("artefakty.json", "w") as f:
        json.dump(artefakty, f, indent=2)

    print("\nZapisano: model_gru.pt, artefakty.json")
