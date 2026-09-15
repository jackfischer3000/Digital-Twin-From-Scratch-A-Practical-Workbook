"""Wizualizacja: prawdziwe DO vs predykcje (fizyka / GRU / hybryda) na horyzoncie 30 min."""
import numpy as np
import matplotlib.pyplot as plt
import torch

from symulacja import symuluj_batch
from model import (
    normalizuj, fizyka_rollout, GRUPredyktor, W, H,
    ds_train_bb, ds_train_res, odstandaryzuj,
)

model_bb = GRUPredyktor()
model_bb.load_state_dict(torch.load("krok6_bioreaktor_gru/model_bb.pt"))
model_bb.eval()

model_res = GRUPredyktor()
model_res.load_state_dict(torch.load("krok6_bioreaktor_gru/model_res.pt"))
model_res.eval()

# batch testowy z wyraznym epizodem metabolicznym (seed spoza treningu, patrz model.py: test = seed 12-15)
df = symuluj_batch(seed=13)
df_norm = normalizuj(df)

czasy, pred_fizyka, pred_bb, pred_hybryda = [], [], [], []
with torch.no_grad():
    for t0 in range(W - 1, len(df) - H - 1):
        okno = torch.from_numpy(df_norm.iloc[t0 - W + 1: t0 + 1].values.astype(np.float32)).unsqueeze(0)
        pf = fizyka_rollout(df, t0)
        pbb = odstandaryzuj(model_bb(okno).item(), ds_train_bb)
        pres = odstandaryzuj(model_res(okno).item(), ds_train_res)
        czasy.append(t0 + H)
        pred_fizyka.append(pf)
        pred_bb.append(pbb)
        pred_hybryda.append(pf + pres)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df["t"], df["DO"], color="#1e293b", linewidth=2.2, label="Prawdziwa DO", zorder=5)
ax.plot(czasy, pred_fizyka, color="#f59e0b", linewidth=1.5, linestyle="--", label="A: tylko fizyka")
ax.plot(czasy, pred_bb, color="#dc2626", linewidth=1.3, linestyle=":", label="B: tylko GRU (czarna skrzynka)")
ax.plot(czasy, pred_hybryda, color="#16a34a", linewidth=1.8, label="C: hybryda (fizyka + GRU)")

# zacieniuj okresy trybu metabolicznego
w_trybie = df["metabolic_mode"].values
start = None
for t in range(len(w_trybie)):
    if w_trybie[t] and start is None:
        start = t
    if (not w_trybie[t] or t == len(w_trybie) - 1) and start is not None:
        ax.axvspan(start, t, color="#fca5a5", alpha=0.2)
        start = None

ax.set_xlabel("Czas (minuty)")
ax.set_ylabel("DO (% saturacji)")
ax.set_title("Predykcja DO z horyzontem 30 min — batch testowy (czerwone pasy = tryb metaboliczny, niewidoczny dla fizyki)")
ax.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("krok6_bioreaktor_gru/benchmark_trajektoria.png", dpi=130, bbox_inches="tight")
print("Zapisano: krok6_bioreaktor_gru/benchmark_trajektoria.png")
