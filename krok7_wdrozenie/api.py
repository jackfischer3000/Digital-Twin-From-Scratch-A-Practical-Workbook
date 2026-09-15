"""
Krok 7: serwis inferencji jako FastAPI - endpoint POST /predict.

Zwraca dokladnie to, co specyfikuje Rozdzial 11/13 ksiazki: predykcje z przedzialem
ufnosci, flaga OOD, wersja modelu - plus rozklad fizyka/korekta ML (explainability,
Rozdzial 12) i logike fallback z MDS.
"""
import json
import time
import sys
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field

TU = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TU, "..", "krok6_bioreaktor_gru"))
from symulacja import kla, QO2_BASE, K_DO, DO_SAT, DT

with open(os.path.join(TU, "artefakty.json")) as f:
    ART = json.load(f)

CECHY = ART["cechy"]
W, H = ART["okno_W"], ART["horyzont_H"]
SREDNIA_M = np.array(ART["mahalanobis_srednia"])
KOW_ODWROTNA = np.array(ART["mahalanobis_kowariancja_odwrotna"])
PROG_OOD = ART["prog_ood_99pct"]
KWANTYL = ART["kwantyl_conformal_90"]


class GRUPredyktor(nn.Module):
    def __init__(self, n=5, h=16):
        super().__init__()
        self.gru = nn.GRU(n, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn.squeeze(0)).squeeze(-1)


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ladowanie wag PRZY STARCIE - nie przy kazdym zapytaniu (Rozdzial 13)
    global model
    model = GRUPredyktor()
    model.load_state_dict(torch.load(os.path.join(TU, "model_gru.pt")))
    model.eval()
    print(f"Model wersja {ART['wersja_modelu']} zaladowany (dane: {ART['hash_danych_treningowych']})")
    yield


app = FastAPI(title="Predykcja DO w bioreaktorze", lifespan=lifespan)


class Odczyt(BaseModel):
    stirring_rpm: float
    aeration_vvm: float
    feed_rate: float
    X_biomasa: float
    DO: float


class ZadaniePredykcji(BaseModel):
    okno: list[Odczyt] = Field(..., description=f"Ostatnie {W} minut odczytow, od najstarszego do najnowszego")


class OdpowiedzPredykcji(BaseModel):
    predykcja_DO: float
    przedzial_dolny: float
    przedzial_gorny: float
    wklad_fizyki: float
    wklad_korekty_ml: float
    flaga_ood: bool
    odleglosc_mahalanobisa: float
    tryb: str
    wersja_modelu: str
    czas_inferencji_ms: float


def fizyka_jeden_krok(stan_do, stirring, aeration, x_biomasa):
    otr = kla(stirring, aeration) * (DO_SAT - stan_do)
    our = QO2_BASE * x_biomasa * stan_do / (stan_do + K_DO)
    return np.clip(stan_do + DT * (otr - our), 0.0, 100.0)


def fizyka_rollout_z_okna(okno_df, horyzont=H):
    do = okno_df["DO"].iloc[-1]
    ostatni_stirring = okno_df["stirring_rpm"].iloc[-1]
    ostatni_aeration = okno_df["aeration_vvm"].iloc[-1]
    ostatni_x = okno_df["X_biomasa"].iloc[-1]
    for _ in range(horyzont):
        do = fizyka_jeden_krok(do, ostatni_stirring, ostatni_aeration, ostatni_x)
    return do


@app.post("/predict", response_model=OdpowiedzPredykcji)
def predict(zadanie: ZadaniePredykcji):
    t0 = time.perf_counter()
    import pandas as pd
    okno_df = pd.DataFrame([o.model_dump() for o in zadanie.okno])[CECHY]

    # --- flaga OOD: odleglosc Mahalanobisa ostatniego punktu ---
    ostatni_punkt = okno_df[CECHY].iloc[-1].values
    roznica = ostatni_punkt - SREDNIA_M
    odl = float(np.sqrt(roznica @ KOW_ODWROTNA @ roznica))
    ood = odl > PROG_OOD

    # --- fizyka (dziala zawsze, nawet w trybie fallback) ---
    pred_fizyka = float(fizyka_rollout_z_okna(okno_df))

    if ood:
        # PLAN FALLBACK z MDS: tryb tylko-fizyka, szersze pasmo
        return OdpowiedzPredykcji(
            predykcja_DO=pred_fizyka,
            przedzial_dolny=max(0.0, pred_fizyka - KWANTYL * 2),
            przedzial_gorny=min(100.0, pred_fizyka + KWANTYL * 2),
            wklad_fizyki=pred_fizyka,
            wklad_korekty_ml=0.0,
            flaga_ood=True,
            odleglosc_mahalanobisa=odl,
            tryb="FALLBACK (tylko fizyka)",
            wersja_modelu=ART["wersja_modelu"],
            czas_inferencji_ms=(time.perf_counter() - t0) * 1000,
        )

    # --- tryb normalny: fizyka + korekta GRU ---
    x = torch.tensor(okno_df[CECHY].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        korekta = model(x).item() * ART["cel_odchylenie"] + ART["cel_srednia"]
    predykcja = pred_fizyka + korekta

    return OdpowiedzPredykcji(
        predykcja_DO=predykcja,
        przedzial_dolny=max(0.0, predykcja - KWANTYL),
        przedzial_gorny=min(100.0, predykcja + KWANTYL),
        wklad_fizyki=pred_fizyka,
        wklad_korekty_ml=korekta,
        flaga_ood=False,
        odleglosc_mahalanobisa=odl,
        tryb="AKTYWNY",
        wersja_modelu=ART["wersja_modelu"],
        czas_inferencji_ms=(time.perf_counter() - t0) * 1000,
    )


@app.get("/health")
def health():
    return {"status": "ok", "wersja_modelu": ART["wersja_modelu"]}
