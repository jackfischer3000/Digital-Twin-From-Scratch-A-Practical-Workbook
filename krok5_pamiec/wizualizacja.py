"""Wizualizacja: przebieg w czasie - prawda vs model bez pamieci vs model z pamiecia."""
import pandas as pd
import matplotlib.pyplot as plt

wyniki_a = pd.read_csv("krok5_pamiec/wyniki_A.csv")
wyniki_b = pd.read_csv("krok5_pamiec/wyniki_B.csv")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(wyniki_a["czas_krok"], wyniki_a["T_prawdziwa"], color="#1e293b", linewidth=2, label="Prawdziwa T")
ax.plot(wyniki_a["czas_krok"], wyniki_a["pred_A_bez_pamieci"], color="#dc2626", linewidth=1.5,
        linestyle="--", label="Model A: bez pamięci (tylko bieżąca moc)")
ax2 = ax.twinx()
ax2.plot(wyniki_a["czas_krok"], wyniki_a["moc"], color="#94a3b8", alpha=0.5, linewidth=1, label="moc (oś prawa)")
ax2.set_ylabel("Moc grzałki (kW)", color="#94a3b8")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Model BEZ pamięci — mija się z prawdą przy każdej zmianie mocy (transient)")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
ax.plot(wyniki_b["czas_krok"], wyniki_b["T_prawdziwa"], color="#1e293b", linewidth=2, label="Prawdziwa T")
ax.plot(wyniki_b["czas_krok"], wyniki_b["pred_B_z_pamiecia"], color="#16a34a", linewidth=1.5,
        linestyle="--", label="Model B: z pamięcią (moc + własna T poprzednia)")
ax.set_xlabel("Krok czasowy (minuty)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Model Z pamięcią — śledzi dynamikę, nie tylko wartość chwilową")
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("krok5_pamiec/pamiec_porownanie.png", dpi=130, bbox_inches="tight")
print("Zapisano: krok5_pamiec/pamiec_porownanie.png")
