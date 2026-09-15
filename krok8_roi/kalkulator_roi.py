"""
Krok 8: kalkulator ROI dla modelu procesowego (Rozdzial 15).

Struktura kalkulacji jest ogolna - podstaw wlasne liczby (koszt osobo-miesiaca,
wartosc partii, czestosc odchylen), model zostaje ten sam. Ponizej: liczby
ILUSTRACYJNE z ksiazki (model DO w bioreaktorze), zeby zweryfikowac ze kalkulator
odtwarza ich wynik (payback ~2.9 roku).
"""
from dataclasses import dataclass


@dataclass
class KosztDevelopmentu:
    osobo_miesiace: float
    stawka_za_osobo_miesiac: float
    infrastruktura_jednorazowo: float

    @property
    def razem(self):
        return self.osobo_miesiace * self.stawka_za_osobo_miesiac + self.infrastruktura_jednorazowo


@dataclass
class KosztUtrzymaniaRocznie:
    monitoring_retrening: float
    rekwalifikacja_okresowa: float
    infrastruktura: float

    @property
    def razem(self):
        return self.monitoring_retrening + self.rekwalifikacja_okresowa + self.infrastruktura


@dataclass
class KorzysciRocznie:
    uniknione_odchylenia_formalne: float   # liczba/rok x koszt investigation/CAPA
    uniknieta_utrata_partii: float          # wartosc partii / oczekiwany okres miedzy utratami

    @property
    def razem(self):
        return self.uniknione_odchylenia_formalne + self.uniknieta_utrata_partii


def oblicz_roi(koszt_dev: KosztDevelopmentu, koszt_utrz: KosztUtrzymaniaRocznie,
                korzysci: KorzysciRocznie, partie_rocznie: int, lata_npv: int = 5,
                stopa_dyskontowa: float = 0.08):
    korzysc_netto_roczna = korzysci.razem - koszt_utrz.razem
    payback_lata = koszt_dev.razem / korzysc_netto_roczna if korzysc_netto_roczna > 0 else float("inf")

    # NPV: suma zdyskontowanych korzysci netto przez `lata_npv`, minus koszt poczatkowy
    npv = -koszt_dev.razem
    for rok in range(1, lata_npv + 1):
        npv += korzysc_netto_roczna / ((1 + stopa_dyskontowa) ** rok)

    koszt_na_partie = koszt_utrz.razem / partie_rocznie
    korzysc_na_partie = korzysci.razem / partie_rocznie
    netto_na_partie = korzysc_na_partie - koszt_na_partie

    return {
        "koszt_developmentu": koszt_dev.razem,
        "koszt_utrzymania_rocznie": koszt_utrz.razem,
        "korzysc_roczna_razem": korzysci.razem,
        "korzysc_netto_roczna": korzysc_netto_roczna,
        "payback_lata": payback_lata,
        "payback_miesiace": payback_lata * 12,
        f"npv_{lata_npv}lat": npv,
        "koszt_na_partie": koszt_na_partie,
        "korzysc_na_partie": korzysc_na_partie,
        "netto_na_partie": netto_na_partie,
    }


if __name__ == "__main__":
    # === Liczby ILUSTRACYJNE z ksiazki (model DO w bioreaktorze, Rozdzial 15) ===
    koszt_dev = KosztDevelopmentu(
        osobo_miesiace=22,
        stawka_za_osobo_miesiac=25_000,
        infrastruktura_jednorazowo=150_000,
    )
    koszt_utrz = KosztUtrzymaniaRocznie(
        monitoring_retrening=45_000,
        rekwalifikacja_okresowa=40_000,
        infrastruktura=60_000,
    )
    korzysci = KorzysciRocznie(
        uniknione_odchylenia_formalne=2 * 60_000,       # 2/rok x 60k PLN
        uniknieta_utrata_partii=800_000 / 3,             # wartosc partii / 3 lata
    )

    wynik = oblicz_roi(koszt_dev, koszt_utrz, korzysci, partie_rocznie=24)

    print("=== Weryfikacja przeciw ksiazce (Rozdzial 15) ===\n")
    for klucz, wartosc in wynik.items():
        print(f"{klucz:<30} {wartosc:>15,.0f}")

    print(f"\nKsiazka podaje: payback ~2.9 roku (34 miesiace), koszt netto ~10 200 PLN/partie")
    print(f"Nasz kalkulator: payback {wynik['payback_lata']:.1f} roku ({wynik['payback_miesiace']:.0f} miesiecy), netto {-wynik['netto_na_partie']:,.0f} PLN/partie")
