# Model Design Specification — predykcja DO w bioreaktorze (ćwiczenie)

> ⚠️ **Status: dokument ćwiczeniowy, nie regulacyjny.** Wzorowany na strukturze z Rozdziału 7 książki #3 i przykładzie MDS dla bioreaktora tam przytoczonym. Dane są syntetyczne (`ML_NAUKA/krok6_bioreaktor_gru/symulacja.py`), nie pochodzą z realnego procesu MODICA.
>
> **Zastrzeżenie metodologiczne, którego książka nie pozwala przemilczeć:** ten MDS jest pisany *po* tym, jak Stage 2 (trening, benchmark) już się odbył — znamy wynik hybrydy (RMSE 4,12%). W prawdziwym projekcie to jest dokładnie ta sekwencja, przed którą książka ostrzega ("próg dobrany post-hoc do wyniku przestaje być kryterium akceptacji, staje się opisem faktu" — Rozdział 7). Kryterium poniżej jest więc wyprowadzone z rozumowania niezależnego od wyniku (margines nad baseline fizyki), nie dopasowane do 4,12% — ale sam fakt, że piszemy to teraz, a nie przed treningiem, jest odstępstwem od procesu, który w realnym projekcie GxP dyskwalifikowałby ten dokument na przeglądzie DQ.

---

## Cel modelu i granice systemu
Predykcja rozpuszczonego tlenu (DO) w bioreaktorze na horyzoncie **t+30 minut**, aktualizacja co krok symulacji (1 min). Model wspiera decyzję operatorską o interwencji (dodanie tlenu, redukcja obciążenia) przed przekroczeniem progu krytycznego, nie zastępuje pomiaru bieżącego DO.

## Wejścia i wyjścia
**Wejścia** (okno historii 60 minut, 5 zmiennych): `stirring_rpm`, `aeration_vvm`, `feed_rate`, `X_biomasa`, `DO` (odczyt bieżący jako część okna).
**Wyjście**: predykcja punktowa DO w t+30 min **+ przedział predykcji 90%** (conformal prediction, Rozdział 10 — kwantyl 7,12 %DO sat. wyliczony na 1560 punktach kalibracyjnych, empiryczne pokrycie na osobnym zbiorze testowym: 96,1%, ≥ celu 90%). Flaga OOD: odległość Mahalanobisa, patrz plan fallback.

## Wybrana architektura i uzasadnienie
Hybrydowa, residual: bilans masy tlenu (dDO/dt = OTR − OUR, korelacja kLa) + GRU (1 warstwa, 24 jednostki ukryte) na rezyduach fizyki.

Uzasadnienie oparte na własnym eksperymencie (krok 6, nie na literaturze — inaczej niż w książce): przy 12 batchach treningowych czysty GRU (czarna skrzynka) osiągnął RMSE 12,05% — **gorzej niż sama fizyka** (7,05%). To potwierdza tezę książki (Rozdział 6/D.1): przy ograniczonej liczbie przebiegów treningowych model bez wbudowanej wiedzy fizycznej nie ekstrapoluje. Hybryda: 4,12%.

## Kryteria akceptacji
| Kryterium | Metryka | Próg | Zbiór danych | Metoda pomiaru |
|---|---|---|---|---|
| Dokładność predykcji | RMSE między predykcją DO na horyzoncie t+30 min a wartością symulowaną | ≤5,0% DO sat. | Zbiór testowy wydzielony na poziomie całych batchy (4 z 16, nieużyte w treningu) | Pojedynczy pomiar na wydzielonym zbiorze testowym po zamrożeniu modelu |
| Przewaga nad baseline | RMSE hybrydy vs RMSE modelu czysto fizycznego | wyraźnie < 7,05% (baseline fizyki) | jw. | jw. |

**Skąd bierze się 5,0%** (metodologia z Rozdziału 7, cztery źródła — stosowana tu do własnych, umownych założeń, nie do rzeczywistej specyfikacji procesu):
1. *Margines operacyjny (założony umownie)*: gdyby próg interwencji wynosił DO<20%, a normalny zakres pracy 40–90%, margines rzędu 20–30 pkt sugeruje budżet błędu o rząd wielkości mniejszy — kilka punktów procentowych.
2. *Niepewność referencji*: n/d — dane syntetyczne, brak realnego czujnika. W prawdziwym projekcie to jest **pierwsza** liczba do sprawdzenia (karta katalogowa czujnika), nie ostatnia.
3. *Baseline, który już działa*: sama fizyka = 7,05%. Próg musi być wyraźnie lepszy, inaczej komponent ML nie ma uzasadnienia — stąd 5,0% jako liczba niezależna od wyniku hybrydy, wyprowadzona wyłącznie z baseline'u.
4. *Konsekwencja błędu dla decyzji*: nie oceniona z prawdziwym SME procesowym (brak takiej osoby w ćwiczeniu) — realny projekt wymagałby tej rozmowy przed ustaleniem progu.

## Plan fallback
> **Aktualizacja po przeglądzie DQ:** poprzednia wersja odwoływała się do "prostego testu zakresu" i sugerowała Mahalanobisa jako przyszłe ulepszenie. W międzyczasie odległość Mahalanobisa została zbudowana i zwalidowana (na rzeczywistej awarii ekstrapolacji z kroku 3 — złapała punkt, który spowodował 128°C absurdalnej korekty) — poniżej zaktualizowana specyfikacja.

Przełączenie na **tryb tylko-fizyka** (sam komponent mechanistyczny), gdy którykolwiek z warunków się spełni:
- odległość Mahalanobisa bieżącego punktu wejściowego (5 cech: `stirring_rpm`, `aeration_vvm`, `feed_rate`, `X_biomasa`, `DO`) względem rozkładu danych treningowych przekracza **99. percentyl z treningu** — próg wyliczony empirycznie, nie umownie (metoda: `krok_monitoring/mahalanobis.py`, zweryfikowana na rzeczywistym przypadku awarii)
- brak odczytu w oknie historii (okno krótsze niż 60 minut)

**Powiadomienia po wyzwoleniu** (poprzednia wersja tego nie precyzowała — luka złapana na przeglądzie DQ):
- wpis w audit trail (znacznik czasu, wartość odległości Mahalanobisa, które wejście przekroczyło próg)
- alert do operatora zmianowego (etykieta "TRYB FALLBACK" na ekranie operatorskim) — wymaga potwierdzenia
- jeśli stan utrzymuje się >3 kolejne cykle: dodatkowe powiadomienie do inżyniera automatyki/MSAT

Po wyzwoleniu: predykcja = wyjście samego modelu fizycznego, z widoczną flagą trybu fallback.

## Alternatywy rozważone i odrzucone
- **Czysty GRU (czarna skrzynka)** — odrzucony: RMSE 12,05% vs 4,12% hybrydy; 12 batchy treningowych za mało, żeby model bez wiedzy fizycznej nauczył się ekstrapolować (krok 6, benchmark).
- **Czysty model fizyczny** — odrzucony jako cel docelowy (użyty jako baseline i logika fallback): RMSE 7,05% nie mieści się w progu 5,0%; źródło błędu zidentyfikowane w analizie rezyduów (krok 3 tej sesji) jako pominięcie efektu metabolicznego overflow.
- **MLP bez pamięci zamiast GRU** — odrzucony: krok 5 tej sesji pokazał na prostszym scenariuszu (reaktor z bezwładnością cieplną), że model bez dostępu do stanu poprzedniego ma RMSE 2,6× gorsze niż model z pamięcią (2,64°C vs 0,52°C). Efekt metaboliczny w bioreaktorze akumuluje się w czasie (zależy od historii karmienia i biomasy) — analogiczny wymóg pamięci.

## Wstępna klasyfikacja GAMP 5
**Kategoria 5** (oprogramowanie na zamówienie) — komponent GRU jest bespoke, trenowany na danych specyficznych dla tego procesu, nie konfiguracją standardowego pakietu.

## Zamrożenie danych treningowych (dodane na przeglądzie DQ — Rozdział 8)
Zbiór treningowy: 12 batchy symulowanych (`symuluj_batch(seed=0..11)`, `krok6_bioreaktor_gru/symulacja.py`).
**Hash SHA-256 (pierwsze 16 znaków): `c9ffb44018fdc9bb`** — jednoznaczny identyfikator wersji danych użytej do treningu. Jakakolwiek zmiana symulacji (stałe fizyczne, zakresy zmiennych) po tym punkcie wymaga nowego hasha i formalnej notatki, nie cichej podmiany.

---
*Rewizja 2 — po przeglądzie DQ: zaktualizowano plan fallback (Mahalanobis zamiast testu zakresu, doprecyzowano powiadomienia), dodano zamrożenie danych. Nieadresowane: brak URS (Rozdział 2) jako dokumentu referencyjnego — strukturalna luka z pominięcia tego etapu w tej sesji nauki, nie backfillowana retrospektywnie.*
*Powiązane: `krok6_bioreaktor_gru/model.py` (implementacja), `krok6_bioreaktor_gru/benchmark_trajektoria.png` (wynik wizualny). Wzorzec: Rozdział 7 i Aneks B książki #3.*
