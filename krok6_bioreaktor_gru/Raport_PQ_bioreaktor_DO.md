# Raport kwalifikacyjny PQ — model DO w bioreaktorze (ćwiczenie)

> ⚠️ **Status: dokument ćwiczeniowy.** Wzorowany na strukturze Rozdziału 11 (studium przypadku bioreaktora). Trzy "przebiegi prospektywne" to trzy nowe, nigdy wcześniej nieużyte symulacje (seed 500/501/502) — analogia do danych zebranych po zamrożeniu modelu, nie prawdziwa produkcja. Model **zamrożony**: ten sam trening co w OQ (12 batchy, seed 0-11), ten sam kwantyl conformal (z kalibracji, seed 12-15), bez żadnego dostrajania po zobaczeniu wyników poniżej.

---

## Protokół
- **Liczba przebiegów prospektywnych:** 3 (zgodnie z praktyką branżową przywołaną w Rozdziale 11 dla procesów ciągłych)
- **Kryteria akceptacji** (z MDS, rewizja 2): RMSE ≤5,0% DO sat.; pokrycie przedziału predykcji 90% ≥88% (próg z przykładu książkowego — nasz MDS nie ustalił go wprost, luka odnotowana niżej); brak fałszywych alarmów krytycznych
- **Postępowanie w razie niepowodzenia:** powrót do Stage 2 z analizą przyczyny, nie doraźna poprawka — zgodnie z Rozdziałem 11

## Wyniki

| Przebieg | RMSE | Pokrycie 90% | Flagi OOD | RMSE ≤5,0%? | Pokrycie ≥88%? |
|---|---|---|---|---|---|
| 1 (seed 500) | 3.895 | 91.3% | 0 | ✅ PASS | ✅ PASS |
| 2 (seed 501) | 3.566 | 97.7% | 0 | ✅ PASS | ✅ PASS |
| 3 (seed 502) | 4.806 | **84.1%** | 3 | ✅ PASS | ❌ **FAIL** |

## Analiza przyczyny — przebieg 3

Zdiagnozowano bezpośrednio (nie odgadnięto): większość "chybień" przedziału predykcji w przebiegu 3 **nie ma podwyższonej odległości Mahalanobisa** — konkretnie długi odcinek (`t=189` do `t=225`, 37 kolejnych predykcji), gdzie model systematycznie przeszacowywał DO o 10-13 punktów procentowych (np. `t=214`: prawda 47.29%, predykcja 59.79%), przy odległości Mahalanobisa 2.3-2.5 — **poniżej** progu OOD (3.637).

**Wniosek:** wejścia w tym oknie wyglądały statystycznie normalnie (mieściły się w tym, co model widział w treningu), ale sam proces — najprawdopodobniej szczególnie ostry epizod trybu metabolicznego — wygenerował dynamikę DO, której komponent residualny (GRU) nie skorygował poprawnie. To jest **inny rodzaj błędu niż drift wejść** (który złapaliśmy wcześniej w monitoringu) — to luka w tym, czego model nauczył się o samej transformacji rezyduów, widoczna tylko w wyniku, nie w rozkładzie wejść.

Osobno: 3 flagi OOD w tym przebiegu (`t=300-302`, odl. Mahalanobisa 3.74-4.78) **zadziałały poprawnie** — dwa z tych trzech punktów rzeczywiście wypadły poza przedział predykcji, więc mechanizm OOD nie jest zepsuty, tylko ma ograniczony zasięg (łapie drift wejść, nie każdą przyczynę złej predykcji).

## Dodatkowa luka wykryta przy pisaniu tego raportu
MDS (rewizja 2) definiuje próg RMSE, ale **nie definiuje formalnie progu pokrycia** dla conformal prediction — użyty tu próg 88% pochodzi z przykładu w Rozdziale 11 książki, nie z naszego dokumentu. To powinno być uzupełnione w MDS przy rewizji 3, nie założone milcząco.

## Wynik PQ: **NIE ZAMKNIĘTE CZYSTO — wymaga analizy i decyzji przed formalnym zamknięciem Stage 2**

Zgodnie z protokołem (powrót do Stage 2 z analizą przyczyny, nie poprawka na szybko), rekomendowane kroki przed ponownym PQ:
1. Rozszerzyć zbiór kalibracyjny o przypadki z ostrymi epizodami metabolicznymi (obecne 4 batche kalibracyjne mogły nie zawierać wystarczająco takich przypadków, stąd kwantyl 90% niedoszacowany dla tego reżimu)
2. Rozważyć osobny, warunkowy kwantyl conformal dla okresów wysokiego trybu metabolicznego (mechanizm z Rozdziału 10 wspomina block/conditional conformal prediction jako opcję przy niejednorodnych błędach)
3. Dodać formalny próg pokrycia do MDS (luka odnotowana wyżej) przed kolejną iteracją

---
*Powiązane: `MDS_bioreaktor_DO.md` (kryteria źródłowe), diagnostyka przebiegu 3 w historii sesji. Wzorzec: Rozdział 11 książki #3.*
