# Model Risk Assessment — model DO w bioreaktorze (ćwiczenie)

> ⚠️ **Status: dokument ćwiczeniowy.** Struktura wg Rozdziału 12 (FMEA + ICH Q9, szablon Aneks B). Wszystkie pięć modów awarii poniżej to **rzeczywiste znaleziska z tej sesji nauki** — złapane na żywo podczas budowy, nie wymyślone dla kompletności dokumentu.

## Identyfikacja modelu i przeznaczenie
Model hybrydowy (bilans masy tlenu + GRU) przewidujący DO w bioreaktorze na horyzoncie 30 min, wspierający decyzję operatorską o interwencji. GAMP Kategoria 5. Szczegóły: `MDS_bioreaktor_DO.md`.

## Znane ograniczenia i zastosowania poza zakresem (out-of-scope)
- Dane syntetyczne — model nie był nigdy walidowany na realnym procesie
- Brak przedziału ufności warunkowego na reżim procesu (patrz mod awarii #3)
- Nie przeznaczony do w pełni autonomicznej decyzji bez nadzoru operatora (zgodnie z Annex 22 — odpowiedzialność człowieka)

## Top 5 modów awarii

| # | Mod awarii | Jak wykryty | Wpływ na CQA | Kontrola mitygująca | Status mitygacji |
|---|---|---|---|---|---|
| 1 | **Ekstrapolacja w rzadko pokrytym obszarze cech** — komponent GRU dał absurdalną korektę (128°C) dla wejścia formalnie w zakresie treningowym, ale w słabo pokrytej kombinacji cech | Złapane na żywo, krok 3 tej sesji (model temperatury reaktora) | Wysoki — predykcja kompletnie nieużyteczna, mogłaby prowadzić do błędnej decyzji operatorskiej | Regularyzacja (`alpha` w MLPRegressor/GRU) ograniczająca wagi; detektor OOD (Mahalanobis) jako druga linia obrony | ✅ Zmitygowany i zweryfikowany — ten sam punkt, który spowodował awarię, dał odległość Mahalanobisa 2.55, powyżej progu 95/99 percentyla |
| 2 | **Czarna skrzynka gorsza niż baseline przy małych danych** — GRU trenowany bez komponentu fizycznego osiągnął RMSE 11.6-12.0%, gorzej niż sama fizyka (7.05%) | Benchmark krok 6, potwierdzone dwukrotnie (dwie niezależne sesje treningowe) | Średni — nie prowadzi do złej predykcji per se, ale uzasadnia samą architekturę | Architektura hybrydowa (fizyka jako nieusuwalny baseline, ML tylko na rezyduach) | ✅ Zmitygowany strukturalnie — wybór architektury, nie parametr do dostrojenia |
| 3 | **Systematyczny błąd predykcji NIEWYKRYWANY przez monitoring wejść** — w przebiegu PQ nr 3, długi odcinek (37 kolejnych predykcji) z błędem 10-13 pkt DO, przy odległości Mahalanobisa poniżej progu OOD | Raport PQ (Rozdział 11), diagnostyka po niskim pokryciu (84.1% vs próg 88%) | **Wysoki** — dokładnie scenariusz, przed którym miał chronić monitoring, a nie chroni w pełni | Częściowa: flaga OOD łapie drift *wejść*, nie łapie błędów wynikających z niedouczonej transformacji rezyduów w danym reżimie | ❌ **NIEZMITYGOWANY** — rekomendacja z raportu PQ (rozszerzenie kalibracji, warunkowy conformal prediction) nie zaimplementowana |
| 4 | **Drift wejść przy zmianie sprzętu/procesu** — nowy zakres mieszania (750-1200 rpm) poza zakresem treningowym, RMSE wzrosło z ~2-4% do 5.9% | Symulacja driftu, sesja monitoringu | Wysoki, ale wykrywalny wcześnie | Detektor Mahalanobisa (wyraźna separacja 2.0-2.4 vs 4.2-4.5, zero nakładania) + protokół retreningu | ✅ Zmitygowany i zweryfikowany — retrening przywrócił RMSE do 1.41 na ukrytym zbiorze testowym (76% poprawy) |
| 5 | **Brak audit trail** — żadna predykcja nigdy nie była logowana z wersją modelu, hashem wejść, przedziałem niepewności, flagą OOD ani decyzją operatora | Sprawdzone bezpośrednio (`grep` po kodzie z tej sesji, zero wyników) przy pisaniu tego dokumentu | Wysoki dla zgodności regulacyjnej (Part 11, Annex 22) — bez tego niemożliwe odtworzenie "dlaczego model to powiedział" przy audycie | Brak | ❌ **NIEZMITYGOWANY** — nie zaimplementowane w żadnym kroku tej sesji |

## Ocena ryzyka rezydualnego
Po zastosowaniu istniejących kontroli, **dwa z pięciu** modów awarii (#3, #5) pozostają otwarte. Model w obecnym stanie **nie kwalifikowałby się** do przejścia z shadow mode do trybu aktywnego (por. Rozdział 11) — zgodnie z zasadą z raportu PQ: powrót do Stage 2, nie poprawka na szybko.

## Rekomendacje przed kolejną iteracją
1. Zaimplementować audit trail (wersja modelu, hash wejść, predykcja + przedział, flaga OOD, decyzja operatora) — minimalna wersja: log do pliku CSV/JSON per predykcja
2. Rozszerzyć zbiór kalibracyjny conformal prediction o więcej przypadków z ostrym trybem metabolicznym (mod #3)
3. Rozważyć conditional/block conformal prediction warunkowany na `metabolic_mode` (nawet jeśli ta zmienna nie jest bezpośrednio mierzalna, można warunkować na proxy: `feed_rate × X_biomasa` skumulowane)

---
*Powiązane: `MDS_bioreaktor_DO.md`, `Raport_PQ_bioreaktor_DO.md`. Wzorzec: Rozdział 12 i Aneks B książki #3.*
