# UFC Fight Predictor — kontekst projekta

> Ovaj fajl je "living document" — ažurira se kroz razgovor. Sadrži: šta smo analizirali, šta smo odlučili, i šta je još otvoreno. Čita se prije nego što se počne s implementacijom.

## Status: MVP IMPLEMENTIRAN + dodatak za scraping/retreniranje/verzionisanje modela + eksperiment sa alternativnom feature reprezentacijom (diff vs concat) (sve verifikovano end-to-end)

---

## 1. Analiza postojećeg repozitorija (`musicle-ai`)

Referentni projekat istog vlasnika — služi kao **arhitekturni predložak**, ne kao kod koji se reusa direktno.

- **Domena:** Analiza audio zapisa (Spotify featurese) — žanr klasifikacija + regresija skorova (commercial/production/viral).
- **Stack:**
  - Backend: ASP.NET Core 9 (C#), ML.NET 3 (LightGBM + FastTree), SignalR (realtime), JWT auth, EF Core + SQL Server 2022.
  - Frontend: Next.js 16, React 19, TypeScript, Tailwind v4.
  - Docker Compose za cijeli stack (SQL Server, API, Next.js).
- **Arhitektura backend-a** (bitno kao predložak za UFC app):
  - `AiAgents.Core` — apstraktni "software agent" framework (perception → policy → actuator).
  - `AiAgents.MusicAgent` — domain/aplikacija/ML/background workeri.
  - `AiAgents.Shared` — DTO-i dijeljeni između API-ja i agent logike.
  - `AiAgents.Web` — ASP.NET host, kontroleri, SignalR hub.
  - Dva `IHostedService` workera: jedan radi analizu (queue → feature extraction → scoring → push preko SignalR), drugi periodično retrenira model na osnovu korisničkog feedbacka.
  - ML kod je izolovan u `/ML` folderu: klasifikator, regresori, cross-validation servis, feature-correlation servis, K-Means, KNN similarity — dakle ne samo "jedan model" nego čitav set ML servisa.
- **Zaključak:** Ako budemo htjeli slično "ozbiljan" backend (ASP.NET Core + ML.NET) za UFC app, imamo gotov obrazac organizacije koda. Alternativa (Python/FastAPI + scikit-learn) je opcija o kojoj tek treba odlučiti — vidi otvorena pitanja.

## 2. Analiza dokumenta (`Musicle Analiza Modela.docx`)

Ovo je **seminarski rad** (mašinsko učenje) koji empirijski evaluira ML modele iz `musicle-ai` sistema. Bitan je kao **predložak metodologije evaluacije** koji ćemo vjerovatno željeti replicirati za UFC modele.

Struktura rada (koristiti kao checklist za našu buduću evaluaciju):
1. Uvod i istraživačka pitanja (tačnost/stabilnost, kalibracija pouzdanosti, poređenje algoritama).
2. Metodologija — dataset, train/test split (80/20), 5-fold cross-validacija, definicije metrika.
3. Model A (regresije) — R², RMSE po cilju.
4. Model B (klasifikacija žanra) — accuracy, precision/recall/F1 po klasi, matrica konfuzije.
5. Cross-validacija (5-fold) — mean ± std po foldu, poređenje stabilnosti.
6. Kalibracija pouzdanosti — **confidence = max(softmax) × 100%**, Log-Loss kao mjera kalibracije (niže = bolje), naglasak da accuracy nije dovoljna metrika ako se korisniku prikazuje "pouzdanost".
7. Feature importance — ANOVA F-statistika, normalizovano 0–100.
8. Korelaciona analiza featura (Pearson r) — opravdava zašto neki featurese mogu predviđati druge.
9. Komparacija algoritama (LightGBM vs FastTree) — tabela pobjednika po kriteriju.
10. Ograničenja i prijetnje validnosti (heurističke oznake umjesto ljudske anotacije, mali test skup za rijetke klase, curenje informacija kroz sintetičke primjere, train/serve skew).
11. Zaključak s jasnim odgovorima na istraživačka pitanja.

**Ključni nalaz iz rada koji je relevantan za nas:** dva modela mogu imati skoro identičnu tačnost, ali se bitno razlikovati u kalibraciji (log-loss) — bitno ako ćemo korisniku prikazivati postotak pouzdanosti predikcije (npr. "73% šanse za KO/TKO").

**Implikacija za UFC projekat:** Ako želimo akademski/stručno utemeljen prikaz kvaliteta modela (a ne samo "radi/ne radi"), trebalo bi planirati od početka: cross-validaciju, matricu konfuzije (3 klase: KO/TKO, Submission, Decision), kalibraciju (log-loss / confidence), feature importance, i eksplicitno navesti ograničenja.

## 3. Analiza CSV podataka

### `ufc_fighters_final.csv` — 4455 boraca, 18 kolona
`Fighter_Name, Height, Weight, Reach, Stance, DOB, Wins, Losses, Draws, SLpM, Str_Acc, SApM, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg, Fighter_URL`

- Career-level statistike po borcu (trenutni/kumulativni presjek, ne po meču).
- **Nedostajući podaci (bitno!):**
  - `Reach` nedostaje kod **1940 / 4455** boraca (~44%).
  - `Stance` nedostaje kod **849 / 4455** (~19%).
  - `Height` nedostaje kod **318 / 4455** (~7%).
- Stance vrijednosti: Orthodox (2769), Southpaw (608), Switch (219), Open Stance (7), Sideways (3), prazno (849).
- `Fighter_URL` je unique ključ (ufcstats.com) — koristi se za povezivanje s meč-fajlom.

### `ufc_gold_dataset_final.csv` — 8551 mečeva, 37 kolona
`Fight_URL, Fighter_1, Fighter_2, Winner, Weight_Class, Method, End_Round, End_Time, Total_Fight_Time_Sec, Time_Format, F1_KD, F2_KD, F1_Sig_Landed, F1_Sig_Att, F2_Sig_Landed, F2_Sig_Att, F1_TD_Landed, F2_TD_Landed, F1_TD_Att, F2_TD_Att, F1_Sub_Att, F2_Sub_Att, F1_Ctrl_Sec, F2_Ctrl_Sec, F1_Head, F2_Head, F1_Body, F2_Body, F1_Leg, F2_Leg, F1_Distance, F2_Distance, F1_Clinch, F2_Clinch, F1_Ground, F2_Ground, Event_Date`

- Per-fight statistike, **ali ovo su rezultati borbe (post-fight stats)**, ne pre-fight predictori! `F1_Sig_Landed`, `F1_TD_Landed`, `F1_Ctrl_Sec` itd. su izmjereni **tokom** borbe koja se desila — dakle **ne mogu se koristiti kao ulazni feature-i za predikciju unaprijed** (data leakage), jer ih ne znamo prije borbe. Ovo je ključna arhitekturna stavka za feature engineering (vidi Otvorena pitanja #3).
- Raspon datuma: 1994-03-11 do 2026-03-07.
- **Method (target varijabla za klasifikaciju)** — distribucija:
  - Decision - Unanimous: 3083
  - KO/TKO: 2688
  - Submission: 1655
  - Decision - Split: 815
  - Decision - Majority: 98
  - TKO - Doctor's Stoppage: 97
  - Overturned: 58
  - Could Not Continue: 32
  - DQ: 23
  - Other: 2
  - → Za "KO / Submission / Odluka sudije" kao 3 klase, sve "Decision - *" treba spojiti u jednu klasu "Decision", "TKO - Doctor's Stoppage" vjerovatno u KO/TKO klasu, a Overturned/Could Not Continue/DQ/Other (ukupno 115 mečeva, ~1.3%) vjerovatno treba **izbaciti** iz treninga jer nisu prava "metoda pobjede" u smislu koji nas zanima.
- **Winner** — 151 mečeva ima `Draw/NC` (nije Fighter_1 ni Fighter_2) → treba odlučiti da li se izbacuju iz dataseta za predikciju pobjednika.
- **Weight_Class** — 123 različitih string vrijednosti (uglavnom zbog "Title Bout" varijanti i starih/rijetkih kategorija) — treba normalizovati u ~10-ak standardnih kategorija (npr. mapiranjem "UFC Lightweight Title Bout" → "Lightweight" itd.), plus title-fight flag kao poseban feature ako bude koristan.
- **Provjereno:** svi borci iz meč-fajla postoje u fighters fajlu (0 nedostajućih) → join po imenu je moguć, ali imena kao ključ su rizična (duplikati imena, promjene pisanja) — bolje bi bilo imati i `Fighter_URL` po borcu u meč-fajlu ako ikad budemo re-scrapali; za sada radimo join po tačnom nazivu.

## 4. Cilj nove aplikacije (UFC Fight Predictor)

- Korisnik bira: **borac 1, borac 2, kategorija (weight class, auto-predložena iz boraca ali izmjenjiva)**.
- Korisnik takođe bira **kojim modelom** želi da se radi predikcija (min. 2, planirano 3 modela — vidi niže).
- Aplikacija predviđa:
  1. **Pobjednika** (Fighter 1 vs Fighter 2).
  2. **Način završetka meča**: KO/TKO, Submission, ili Decision — sa vjerovatnoćama/confidence po klasi.
- UI prikazuje i profile boraca, poređenje statistika i tačnost/pouzdanost odabranog modela — širi, detaljan prikaz, ne samo gola predikcija.
- Feature-i za model dolaze isključivo iz pre-fight poznatih podataka (career stats do datuma meča, razlike između boraca, fizički atributi) — nikad iz per-fight rezultata tog istog meča (data leakage).

## 5. Odluke donesene do sada

**Analiza**
- Analiza repozitorija i dokumenta je završena (ovaj fajl je njen zapisnik).
- Cilj proizvoda: predikcija pobjednika + metode završetka UFC meča (KO/TKO / Submission / Decision) na osnovu izbora dva borca i kategorije, uz izbor ML modela od strane korisnika.
- Implementacija **neće** početi dok se ne završi kompletna analiza i diskusija.

**Stack**
- **Backend: Python + FastAPI**, ne .NET/ML.NET. Razlog: bogatiji ML ekosistem (scikit-learn, LightGBM, scipy za ANOVA, SHAP za feature importance, imbalanced-learn) potreban za akademski nivo evaluacije — manje truda nego repliciranje istog u ML.NET-u.
- **Frontend: Next.js + TypeScript + Tailwind + shadcn/ui**, isti dizajn-jezik kao musicle-ai (koji već ima `Command` komponentu za search/autocomplete i `Progress` bar za prikaz vjerovatnoća — reusable inspiracija, ne kod).
- **Bez klasične baze podataka.** Podaci: predprocesiran dataset (parquet/csv) učitan u memoriju (pandas) pri startu; istrenirani modeli sačuvani kao fajlovi (`.joblib`/LightGBM native format). Dataset je premali (4455 boraca / 8551 mečeva) da opravda SQL server, nema korisničkih naloga niti pisanja podataka.
- Projekat se **ne hostuje** za sada (fakultetski projekat) — nema potrebe za auth-om, Dockerom za produkciju i sl. (može se dodati kasnije ako zatreba).

**Targeti i modeliranje**
- Dva cilja: **Winner** (binarna klasifikacija) i **Method** (3-klasna: KO/TKO, Submission, Decision).
- Predikcija runde se **ne radi** (suvišno za sada).
- Weight class se auto-predlaže iz istorije odabranih boraca, korisnik može promijeniti (podržava hipotetičke mečeve van uobičajene kategorije).
- Winner model trenira se na **razlikama statistika** (borac A − borac B) uz augmentaciju zamjenom redoslijeda (svaki meč ulazi dvaput, s obrnutim predznakom/labelom) da bi se izbjegla pozicijska pristrasnost (Fighter_1 vs Fighter_2 redoslijed ne smije uticati na predikciju).
- **3 modela** planirana (min. 2 traženo, 3 daje bolju priču za komparativnu evaluaciju, po uzoru na Musicle rad):
  1. **Logistic Regression** — jednostavan, interpretabilan baseline.
  2. **Random Forest** — bagging ansambl, robustan, dobra feature importance.
  3. **LightGBM** — gradient boosting, očekivano najbolja tačnost/kalibracija.
  Korisnik u UI bira kojim od ova tri se radi predikcija; aplikacija po potrebi prikazuje i tačnost/metrike odabranog modela.

**Podaci / feature engineering**
- Potvrđen data-leakage problem: `F1_Sig_Landed`, `F1_TD_Landed`, `F1_Ctrl_Sec` itd. su rezultati meča i ne smiju se koristiti kao ulaz. Gradimo **rekonstrukciju career statistika "na dan meča"** iz istorije (kumulativno/rolling po datumu), a ne trenutni presjek iz fighters fajla.
- **Reach** (44% nedostaje): imputacija iz Height (jaka korelacija u MMA/boksu) + missing-indicator feature.
- **Height** (7%): imputacija medijanom po weight class-i + missing-indicator feature.
- **Stance** (19%): tretira se kao posebna kategorija "Unknown", ne imputira se.
- **Weight_Class**: 123 sirove vrijednosti normalizovati u ~10-12 standardnih kategorija + poseban title-fight flag feature.
- **Draw/NC, Overturned, DQ, Could Not Continue, Other** (~2% mečeva) — izbacuju se iz treninga (nisu čist signal o ishodu vezanom za vještinu boraca).
- **Class imbalance** (Method: Decision ~3996 / KO-TKO ~2785 / Submission ~1655): koristiti `class_weight='balanced'` kao default; **izbjegavati sintetičko oversampling-ovanje (SMOTE)** — Musicle rad je eksplicitno naveo curenje informacija kroz sintetičke primjere kao ograničenje, izbjegavamo istu grešku.

**Evaluacija i izvještavanje**
- Radimo pun akademski evaluacioni pipeline (isti nivo kao Musicle rad): 80/20 split + 5-fold cross-validacija, matrica konfuzije, kalibracija/log-loss, ANOVA feature importance, korelaciona analiza, poređenje modela, eksplicitna ograničenja — ovi podaci/grafici će biti osnova i za app (transparentnost u UI) i za budući seminarski rad o ovoj aplikaciji.
- UI prikazuje detalje predikcije: vjerovatnoće po klasi, koji faktori su najviše uticali, tačnost odabranog modela — ne samo krajnji ishod.

**UI/UX obim**
- Širi UI: search/autocomplete boraca, forma za poređenje (side-by-side profili i statistike), izbor modela, prikaz predikcije s vjerovatnoćama i objašnjenjem, prikaz metrika modela.

## 6. Implementacija — šta je urađeno

Kod živi u `ML/ufc-predictor/` (`backend/` — Python/FastAPI, `frontend/` — Next.js). Struktura prati plan iz `.claude/plans/polymorphic-brewing-parrot.md`.

### 6.1 Data pipeline i feature engineering
- `data_pipeline.py`: parsira Height/Weight/Reach/postotke/DOB, normalizuje Method u 3 klase i Weight_Class u 15 kanonskih kategorija (uključujući "Unknown" za rijetke pred-UFC turnirske mečeve bez kategorije). Od 8551 mečeva, 8377 ostaje nakon filtriranja Draw/NC/Overturned/DQ/Could Not Continue/Other (~2%).
- `features.py`: **dva odvojena mjesta gdje se "jedan meč" razdvaja u dva reda**, iz različitih razloga:
  1. `build_history_long()` — da bi se career agregati (broj pobjeda, win streak, prosjeci) mogli računati po borcu kroz vrijeme (`groupby("fighter_name")`), svaki od 8377 mečeva se prvo pretvara u dva reda (jedan iz perspektive svakog borca) → 16754-redni međukorak, nad kojim se onda radi point-in-time rekonstrukcija (cumsum-minus-current trik za agregate, jednoprolazna petlja po borcu za win streak).
  2. Kasnije, u finalnoj trening tabeli, svaki (već spojen par borac-A/borac-B) meč se **ponovo** duplicira zamjenom strana (A↔B, obrnut predznak razlike, obrnuta oznaka pobjednika) da model ne bude osjetljiv na redoslijed Fighter_1/Fighter_2 → opet 16754 redova (2×8377), savršeno balansirano po `winner_is_a`.

  Ova dva koraka rade istu stvar ("1 red → 2 reda") na različitim mjestima pipeline-a i s različitom svrhom — lako ih je pobrkati, otud ova napomena.
- Imputacija Height/Reach/Weight (regresija/medijan po kategoriji + missing-flag).
- **Otkriven i ispravljen bug tokom pisanja unit testova**: `win_streak` kolona je originalno računata pogrešno zbog pandas `groupby.apply` nekonzistentnosti (vraćao DataFrame umjesto Series u određenim slučajevima) — testovi (`test_win_streak_resets_on_loss`) su to uhvatili prije treniranja modela.
- **Poznato ograničenje (dokumentovano, ne popravljeno)**: 7 imena boraca (14 stvarnih osoba, ~0.3%) su duplikati u `ufc_fighters_final.csv` (npr. dva različita "Bruno Silva"). Pošto meč-dataset nosi samo imena (ne URL), point-in-time "forma" za ta imena miješa istorije oba borca. Za live predikciju, statički atributi (visina/reach/starost) su ispravni jer se biraju po `fighter_id` (URL), ali rolling stats (win streak, prosjeci) nisu 100% pouzdani za tih 14 boraca. Prihvaćeno kao ograničenje niske težine (analogno "heurističkim oznakama" u Musicle radu).

### 6.2 Modeli — rezultati treniranja (80/20 grupisan split po `fight_url` + 5-fold grupisana CV, da augmentovane parnjak-verzije istog meča ne završe na različitim stranama splita)

**Winner (binarna klasifikacija)**:
| Model | Accuracy | Macro F1 | Log-loss | CV acc |
|---|---|---|---|---|
| Logistic Regression | 62.7% | 0.627 | 0.635 | 62.7% ± 1.2 |
| Random Forest | 61.7% | 0.617 | 0.643 | 62.3% ± 0.5 |
| LightGBM | 60.7% | 0.607 | 0.643 | 61.7% ± 0.9 |

**Method (3-klasna: KO_TKO/Submission/Decision)**:
| Model | Accuracy | Macro F1 | Log-loss |
|---|---|---|---|
| Logistic Regression | 48.1% | 0.425 | 1.189 |
| Random Forest | 50.2% | 0.366 | 1.289 |
| LightGBM | 44.6% | 0.421 | 1.263 |

Najvažniji featurei (ANOVA): za Winner — `age_years_diff`, `win_streak_diff`, `prior_avg_td_landed_diff`; za Method — `prior_avg_sig_landed_diff`, `prior_avg_td_att_diff`, `prior_avg_sub_att_diff`. Sve smisleno (starost/iskustvo/forma odlučuju pobjednika; volumen udaraca/pokušaji submissiona nagovještavaju način završetka).

**Nalaz vrijedan za budući rad**: Method-klasifikacija ima nisku tačnost (44-50%, blizu majority-class baseline-a ~47%) — predviđanje NAČINA završetka meča iz pred-meč statistika je suštinski teško (zavisi od dinamike konkretne borbe, ne samo career prosjeka). Random Forest ima najvišu accuracy ali najniži macro-F1 (0.366) — kolabira ka većinskoj klasi (Decision) uprkos `class_weight='balanced'`, dok Logistic Regression bolje balansira klase uz nižu accuracy. Ovo je direktna paralela sa Musicle nalazom da accuracy nije dovoljna metrika — ovdje macro-F1/matrica konfuzije otkrivaju da "bolji" model po accuracy-ju (RF) je zapravo lošiji u prepoznavanju Submission ishoda.

### 6.3 Backend (FastAPI)
Endpointi: `GET /api/fighters?search=`, `GET /api/fighters/{id}`, `GET /api/models`, `POST /api/predict`. `fighter_id` = hex sufiks iz `Fighter_URL` (potvrđeno jedinstven za svih 4455 boraca). Servisi (`FighterService`, `PredictionService`) učitavaju parquet/joblib artefakte jednom pri startu (singleton). Testirano end-to-end preko curl-a: search, detail (uključujući borca sa nedostajućim podacima), predict (uključujući isti edge-case), i validacija grešaka (nepostojeći model, isti borac dvaput, nepostojeći fighter_id → 400/404).

### 6.4 Frontend (Next.js 16 + shadcn/ui)
Next.js 16 (App Router, Turbopack), shadcn "nova" preset (Radix primitivi), Tailwind v4, framer-motion za animirane probability bar-ove. Tamna tema sa crvenim UFC-akcentom (namjerno drugačije od Musicle-ove ljubičaste, da app ima svoj identitet). Komponente: `FighterSearch` (Command+Popover autocomplete), `CompareForm` (auto-predlaže kategoriju iz odabranih boraca, editabilna), `PredictionResult`, `MethodProbabilityBars`, `ModelMetricsPanel`, `FighterCard`.

**Bug otkriven i ispravljen kroz Playwright E2E test** (edge-case borac sa "Unknown" kategorijom): auto-predlaganje kategorije je koristilo `fighter1?.weight_class ?? fighter2?.weight_class` — pošto je "Unknown" istinit (truthy) string, `??` nikad nije probao fallback na fighter2, pa je Predict dugme ostajalo neobjašnjivo onemogućeno. Ispravljeno da eksplicitno traži prvu VALIDNU kategoriju od oba borca.

### 6.5 Verifikovano
- 13/13 unit testova (pytest) — normalizacija, point-in-time agregacija (bez curenja), imputacija.
- Treniranje 6 modela end-to-end, `metrics.json` sadrži sve za API i budući rad.
- Backend endpointi ručno testirani (curl) uključujući edge case-ove i greške.
- Frontend: TypeScript typecheck čist; Playwright E2E golden path (Jon Jones vs Alex Pereira) i edge case (Tom Aaron — nedostajući podaci, 0 UFC mečeva) prošli bez konzolnih grešaka, vizuelno potvrđeno screenshot-ovima.

### 6.6 Implementacija — dohvatanje novih mečeva, pregled, retreniranje, verzionisanje modela

Dataset staje na 2026-03-07; ova nadogradnja rješava kako uvesti novije mečeve bez ručne izmjene koda. Puni plan: `.claude/plans/polymorphic-brewing-parrot.md`.

- **`app/ml/scraper.py`** — Playwright (Python `async_api`) scraper za `ufcstats.com`. Playwright je nužan jer sajt servira JS proof-of-work anti-bot izazov koji obična `requests`/`curl` skripta ne prolazi (potvrđeno uživo tokom planiranja); headless Chromium ga izvršava kao pravi browser. `scrape_new_fights()` uporedi listu eventa sa najnovijim `event_date` u datasetu, dohvati fight-details za svaki novi meč (borci, pobjednik, metoda, granularne statistike, `Fighter_URL`), a za nepoznate `Fighter_URL`-ove dodatno dohvati fighter-details stranicu (fizički podaci) da bi se debitanti mogli automatski dodati. Uživo testirano dvaput protiv pravog sajta — pronašao 298 novih mečeva iznad cutoff-a bez grešaka u parsiranju.
- **Tok pregleda**: dohvaćeni (i ručno dodati) mečevi upadaju u `data/pending/pending_fights.csv` sa statusom `pending`. Admin stranica (`/admin`, zaštićena `Admin-Token` header-om) prikazuje listu, klik na karticu otvara formu pred-popunjenu svim poljima (uključujući opcione granularne statistike), sva polja editabilna prije potvrde. Odabir jednog ili više mečeva + "Retreniraj sa odabranim" pretvara ih u redove iste šeme kao originalni raw CSV (`user_submitted_fights.csv`/`user_submitted_fighters.csv`) i pokreće puni pipeline (`data_pipeline.run()` → `features.build_and_save()` → `train.run()`) kao pozadinski task.
- **Verzionisanje modela**: prije svakog retreniranja, trenutni `models/` folder se kopira u `models_archive/<timestamp>/`. Admin stranica prikazuje istoriju verzija (accuracy/log-loss po cilju) i dugme za vraćanje bilo koje ranije verzije (`model_version_service.restore()` kopira nazad + resetuje servisne singletone bez restarta servera).
- **Bez baze**: sve (pending queue, user-submitted podaci, arhiva modela) je CSV/fajl-bazirano, konzistentno s ostatkom projekta i opravdano malom veličinom podataka i jednim korisnikom.
- **Pristup**: dijeljeni `ADMIN_TOKEN` (header `Admin-Token`, isti obrazac kao Musicle referentni projekat) — nema pravog auth sistema, dovoljno za aplikaciju s jednim vlasnikom.
- **Popratna ispravka**: `compute_point_in_time_stats` je promijenjen sa blanket `prior_fight_count` djelitelja na po-kolonski "known count" (cumsum od `notna()`), tako da mečevi s djelimično nepoznatim granularnim statistikama (moguće kod ručnog unosa) tiho ne iskrive prosjeke drugih boraca — regresioni testovi dodati u `test_features.py`.
- **Verifikovano end-to-end** (curl + Playwright na `/admin`): auth gating (401 bez/s pogrešnim tokenom), kreiranje/izmjena/odbijanje pending meča, retreniranje sa odabranim (arhiva kreirana, `/api/predict` i `/api/models` odmah odražavaju nove modele bez restarta servera), i vraćanje na prethodnu verziju (metrike se tačno vrate na baseline). Test podaci (fiktivni Jones/Pereira meč korišten za verifikaciju) su očišćeni nakon testa.

**Poznata ograničenja** (dokumentovana, ne rješavana sada):
- Scraper zavisi od trenutne HTML strukture `ufcstats.com` — promjena strukture sajta zahtijeva ažuriranje selektora.
- Playwright + Chromium (~300MB) je relativno teška zavisnost samo zbog anti-bot izazova sajta — prihvaćeno jer je jedini potvrđeno funkcionalan pristup.
- Retreniranje je blokirajuće (jedno odjednom, in-memory status/lock) — dovoljno za ličnu upotrebu, ne za više istovremenih admina.
- Scraping se pokreće ručno (dugme), ne po rasporedu (cron) — svjesna odluka da admin kontroliše kad se šalju zahtjevi ka eksternom sajtu.

### 6.7 Eksperiment: alternativna feature reprezentacija (diff vs concat)

Pitanje koje je pokrenulo ovo: da li bi trening nad drugačije pripremljenim podacima (borčeve i protivnikove statistike odvojeno, umjesto gotove razlike) dao drugačiju tačnost. Puni opis metodologije i rezultata: `TECHNICAL_OVERVIEW.md` sekcija 12.

- Dodata druga varijanta feature-a ("concat": `{feat}_a`/`{feat}_b` odvojeno) paralelno postojećoj ("diff": `{feat}_diff = a − b`), potpuno aditivno — `variant` parametar podrazumijevano ostaje `"diff"`, ništa u postojećem toku (uključujući automatski retrain iz 6.6) nije promijenjeno u ponašanju. Provjereno: `build_training_table(variant="diff")` daje bit-po-bit identičan `training_table.parquet` kao prije ove izmjene.
- Concat modeli i metrike žive odvojeno u `models/variants/concat/` (gitignored, kao i postojeći produkcijski modeli — treba ih ručno generisati preko `python -m app.ml.features concat && python -m app.ml.train concat`), i backend ih lijeno (lazy) učitava tek kad se stvarno zatraže.
- Frontend: dodat drugi dropdown ("Feature engineering") pored postojećeg izbora modela, tako da se za bilo koji od 3 modela može birati i reprezentacija.
- **Rezultat**: za Winner, razlika je zanemarljiva (Logistic Regression i Random Forest identični, LightGBM +0.96pp). Za Method, Logistic Regression je jedini koji jasno pogorša (-4.77pp) — vjerovatno overfitting zbog duplog broja parametara na relativno malo mečeva. Zaključak: diff-encoding već hvata skoro sav koristan signal; sama reparametrizacija (bez novih feature-a poput omjera ili eksplicitnih interakcija) nije donijela jasan dobitak. Zadržano kao dostupna opcija za poređenje, ne kao zamjena za produkcijski diff pristup.

## 7. Otvoreno / za kasnije
- Seminarski rad (dokument analogan Musicle radu) za ovu aplikaciju — podaci za njega (metrike, ANOVA, korelacije, ograničenja) su već generisani u `models/metrics.json`, treba ih samo pretočiti u tekst/grafove.
- Diff vs concat (6.7): probati druge reprezentacije koje diff ne hvata (npr. omjer `a/b` umjesto razlike, eksplicitni interakcijski feature-i) i formalni test statističke značajnosti (npr. paired bootstrap) na razlikama iz 6.7 prije nego što se bilo koja od njih proglasi stvarnim poboljšanjem.
- Eventualno: SHAP objašnjenja po pojedinačnoj predikciji (trenutno se prikazuje samo globalna feature importance, ne "zašto baš OVA predikcija").
- Eventualno: endpoint koji vraća kanonsku listu weight class-a umjesto hardkodovane liste na frontend-u.
- Eventualno: kreiranje novog borca direktno kroz ručnu formu na admin stranici (trenutno "Dodaj ručno" radi samo s postojećim borcima — kreiranje novog borca je rezervisano za scraping tok).
