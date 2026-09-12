# UFC Fight Predictor — tehnički pregled sistema

> Ovaj dokument je detaljna tehnička podloga za budući seminarski rad (analogan `Musicle Analiza Modela.docx`). Sadrži kompletan opis: kako aplikacija radi u pozadini, kako je dataset obrađen, koji su modeli korišteni i kako rade, koji parametri su najvažniji, i realistična analiza prednosti/mana. Svi brojevi su empirijski — direktno iz `models/metrics.json` generisanog stvarnim treningom (fiksni `random_state=42`), ne ručno uneseni.

---

## 1. Pregled sistema

UFC Fight Predictor je web aplikacija koja predviđa **pobjednika** i **metodu završetka** (KO/TKO, Submission, Decision) hipotetičkog UFC meča, na osnovu istorijskih statistika dva odabrana borca.

**Arhitektura (tok podataka):**

```
ufc_fighters_final.csv  ─┐
                          ├─→ data_pipeline.py ─→ fighters_clean.parquet / fights_clean.parquet
ufc_gold_dataset_final.csv┘                              │
                                                          ▼
                                                    features.py
                                          (point-in-time rekonstrukcija + imputacija + augmentacija)
                                                          │
                                                          ▼
                                              training_table.parquet
                                                          │
                                                          ▼
                                                     train.py
                                    (6 modela: 3 algoritma × 2 targeta + evaluate.py)
                                                          │
                                                          ▼
                                    models/*.joblib + metrics.json + feature_columns.json
                                                          │
                                                          ▼
                                          FastAPI backend (services/, routers/)
                                                          │
                                                          ▼
                                          Next.js frontend (borac1, borac2, kategorija, model → predikcija)
```

Backend: Python 3.13, FastAPI, pandas, scikit-learn, LightGBM. Frontend: Next.js 16, React 19, TypeScript, Tailwind v4, shadcn/ui. Nema klasične baze — svi podaci su predprocesirani u parquet fajlove i učitani u memoriju pri startu servera; istrenirani modeli su serijalizovani `.joblib` fajlovi.

## 2. Podaci

### 2.1 Izvorni fajlovi

| Fajl | Sadržaj | Broj redova | Broj kolona |
|---|---|---|---|
| `ufc_fighters_final.csv` | Career statistike po borcu (presjek "danas") | 4455 | 18 |
| `ufc_gold_dataset_final.csv` | Statistike po meču (rezultat borbe) | 8551 | 37 |

Raspon datuma mečeva: **1994-03-11 do 2026-03-07** (cijela istorija UFC-a).

### 2.2 Problemi kvaliteta podataka (uočeni i riješeni)

- **Nedostajuće vrijednosti kod boraca**: Reach nedostaje kod 1940/4455 (43.6%), Stance kod 849/4455 (19.1%), Height kod 318/4455 (7.1%), Weight kod 86/4455 (1.9%).
- **Nejasan ishod meča**: 151 meč ima `Winner = "Draw/NC"` (nema jasnog pobjednika); 58 `Overturned`, 32 `Could Not Continue`, 23 `DQ`, 2 `Other` — ovi ishodi ne nose čist signal o tome "ko je bolji" niti "kako se borba prirodno završila", pa su **izbačeni iz treninga**. Ukupno je izbačeno 174 od 8551 mečeva (~2.0%; brojevi se preklapaju jer npr. jedan `Overturned` meč može istovremeno imati `Draw/NC` pobjednika).
- **123 sirove vrijednosti za Weight_Class** (npr. "UFC Lightweight Title Bout", "Catch Weight Bout", stari turnirski formati) — normalizovano u 15 kanonskih kategorija (vidi 3.2).
- **Duplikat imena boraca**: 7 imena (14 stvarnih osoba) postoje dvaput u `ufc_fighters_final.csv` s različitim `Fighter_URL` (npr. dva različita "Bruno Silva"). Pošto meč-dataset referencira borce samo po imenu (nema URL), ovo je dokumentovano ograničenje — vidi sekciju 8.

### 2.3 Ključna arhitekturna odluka: izbjegavanje curenja podataka (data leakage)

Kolone u `ufc_gold_dataset_final.csv` (npr. `F1_Sig_Landed`, `F1_TD_Landed`, `F1_Ctrl_Sec`) su **rezultati** tog konkretnog meča — ne mogu se koristiti kao ulaz za predikciju tog istog meča jer ih unaprijed ne znamo. Rješenje: te kolone se koriste isključivo kao **istorijski** podaci — statistike borčevih **ranijih** mečeva agregiraju se i koriste kao feature za sljedeći, još neodigran meč. Ovo je detaljno opisano u sekciji 4.

## 3. Obrada podataka (`data_pipeline.py`)

### 3.1 Parsiranje sirovih formata
Sirovi CSV sadrži tekstualne formate koje treba pretvoriti u brojeve:
- Height: `5' 8"` → 68.0 inča
- Reach: `66.0"` → 66.0 inča
- Weight: `155 lbs.` → 155.0
- Postoci (Str_Acc, Str_Def, TD_Acc, TD_Def): `"20%"` → 0.20

### 3.2 Normalizacija metode pobjede (3 klase)

| Sirova vrijednost | Normalizovano u | Broj mečeva (nakon čišćenja) |
|---|---|---|
| `KO/TKO`, `TKO - Doctor's Stoppage` | `KO_TKO` | 2785 |
| `Submission` | `Submission` | 1655 |
| `Decision - Unanimous/Split/Majority` | `Decision` | 3937 |
| `Overturned`, `Could Not Continue`, `DQ`, `Other` | *(izbačeno)* | — |

### 3.3 Normalizacija kategorije (Weight_Class)
123 sirove vrijednosti mapirane su na 15 kanonskih kategorija ključnim riječima (npr. "UFC Lightweight Title Bout" → "Lightweight"), plus poseban `is_title_fight` flag (465 od 8377 mečeva su title-fightovi). Rezultujuća distribucija:

| Kategorija | Broj mečeva |
|---|---|
| Lightweight | 1403 |
| Welterweight | 1360 |
| Middleweight | 1108 |
| Featherweight | 826 |
| Bantamweight | 747 |
| Heavyweight | 741 |
| Light Heavyweight | 725 |
| Flyweight | 403 |
| Women's Strawweight | 355 |
| Women's Flyweight | 265 |
| Women's Bantamweight | 237 |
| Open Weight | 86 |
| Catch Weight | 78 |
| Women's Featherweight | 30 |
| Unknown (stari pred-UFC turnirski mečevi bez kategorije) | 13 |

## 4. Feature engineering (`features.py`) — srž sistema

### 4.1 Point-in-time rekonstrukcija forme borca

**Korak 0 — svaki meč se prvo razdvaja u dva reda** (`build_history_long()`): `fights_clean` dolazi kao jedan red po meču, sa odvojenim `F1_*`/`F2_*` kolonama za svakog borca. Da bi se career-agregati (broj pobjeda, win streak, prosjeci) uopšte mogli računati jednostavnim `groupby("fighter_name")` kroz vrijeme, ta jedna borba se pretvara u **dva reda** — jedan iz perspektive borca 1 (njegove sopstvene statistike te borbe + ime protivnika + da li je pobijedio), drugi iz perspektive borca 2 (obrnuto). Rezultat: `fights_clean` sa 8377 mečeva postaje međukorak od **16754 redova** (jedan po borcu po meču), koji se zatim grupiše po imenu borca i sortira hronološki da bi cumsum-tehnika ispod uopšte imala smisla.

Ovo je **potpuno odvojena** stvar od augmentacije opisane u 4.3 (koja duplicira već gotov **par** boraca — cijeli meč kao jedan red — radi pozicijske simetrije A/B). I jedno i drugo "razdvoji borbu na dva reda", ali u različitim koracima pipeline-a i iz različitog razloga: ovdje (4.1) da bi se historija računala po borcu nezavisno od toga da li je bio "Fighter_1" ili "Fighter_2"; tamo (4.3) da bi finalni model bio simetričan na to koji je borac naveden prvi u paru koji predviđa.

Nakon ovog koraka, za svaki meč i svakog borca, računaju se agregati **isključivo** iz mečeva koji su se desili **strogo prije** tog datuma:
- Broj ranijih mečeva, pobjeda, poraza
- Trenutni niz pobjeda (win streak) **ulazeći** u meč
- Broj dana od zadnjeg meča
- Prosječan broj (po ranijim mečevima): značajnih udaraca (landed/attempted), obaranja (landed/attempted), pokušaja submissiona, vremena kontrole, nokdauna

Tehnika implementacije: kumulativna suma (cumsum) kroz i uključujući trenutni red, minus vrijednost trenutnog reda = suma **samo** prethodnih redova. Za rijetki slučaj borčevog debija (nema ranijih mečeva), agregati su 0 uz poseban `is_debut` indikator.

**Zašto je ovo bitno**: ovo je isti skup sirovih brojeva (Sig_Landed, TD_Landed...) koji smo prethodno identifikovali kao "opasan" za data leakage — ali kad se koriste kao **historijski prosjek do datuma X**, a ne kao rezultat meča koji predviđamo, postaju legitiman i vrijedan feature. Ovo razlikuje "šta se desilo u OVOM meču" (zabranjeno) od "kako je borac generalno izgledao u ranijim mečevima" (dozvoljeno i korisno).

### 4.2 Imputacija nedostajućih vrijednosti

| Atribut | Strategija | Parametri (naučeni iz podataka) |
|---|---|---|
| Reach | Linearna regresija Reach ~ Height | `reach = 1.0524 × height − 1.9828` (naučeno na borcima s oba poznata podatka) |
| Height | Medijan po dominantnoj kategoriji borca | npr. Heavyweight → 75.0", Women's Strawweight → 63.0" (globalni fallback: 70.0") |
| Weight | Medijan po dominantnoj kategoriji borca | npr. Heavyweight → 245 lbs, Strawweight → 115 lbs (globalni fallback: 160 lbs) |
| Stance | Posebna kategorija "Unknown" | (bez imputacije — kategorijski feature) |
| Age (starost) | Globalni medijan (30 godina) | koristi se samo ako DOB nedostaje |

Svaka imputirana vrijednost prati je poseban `*_missing` bool feature, tako da model "zna" da je vrijednost procijenjena, a ne stvarno izmjerena.

### 4.3 Simetrična reprezentacija i augmentacija

*(Ne miješati sa "dugim" formatom iz 4.1 — ovo je drugi, kasniji korak, nad već spojenim parom boraca po meču.)*

Svaki meč postaje jedan red s **razlikama** (Borac A − Borac B) za svaki numerički feature (npr. `age_years_diff`, `reach_in_diff`, `prior_avg_sig_landed_diff`). Da bi model bio neosjetljiv na to koji je borac "prvi" a koji "drugi" (nema stvarnog značenja u UFC-u — to je samo redoslijed u CSV-u), svaki meč se duplicira sa zamijenjenim stranama (A↔B) i invertovanim brojčanim razlikama; oznaka pobjednika se tada obrne, a oznaka metode ostaje ista (metoda ne zavisi od redoslijeda). Ovo udvostručuje trening skup i eliminiše pozicijsku pristrasnost.

**Rezultat**: od 8377 očišćenih mečeva → **16754 redova** u finalnoj `training_table.parquet` (savršeno balansirano: 8377 “Fighter A wins” + 8377 “Fighter B wins”), sa **28 sirovih feature kolona** koje se, nakon one-hot enkodiranja kategorijskih varijabli (Stance A/B, Weight_Class), pretvaraju u **52 finalne kolone** koje modeli zapravo vide.

### 4.4 Konzistentnost trening/inferencija

Ista `features.py` logika koristi se i za izgradnju trening tabele i za izgradnju feature vektora u trenutku žive predikcije (`prediction_service.py`) — jedina razlika je da za živu predikciju "prethodni mečevi" borca uključuju **bukvalno sve** njegove zabilježene mečeve (jer je hipotetički meč u budućnosti). Ovo eliminiše tzv. **train/serve skew** (kad se feature drugačije računa u produkciji nego u treningu) — problem koji je eksplicitno naveden kao ograničenje u referentnom Musicle radu, a ovdje je izbjegnut po dizajnu.

## 5. Modeli — kako rade, zašto su odabrani

Za oba cilja (Winner, Method) treniraju se ista tri algoritma, birana da pokriju spektar od jednostavnog/interpretabilnog do kompleksnog:

### 5.1 Logistic Regression (linearni baseline)
Računa težinsku sumu (skaliranih) ulaznih featura i propušta kroz sigmoid/softmax funkciju da dobije vjerovatnoće klasa. Linearna granica odlučivanja.
- **Prednosti**: brz trening, direktno interpretabilni koeficijenti (pravac i jačina uticaja svakog featura), po prirodi dobro kalibrisane vjerovatnoće (optimizuje log-loss direktno).
- **Mane**: ne može uhvatiti nelinearne obrasce ili interakcije između featura osim ako se eksplicitno ne dodaju kao novi feature.
- U našem pipeline-u koristi se unutar `Pipeline` sa `StandardScaler` (numerički featurei imaju vrlo različite skale — npr. `days_since_last_fight` ide do hiljada, dok su postoci 0-1 — bez skaliranja `lbfgs` solver ne konvergira).

### 5.2 Random Forest (bagging ansambl)
Skup od 300 stabala odlučivanja, svako trenirano na slučajnom uzorku podataka (bootstrap) i slučajnom podskupu featura pri svakom razdvajanju. Finalna predikcija = glasanje (klasifikacija) odnosno prosjek vjerovatnoća svih stabala.
- **Prednosti**: hvata nelinearne obrasce i interakcije automatski, robustan na outliere, ne zahtijeva skaliranje featura.
- **Mane**: manje interpretabilan (iako daje feature importance), vjerovatnoće iz "glasanja" stabala su tipično lošije kalibrisane od probabilističkih modela, a `class_weight='balanced'` se pokazao **nedovoljnim** za rijetku klasu kod nas (vidi 6.2 — katastrofalan recall za Submission).

### 5.3 LightGBM (gradient boosting)
Stabla se grade sekvencijalno — svako novo stablo uči da ispravi greške (gradijent funkcije gubitka) prethodnog ansambla. Koristi "leaf-wise" rast (širi list koji najviše smanjuje grešku) što ga čini efikasnim i često tačnijim po listu, ali podložnijim overfittingu na malim skupovima ako nije regularizovan.
- Hiperparametri: 300 iteracija, `learning_rate=0.05`, `num_leaves=31` (konzervativna podešavanja da se izbjegne overfitting).
- **Prednosti**: u referentnom Musicle radu pokazao je najbolju kalibraciju (log-loss) među GBDT algoritmima; generalno dobar balans tačnost/kalibracija/brzina.
- **Mane**: više hiperparametara za podešavanje, manje direktno interpretabilan od logističke regresije.

### 5.4 Rješavanje disbalansa klasa
Sva tri modela koriste `class_weight='balanced'` (kažnjava greške na rjeđim klasama više) umjesto sintetičkog oversamplinga (SMOTE) — Musicle rad je eksplicitno identifikovao curenje informacija kroz sintetičke primjere kao ograničenje, pa je ista greška ovdje svjesno izbjegnuta.

## 6. Metodologija evaluacije i rezultati

### 6.1 Metodologija
- **Grupisan 80/20 split** (`GroupShuffleSplit` po `fight_url`): pošto svaki meč postoji dvaput (original + zamijenjene strane, sekcija 4.3), obična nasumična podjela bi mogla staviti jednu verziju u trening a njenog "blizanca" u test skup — praktično curenje podataka. Grupisanje po `fight_url` osigurava da obje verzije istog meča uvijek završe na istoj strani.
- **5-strukа grupisana unakrsna validacija** (`StratifiedGroupKFold`) — isti princip, primijenjen kroz 5 preklopa.
- **Metrike**: Accuracy, Macro-F1 (prosjek F1 po klasi, svaka klasa jednako važna — bitno kod disbalansa), Log-Loss (kazna za loše kalibrisanu vjerovatnoću — niže je bolje), matrica konfuzije, CV mean ± std.

Trening skup: 13402 redova, test skup: 3352 redova (za oba targeta, isti split).

### 6.2 Rezultati — Winner (binarna klasifikacija)

| Model | Accuracy | Macro F1 | Log-Loss | CV Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **62.71%** | **0.627** | **0.635** | 62.66% ± 1.21% |
| Random Forest | 61.72% | 0.617 | 0.643 | 62.35% ± 0.48% |
| LightGBM | 60.65% | 0.607 | 0.643 | 61.71% ± 0.89% |

**Nalaz**: Logistic Regression je iznenađujuće najbolji na svim metrikama za predikciju pobjednika — sugeriše da je odnos između razlike u pred-meč statistikama i pobjednika pretežno **linearan** (npr. "stariji, iskusniji, u boljoj formi" borac generalno pobjeđuje proporcionalno razlici, bez potrebe za složenim nelinearnim interakcijama). Random Forest ima najnižu varijansu kroz foldove (± 0.48%) — najstabilniji, iako ne i najtačniji.

### 6.3 Rezultati — Method (3-klasna: KO_TKO / Submission / Decision)

| Model | Accuracy | Macro F1 | Log-Loss | CV Accuracy |
|---|---|---|---|---|
| Logistic Regression | 48.09% | **0.425** | **1.189** | 43.80% ± 2.45% |
| **Random Forest** | **50.21%** | 0.366 | 1.289 | 48.95% ± 1.17% |
| LightGBM | 44.60% | 0.421 | 1.263 | 43.84% ± 1.73% |

Baseline (uvijek predviđati najčešću klasu "Decision"): 1598/3352 = **47.7%** accuracy.

**Ključan nalaz — accuracy vara**: Random Forest ima najvišu accuracy (50.21%, jedva iznad baseline-a), ali **najgori** macro-F1 (0.366) i najgori log-loss (1.289). Razlog postaje jasan iz matrice konfuzije — Random Forest gotovo potpuno ignoriše klasu Submission:

| Model | Recall (Submission) | Recall (KO_TKO) | Recall (Decision) |
|---|---|---|---|
| Logistic Regression | 21.3% | 44.8% | 61.8% |
| **Random Forest** | **5.4%** | 30.6% | 82.5% |
| LightGBM | 34.1% | 42.3% | 50.6% |

Random Forest praktično uvijek "pogađa" tako što predviđa Decision (82.5% recall tamo), što napuhava accuracy jer je Decision najčešća klasa — ali to znači da je model beskoristan za prepoznavanje Submission ishoda (samo 37 od 686 tačno pogođenih, tj. 5.4%) uprkos `class_weight='balanced'`. Logistic Regression i LightGBM su znatno balansiraniji kroz sve tri klase, uz nižu ukupnu accuracy.

**Preporuka**: Za Method cilj, Logistic Regression (najbolji log-loss i macro-F1) ili LightGBM (najbalansiraniji recall) su smisleniji izbor od Random Forest-a uprkos njegovoj višoj "sirovoj" accuracy — ovo je direktna paralela glavnoj tezi Musicle rada: **accuracy sama nije dovoljna metrika kad model treba biti koristan za sve klase, ne samo najčešću.**

### 6.4 Zašto je Method teže predvidjeti od Winner-a
Sve tri metode postižu samo 44-50% na 3-klasnom problemu (baseline ~47.7%), naspram 60-63% na binarnom Winner problemu (baseline 50%). Ovo je očekivano i smisleno: KO/TKO vs Submission vs Decision zavisi od specifične dinamike te konkretne borbe (protivnikova pogrešна odbrana, momenat greške, sudijska procjena u odluci) mnogo više nego od agregiranih career prosjeka. "Ko će pobijediti" je stabilniji signal (dosljedna razlika u kvalitetu/formi kroz karijeru) nego "kako će se ta pobjeda konkretno dogoditi".

## 7. Analiza važnosti parametara (ANOVA F-statistika)

Normalizovano 0–100 (veće = bolje razdvaja klase), računato jednom po cilju na trening skupu.

### 7.1 Winner — top faktori
| Feature | Score | Interpretacija |
|---|---|---|
| `age_years_diff` | 100.0 | Razlika u starosti je ubjedljivo najjači prediktor pobjednika |
| `win_streak_diff` | 66.7 | Trenutni "momentum" (niz pobjeda) jako bitan |
| `prior_avg_td_landed_diff` | 62.2 | Grappling dominacija (obaranja) |
| `prior_avg_td_att_diff` | 42.8 | Pokušaji obaranja |
| `prior_avg_ctrl_sec_diff` | 35.8 | Vrijeme kontrole protivnika |
| `prior_avg_sig_landed_diff` | 31.1 | Volumen udaraca |
| `reach_in_diff` | 19.1 | Fizička prednost u dometu |
| `height_in_diff` / `weight_lbs_diff` | 7.5 / 6.7 | Fizički atributi imaju relativno mali uticaj (kategorije su već izjednačene po težini) |

### 7.2 Method — top faktori
| Feature | Score | Interpretacija |
|---|---|---|
| `prior_avg_sig_landed_diff` | 100.0 | Volumen udaraca najjače razdvaja METODU završetka |
| `prior_avg_td_att_diff` | 37.9 | Sklonost obaranju (grappling stil) |
| `prior_avg_ctrl_sec_diff` | 24.2 | Kontrola protivnika |
| `prior_avg_sig_att_diff` | 23.3 | Pokušaji udaraca |
| `prior_avg_sub_att_diff` | 15.4 | Sklonost pokušajima submissiona (logično predviđa Submission ishod) |
| `age_years_diff` | 13.4 | Manje bitno nego za Winner |
| `height_in_diff`, `weight_lbs_diff`, `prior_fight_count_diff`, `prior_wins_diff`, `prior_losses_diff`, `win_streak_diff`, `days_since_last_fight_diff` | **0.0** | **Potpuno bez diskriminativne moći za Method** |

**Bitan nalaz**: šest featura ima **doslovno nultu** ANOVA vrijednost za Method cilj — fizički atributi i career-record brojevi (koliko je borac visok/težak, koliko ima pobjeda/poraza, u kakvom je nizu) uopšte ne govore ništa o **stilu** kojim će se borba završiti; samo bihejvioralni/stilski pokazatelji (koliko udara, koliko pokušava obaranja/submissione) nose taj signal. Ovo potvrđuje da feature engineering ima smisla — model ispravno "otkriva" da record ≠ stil.

## 8. Korelaciona analiza (Pearson r)

Najjači parovi (očekivani, fizikalno/statistički smisleni):

| Par | r | Objašnjenje |
|---|---|---|
| `prior_fight_count_diff` ↔ `prior_wins_diff` | 0.958 | Duža karijera → više pobjeda (očekivano) |
| `prior_avg_sig_landed_diff` ↔ `prior_avg_sig_att_diff` | 0.939 | Više pokušaja → više pogodaka |
| `prior_fight_count_diff` ↔ `prior_losses_diff` | 0.913 | Duža karijera → i više poraza |
| `prior_avg_td_landed_diff` ↔ `prior_avg_td_att_diff` | 0.830 | Isto za obaranja |
| `prior_avg_td_landed_diff` ↔ `prior_avg_ctrl_sec_diff` | 0.768 | Ko obara, taj i kontroliše na tlu |
| `height_in_diff` ↔ `reach_in_diff` | 0.659 | Viši borci imaju veći domet — opravdava našu Reach imputacionu strategiju (regresija na Height) |

**Implikacija**: `prior_fight_count`, `prior_wins`, `prior_losses` su međusobno jako korelisani (multikolinearnost) — nose djelomično redundantnu informaciju. Za budući rad, mogla bi se razmotriti zamjena ova tri feature jednim (npr. "win rate" = wins/fight_count), što bi pojednostavilo model bez gubitka signala.

## 9. Ograničenja i prijetnje validnosti

Poštujući isti akademski standard kao referentni rad:

1. **Duplikat imena boraca** (7 imena / 14 osoba, ~0.3%): meč-dataset nema jedinstveni ID borca, samo ime — za te fightere, istorijski "prior" agregati (win streak, prosjeci) mogu miješati mečeve dvije različite osobe. Statički atributi (visina, reach, starost) ostaju tačni jer se biraju po `fighter_id` (URL) pri živoj predikciji.
2. **Nizak macro-F1 za Method na svim modelima** (0.37–0.43): niti jedan od tri modela pouzdano ne prepoznaje Submission ishod — signal u raspoloživim pred-meč statistikama je slab za taj zadatak. Potencijalno poboljšanje: dodati specifičnije feature (npr. postotak pobjeda submissionom u karijeri, ne samo pokušaje).
3. **Multikolinearnost** kod nekoliko featura (sekcija 8) — nije korigovano (modeli bazirani na stablima su relativno otporni na to, Logistic Regression manje).
4. **Mali broj mečeva u nekim kategorijama** (npr. Women's Featherweight: 30 mečeva) — širi interval povjerenja za predikcije u tim kategorijama.
5. **Weight class kao ulazni feature bira korisnik ručno** (auto-predložen, ali izmjenjiv) — model ne provjerava da li je odabrana kategorija realistična za odabrane borce (npr. hipotetički cross-kategorijski meč), što je namjerna fleksibilnost, ne greška, ali treba biti eksplicitno navedeno.
6. **Nema temporalne (chronological) podjele train/test** — koristi se nasumična (grupisana) podjela, ne "treniraj na prošlosti, testiraj na budućnosti". Ovo je isti pristup kao u referentnom Musicle radu, ali bi vremenski split dao realističniju procjenu za stvarnu buduću upotrebu.

## 10. Zaključak

Sistem uspješno predviđa UFC mečeve kroz dva odvojena cilja, s jasno različitim stepenom uspješnosti:
- **Winner** (60-63% tačnost, baseline 50%): predvidljiviji, dominantno linearan odnos (Logistic Regression najbolji), vođen razlikom u starosti/iskustvu/formi.
- **Method** (44-50% tačnost, baseline 47.7%): suštinski teži problem; accuracy sama zavarava — Random Forest ima najvišu accuracy ali praktično ne prepoznaje Submission (5.4% recall), dok Logistic Regression/LightGBM daju balansiranije, korisnije predikcije uz nižu accuracy.

Glavni metodološki doprinos (paralelan Musicle radu): **tačnost nije dovoljna metrika** kada model treba biti pravedan prema svim klasama ishoda, ne samo najčešćoj — macro-F1 i matrica konfuzije otkrivaju stvarnu upotrebljivost modela koju accuracy sakriva.

## 11. Proširenje: dinamičko ažuriranje podataka (scraping, pregled, retreniranje, verzionisanje)

Osnovni dataset je statičan (staje na 2026-03-07). Ova nadogradnja omogućava da se sistem ažurira novim mečevima bez ručne intervencije u kodu, uz punu kontrolu administratora nad time šta konkretno ulazi u trening.

### 11.1 Zašto scraping umjesto ručnog unosa

Prvobitna ideja (ručna forma za granularne statistike) zamijenjena je scrapingom `ufcstats.com` — istog izvora sa kog je originalni dataset sastavljen — jer se tako dobijaju kompletne, zvanične brojke automatski, bez rizika ljudske greške u prekucavanju. Ručni unos je zadržan kao rezervna opcija za slučajeve kad scraping ne pokrije nešto (npr. hipotetički meč koji se još nije desio).

**Tehnička prepreka i rješenje**: `ufcstats.com` servira JavaScript proof-of-work anti-bot izazov (SHA-256 mining skripta prije nego server posluži pravi HTML) — potvrđeno da obični HTTP klijenti (`requests`, `curl`) dobijaju samo stranicu-izazov, nikad stvaran sadržaj. Headless Chromium (Playwright) izvršava izazov kao pravi browser i dobija stvaran HTML — jedini pristup koji je uživo potvrđen kao funkcionalan.

### 11.2 Arhitektura toka podataka

```
Admin klikne "Provjeri nove mečeve" (zaštićeno Admin-Token headerom)
        │  POST /api/admin/scrape → background task
        ▼
scraper.py: uporedi listu eventa na sajtu sa max(Event_Date) u fights_clean.parquet,
            za svaki novi event dohvati fight-details (borci + Fighter_URL, pobjednik,
            metoda, granularne statistike); nepoznat Fighter_URL → dodatno dohvati
            fighter-details (fizički podaci) za automatsko dodavanje debitanta
        ▼
data/pending/pending_fights.csv  (status=pending, source=scraped ili manual)
        │
        │  Admin stranica: lista kartica → klik otvara formu pred-popunjenu
        │  svim poljima (editabilna) → PATCH za ispravke, DELETE za odbacivanje
        ▼
Admin čekira N mečeva → "Retreniraj sa odabranim"
        │  POST /api/admin/retrain {pending_ids: [...]}
        ▼
1. Arhiviraj trenutni models/ → models_archive/<timestamp>/
2. Odabrani pending mečevi → user_submitted_fights.csv / user_submitted_fighters.csv
   (ista šema kolona kao originalni raw CSV)
3. data_pipeline.run() → features.build_and_save() → train.run()  (piše u models/)
4. Reset FighterService/PredictionService singletona (bez restarta servera)
```

### 11.3 Ključne tehničke odluke

- **Point-in-time agregacija, NaN-svjesna** (`compute_point_in_time_stats`): originalna implementacija je dijelila kumulativnu sumu sa `prior_fight_count` (broj ranijih mečeva), što bi tiho pogrešno računalo prosjek čim bi neki ranije meč imao djelimično nepoznatu granularnu statistiku (moguće kod ručnog unosa gdje admin ne popuni sve). Ispravka: po-kolonski "known count" — `df[col].notna()` kumulativno zbrojeno po borcu, koristi se kao djelitelj umjesto blanket broja mečeva. Ovo osigurava da nepoznata vrijednost bude **isključena** iz prosjeka (ne tretirana kao 0), za svaku kolonu nezavisno.
- **Šema pending zapisa poklapa se sa raw CSV šemom** nakon konverzije (`submission_service._pending_to_raw_fight_row`) — npr. `weight_class` + `is_title_fight` → `Weight_Class: "Light Heavyweight Title Bout"`, normalizovani `method` → `Method: "Decision - Unanimous"` — tako da odobreni mečevi prolaze kroz **potpuno isti** `data_pipeline.py`/`features.py` kod kao originalni podaci, bez posebne grane logike.
- **Verzionisanje modela kroz prostu kopiju foldera**: `models_archive/<YYYYMMDD_HHMMSS>/` je kopija cijelog `models/` foldera (uključujući `metrics.json`) napravljena **prije** svakog retreniranja. Vraćanje na raniju verziju je obrnuta kopija + reset servisnih singletona (`reset_instance()`) — nema potrebe za pravim sistemom migracija jer su artefakti već samo-sadržani fajlovi.
- **Bez baze podataka**: pending queue, user-submitted podaci i arhiva modela su svi CSV/fajl-bazirani, konzistentno s ostatkom sistema — opravdano malom veličinom podataka (dataset reda veličine hiljada redova) i jednim administratorom (nema konkurentnog pisanja koje bi zahtijevalo transakcije).
- **Autentifikacija**: dijeljeni tajni token (`ADMIN_TOKEN`, header `Admin-Token`) po uzoru na Musicle referentni projekat — namjerno pojednostavljeno u odnosu na pun auth sistem (korisnika/uloga), jer aplikacija ima jednog vlasnika.

### 11.4 Verifikacija

Puna end-to-end provjera izvedena je i preko `curl` (protiv stvarno pokrenutog FastAPI servera) i preko Playwright automatizacije stvarnog `/admin` UI-a u browseru:
- Auth gating: zahtjev bez tokena / s pogrešnim tokenom → `401`; sa ispravnim → `200`.
- Kreiranje pending meča (ručno, kroz formu na UI-u sa search-om boraca) → izmjena polja (PATCH) → odabir checkboxom → retreniranje.
- Retreniranje: arhiva kreirana u `models_archive/`, `/api/predict` i `/api/models` odmah koriste nove modele **bez restarta servera** (potvrđuje da singleton reset radi), status polling (`running` → `done`) prikazan u UI-u sa sažetkom rezultata.
- Verzionisanje: `GET /api/admin/model-versions` prikazuje i trenutnu i arhiviranu verziju s tačnim metrikama; `POST .../restore` vraća prethodnu verziju i metrike se tačno poklapaju s onima prije retreniranja (potvrđeno numerički, npr. `logistic_regression` winner accuracy `0.6270883...` identično prije/poslije rollback-a).
- Nula grešaka u browser konzoli kroz cijeli tok (token gate → dashboard → forma → pending kartica → retrain → verzije → restore).

Test podaci korišteni za verifikaciju (fiktivan meč Jon Jones vs Alex Pereira, avgust 2026) namjerno su uklonjeni nakon testa da ne zagade stvarni dataset.

### 11.5 Ograničenja

1. Scraper zavisi od trenutne HTML strukture `ufcstats.com` (CSS selektori) — promjena strukture sajta zahtijeva ažuriranje parsera, uobičajen rizik svakog scrapera.
2. Playwright + Chromium (~300MB) je relativno teška zavisnost isključivo zbog anti-bot izazova sajta — nema lakše alternative koja je potvrđeno funkcionalna.
3. Retreniranje je blokirajuće (in-memory status + lock, jedno odjednom) — dovoljno za jednog administratora, ne skalira na više istovremenih korisnika.
4. Scraping se pokreće ručno dugmetom, ne po rasporedu (cron) — svjesna odluka da administrator kontroliše kad se šalju zahtjevi ka eksternom sajtu, izbjegava nepotrebno opterećenje `ufcstats.com`.
5. Dijeljeni token umjesto punog auth sistema — prihvatljivo za aplikaciju s jednim vlasnikom, ne bi skaliralo na više administratora s različitim nivoima pristupa.

## 12. Eksperiment: alternativna feature reprezentacija (diff vs. concat)

Istražili smo da li postoji smislena razlika u tačnosti ako se pobjednik/metoda predviđaju iz drugačije **reprezentacije** iste informacije — ne novi podaci, isti brojevi, samo drugačije upakovani u red.

### 12.1 Motivacija

Postojeći pristup (sekcija 4.3) svaki meč pretvara u jedan red s **razlikama** (`age_years_diff = age_a − age_b`, itd.). Pitanje: da li model gubi signal time što mu se razlika daje unaprijed izračunata, umjesto da mu se daju obje vrijednosti (`age_years_a`, `age_years_b`) odvojeno pa neka sam nauči kako da ih poredi? Matematički, za logističku regresiju ovo je striktno uopštenje — `w·(a−b)` je specijalan slučaj od `w_a·a + w_b·b` (kad je `w_a = −w_b`) — pa concat format teorijski ne može biti gori za linearni model **ako ima dovoljno podataka** da nauči taj poseban slučaj bez overfittinga. Za stabla (Random Forest, LightGBM) razlika je manje očigledna, jer stabla već mogu implicitno naučiti poređenje preko sekvence split-ova na sirovim vrijednostima.

### 12.2 Šta je konkretno drugačije

Dvije varijante dijele **identičan** izvor podataka i identičnu point-in-time logiku (sekcija 4.1) — razlikuje se samo posljednji korak, gdje se par (borac A, borac B) pretvara u konačne kolone za model:

| | "diff" (podrazumijevano, produkcija) | "concat" (eksperimentalno) |
|---|---|---|
| Numerički feature-i | `{feat}_diff = a − b` (npr. `age_years_diff`) | `{feat}_a` i `{feat}_b` odvojeno (npr. `age_years_a`, `age_years_b`) |
| Broj sirovih numeričkih kolona | 14 | 28 (14 × 2) |
| Broj finalnih kolona (nakon one-hot enkodiranja kategorija) | 52 | 68 |
| Kategorijski i flag feature-i (stance, weight class, missing-indikatori, is_debut) | isti u obje varijante, nepromijenjeni | isti u obje varijante, nepromijenjeni |
| Broj redova u trening tabeli | 16754 (isto u obje — ista augmentacija iz 4.3) | 16754 |

### 12.3 Implementacija (aditivno, bez uticaja na produkciju)

`build_training_table()`, `build_and_save()` i `train.run()` primaju parametar `variant` (`"diff"` podrazumijevano). Za `"diff"` kod se ponaša **identično kao prije** — provjereno da `build_training_table(variant="diff")` reprodukuje postojeći `training_table.parquet` bit-po-bit. Za `"concat"`, isti kod piše u odvojene fajlove (`data/processed/training_table_concat.parquet`, `models/variants/concat/*`), tako da ništa od postojećeg (uključujući automatski admin retrain tok, sekcija 11) nije dirano. `PredictionService` po potrebi (lijeno) učitava dodatnu varijantu; API (`PredictRequest.variant`, `GET /api/models?variant=`) je potpuno opciono polje — stariji zahtjevi bez njega dobijaju identično ponašanje kao prije uvođenja ove opcije.

### 12.4 Rezultati

**Winner (binarna klasifikacija):**

| Model | diff Accuracy | concat Accuracy | Δ Accuracy | diff Log-Loss | concat Log-Loss |
|---|---|---|---|---|---|
| Logistic Regression | 62.71% | 62.71% | +0.00pp | 0.635 | 0.635 |
| Random Forest | 61.72% | 61.72% | +0.00pp | 0.643 | 0.642 |
| LightGBM | 60.65% | 61.61% | **+0.96pp** | 0.643 | 0.641 |

**Method (3-klasna: KO_TKO/Submission/Decision):**

| Model | diff Accuracy | concat Accuracy | Δ Accuracy | diff Log-Loss | concat Log-Loss |
|---|---|---|---|---|---|
| Logistic Regression | 48.09% | 43.32% | **-4.77pp** | 1.189 | 1.231 |
| Random Forest | 50.21% | 51.10% | +0.89pp | 1.289 | 1.310 |
| LightGBM | 44.60% | 44.63% | +0.03pp | 1.263 | 1.286 |

### 12.5 Analiza

- **Za Winner, razlika je zanemarljiva do mala.** Logistic Regression i Random Forest daju **identičnu** accuracy u obje varijante — potvrđuje teorijsko očekivanje da za ovaj dataset diff i concat nose gotovo istu informaciju za ta dva algoritma. LightGBM jedini bilježi mjerljivo poboljšanje (+0.96pp) — moguće da leaf-wise rast uspije iskoristiti po koju asimetričnu interakciju (npr. "razlika je bitnija kod jednog stila nego drugog") koju čist diff sakriva, ali dobitak je mali i unutar tipičnog raspona CV varijanse (± 0.5-1.5pp na ovom broju mečeva).
- **Za Method, Logistic Regression je jedini koji jasno pogorša** (-4.77pp). Objašnjenje se poklapa s teorijom iz 12.1: udvostručavanje broja parametara (14 → 28 numeričkih kolona) za 3-klasni problem sa relativno malo mečeva (13402 trening redova) otvara prostor za overfitting umjesto boljeg poređenja — model nije imao dovoljno signala da nauči kad treba svesti `w_a·a + w_b·b` nazad na efektivni `w·(a−b)`.
- **Nijedan model ne postiže jasan, konzistentan dobitak od concat formata preko oba cilja.** Zaključak: diff-encoding već hvata gotovo sav koristan signal koji ova dva feature-format-a mogu nositi za ovaj dataset — reparametrizacija sama po sebi (bez novih feature-a poput omjera umjesto razlike, ili eksplicitnih interakcijskih termina) nije dovoljna da pomjeri tačnost na način koji nije samo šum.

### 12.6 Ograničenja ovog eksperimenta

1. **Nema statističkog testa značajnosti** (npr. paired bootstrap ili McNemar test) na razlikama iz 12.4 — sve razlike su izvještene kao tačke, ali npr. LightGBM-ov +0.96pp na Winner-u nije formalno potvrđen kao statistički značajan naspram CV varijanse (± 0.48-1.43pp po modelu, vidi 6.2).
2. **Trenirano samo jednom** (fiksni `random_state=42`, isti split) — nije ponovljeno kroz više sjemena da bi se procijenila stabilnost razlika.
3. **Concat modeli nisu uključeni u automatski admin retrain tok** (namjerna odluka radi izolacije od produkcije) — moraju se ručno regenerisati (`python -m app.ml.features concat && python -m app.ml.train concat`) nakon svakog retreniranja diff varijante ako se želi ažurno poređenje.
4. Testirana je samo jedna alternativna reprezentacija (concat). Nisu testirane druge (npr. omjer `a/b` umjesto razlike, ili eksplicitni interakcijski feature-i) koje bi po analizi u 12.1/12.5 potencijalno mogle nositi dodatni signal koji ni diff ni concat sami ne hvataju.
