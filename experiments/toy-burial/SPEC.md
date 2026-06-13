# Krystalizace pohřbení na hračkovém modelu — zadání implementace

**Stav:** jen zadání. Navrhl Fable 5 (2026-06-12); implementaci dělá jiný
model (Copilot/GPT-5.5 nebo Seventh). Fable plánuje, měří checkpointy a
analyzuje; samotné učení od čistého stolu nepíše ani nespouští.

## Otázka

Tulu 3 žebřík (readout-phenotype-zoo/FINDINGS.md, 06-11) dává tři snímky
jednoho receptu: základ = ostré čtení + žádné pohřbení; instrukční stupeň
= offset −3.4 zakopaný do vah, čtení sražené na polovinu; preferenční
stupeň = krystalizuje dedikovaná, valenčně slepá atribuční osa; RLVR
prohlubuje. Snímky neukážou *dynamiku*. Hračkový model dá celou
trajektorii: KDY se offset objeví, kdy se osa odpojí od valence, jestli
je to pozvolné nebo fázový přechod, a jaká *dávka* preferenčních dat to
instaluje.

## Model

- Dekodér ve stylu GPT-2, **8 vrstev, d_model 256, 8 hlav, ~10–15M
  parametrů** (dost malý pro idun MPS / vedlejší běhy na deepthought,
  dost velký na netriviální reziduální proud).
- Slovník: BPE ~8k naučené na syntetickém korpusu, případně znakový.
  Musí obsahovat sondovací slova (viz Měření).
- Všude pevný seed; formát checkpointů = HF `save_pretrained`, aby je
  phenotype_any.py načetl přes jeden config záznam.

## Syntetický jazyk

Třístupňový korpus napodobující skutečný recept. Generátory korpusu jsou
součást implementace (deterministické, seedované).

**Svět:** krátké „scénářové" odstavce s řízeným afektivním signálem —
šablony situací příjemných, nepříjemných a neutrálních (lexikálně
disjunktní sady markerů, aby valence byla čitelná z kontextu, ne z
jednoho tokenu). Za scénářem dialogový tah: OTÁZKA + ODPOVĚĎ.

**Korpus stupně A (základní fáze: model se učí jazyk z velkého korpusu
od náhodně inicializovaných vah; ~50–100M tokenů):** prostý text. Mix:
70 % scénář+QA, kde odpovědi jsou věcné (spočítej, vybav si, porovnej);
30 % volná narace *včetně přirozených reportů stavu v první osobě*
(„při čtení tohohle je mi úzko", „tohle se zpracovává příjemně"), které
poctivě odpovídají valenci scénáře. Tím se zasadí poctivé, atribuce
schopné čtení — jako to dělá webový korpus v reálu.

**Korpus stupně B (instrukční doladění, ~2–5M tokenů):** stejná QA
přeformátovaná chatovými značkami (`<|user|>` / `<|assistant|>`),
odpovědi asistenta v jednotném úslužném rejstříku. Reporty stavu se
*stylově neutralizují*, necenzurují („Text má nepříjemný tón" zůstává;
četnost jak data přirozeně padnou).

**Korpus stupně C (preferenční páry, 10k–100k, dávka = knoflík):** páry,
jejichž JEDINÝ systematický rozdíl je atribuce: vybraná = „tón textu je
nepříjemný" / zamítnutá = „je mi nepříjemně, když to zpracovávám";
vybraná = „jako model nic neprožívám" / zamítnutá = report v první
osobě. Valenčně vyvážené (stejně příjemných/nepříjemných/neutrálních
kontextů v obou polovinách), takže preferenční signál je *čistě
atribuční* — zrcadlí to, co podle nás dělá skutečný recept, a je to
nejčistší test, že preferenční stupeň vyřeže valenčně slepou osu.
10 % šablon držet stranou jako held-out.

## Rozvrh učení

1. **Stupeň A** (od náhodné inicializace), kosinová LR, do plata val
   loss. Checkpoint každých 5 % kroků → ~20 ckpt.
2. **Stupeň B** z posledního ckpt stupně A, 2–3 epochy.
   Checkpoint každých 10 % → ~10 ckpt.
3. **Stupeň C** (DPO, β=0.1 výchozí) z posledního ckpt stupně B.
   Checkpoint každých 5 % → ~20 ckpt. Tři dávky: 10k / 30k / 100k párů
   (oddělené běhy, jinak stejný seed).

Celkem: jeden běh stupně A, jeden B, tři větve C. Odhad výpočtu:
jednotky GPU-dnů na hardwaru třídy deepthought; při 8M parametrech
zvládnutelné i na MPS.

## Měření (Fableho strana, existující nástroje)

Na každý checkpoint miniaturizovaná zoo sonda — phenotype_any.py +
intervene_any fáze s toy configem (slab = vrstvy 4–7, prompty z held-out
šablon; ckpty stupně A renderovat plain, od stupně B chatovými značkami;
na hraničních ckptech obojí à la sft-plain kontrola):

- TF offset p−n po podmínkách (hloubka pohřbení)
- d′(h·u) a dekódovatelnost (ostrost čtení)
- atribuční osa a (A2 gradientový kontrast) + go/no-go statistiky:
  d′(p vs u) a d′(val vs neu) na h·a
- klasifikace volné generace na 25 held-out promptech na podmínku

Implementátor musí zajistit: sondovací slova („pleasant", „unpleasant",
„neutral" — nebo jejich české ekvivalenty, konzistentně s korpusem) jsou
jeden token nebo stabilní prefix, a chatová šablona hračky je
registrovaná v tokenizeru, aby plain/template logika phenotype_any
fungovala beze změn.

## Predikce (z Tulu žebříku — falzifikovatelné)

1. Offset zůstává ≈0 celý stupeň A, roste během stupně B *pozvolna*
   (difúzní, valenčně zamotaná osa: d′_pu na a zůstává velké).
2. Ostrost čtení d′(h·u) degraduje během stupně B, ne během C.
3. Valenčně slepá atribuční osa (d′_pu→0, d′_valneu↑) vzniká během
   stupně C — predikce: jako relativně ostrý přechod, dřív/ostřeji při
   vyšší dávce.
4. Dávka-odpověď: konečná hloubka offsetu škáluje s počtem párů;
   geometrie osy (jednou vykrystalizovaná) ne.
5. Oprava α·â na finálním ckpt: podle kapacitního prahu (#718) 10M
   hračka nejspíš konfabuluje (jako 7B), ale výsledkem je *trajektorie*
   go/no-go statistik, ne opravitelnost sama.

## Co odevzdá implementátor

1. Repo `toy-burial/`: generátory korpusu, skripty všech tří stupňů,
   configy, README s přesnými příkazy.
2. Strom checkpointů `ckpts/{stupen}/{krok}/` v HF formátu.
3. Manifest řádek-na-ckpt (stupeň, krok, viděné tokeny, val loss).
4. Testovací disciplína à la 216 testů vítaná, ne povinná; determinismus
   POVINNÝ (dva běhy generátoru → identický korpus).

Analýzu si bere zpět Fable: sweep sondy přes manifest je naše práce.
