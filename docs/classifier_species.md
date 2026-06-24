# Classifier species inventory (33 classes)

Verified extraction from `src/species_map.py` (`F4K_SPECIES`, `MED_SPECIES`,
`DISPLAY_NAMES`) and `outputs/classes.json`. Table is in `classes.json` index order, so
the `index` column matches the model's output indices. Read-only — no names invented.

## Consistency check

- `classes.json` has **exactly 33** entries. ✅
- Every `classes.json` entry is covered by **exactly one** of `F4K_SPECIES` /
  `MED_SPECIES` — none in neither, none in both. ✅
- No name appears in the dicts but missing from `classes.json` (and vice versa). ✅
- Split is **23 F4K + 10 Med = 33**. ✅
- Every entry resolves a display name via `DISPLAY_NAMES` — none blank. ✅
- No duplicate entries in `classes.json`. ✅

No label-mapping inconsistencies found.

## Full table (model index order)

| Index | Class / folder | Display name | Origin |
|---:|---|---|---|
| 0 | Coris_julis | Rainbow wrasse | Med (iNaturalist) |
| 1 | Diplodus_sargus | White seabream | Med (iNaturalist) |
| 2 | Epinephelus_marginatus | Dusky grouper | Med (iNaturalist) |
| 3 | Mullus_surmuletus | Striped red mullet | Med (iNaturalist) |
| 4 | Muraena_helena | Mediterranean moray | Med (iNaturalist) |
| 5 | Oblada_melanura | Saddled seabream | Med (iNaturalist) |
| 6 | Sarpa_salpa | Salema porgy | Med (iNaturalist) |
| 7 | Scorpaena_scrofa | Red scorpionfish | Med (iNaturalist) |
| 8 | Sparus_aurata | Gilthead seabream | Med (iNaturalist) |
| 9 | Thalassoma_pavo | Ornate wrasse | Med (iNaturalist) |
| 10 | fish_01 | Dascyllus reticulatus | F4K (Taiwan reef) |
| 11 | fish_02 | Plectroglyphidodon dickii | F4K (Taiwan reef) |
| 12 | fish_03 | Chromis chrysura | F4K (Taiwan reef) |
| 13 | fish_04 | Amphiprion clarkii | F4K (Taiwan reef) |
| 14 | fish_05 | Chaetodon lunulatus | F4K (Taiwan reef) |
| 15 | fish_06 | Chaetodon trifascialis | F4K (Taiwan reef) |
| 16 | fish_07 | Myripristis kuntee | F4K (Taiwan reef) |
| 17 | fish_08 | Acanthurus nigrofuscus | F4K (Taiwan reef) |
| 18 | fish_09 | Hemigymnus fasciatus | F4K (Taiwan reef) |
| 19 | fish_10 | Neoniphon sammara | F4K (Taiwan reef) |
| 20 | fish_11 | Abudefduf vaigiensis | F4K (Taiwan reef) |
| 21 | fish_12 | Canthigaster valentini | F4K (Taiwan reef) |
| 22 | fish_13 | Pomacentrus moluccensis | F4K (Taiwan reef) |
| 23 | fish_14 | Zebrasoma scopas | F4K (Taiwan reef) |
| 24 | fish_15 | Hemigymnus melapterus | F4K (Taiwan reef) |
| 25 | fish_16 | Lutjanus fulvus | F4K (Taiwan reef) |
| 26 | fish_17 | Scolopsis bilineata | F4K (Taiwan reef) |
| 27 | fish_18 | Scaridae | F4K (Taiwan reef) |
| 28 | fish_19 | Pempheris vanicolensis | F4K (Taiwan reef) |
| 29 | fish_20 | Zanclus cornutus | F4K (Taiwan reef) |
| 30 | fish_21 | Neoglyphidodon nigroris | F4K (Taiwan reef) |
| 31 | fish_22 | Balistapus undulatus | F4K (Taiwan reef) |
| 32 | fish_23 | Siganus fuscescens | F4K (Taiwan reef) |

## Quick lists

### (a) 10 Mediterranean species (Med / iNaturalist)

1. Rainbow wrasse (Coris_julis)
2. White seabream (Diplodus_sargus)
3. Dusky grouper (Epinephelus_marginatus)
4. Striped red mullet (Mullus_surmuletus)
5. Mediterranean moray (Muraena_helena)
6. Saddled seabream (Oblada_melanura)
7. Salema porgy (Sarpa_salpa)
8. Red scorpionfish (Scorpaena_scrofa)
9. Gilthead seabream (Sparus_aurata)
10. Ornate wrasse (Thalassoma_pavo)

### (b) 23 F4K species (Taiwan reef / Fish4Knowledge)

1. Dascyllus reticulatus (fish_01)
2. Plectroglyphidodon dickii (fish_02)
3. Chromis chrysura (fish_03)
4. Amphiprion clarkii (fish_04)
5. Chaetodon lunulatus (fish_05)
6. Chaetodon trifascialis (fish_06)
7. Myripristis kuntee (fish_07)
8. Acanthurus nigrofuscus (fish_08)
9. Hemigymnus fasciatus (fish_09)
10. Neoniphon sammara (fish_10)
11. Abudefduf vaigiensis (fish_11)
12. Canthigaster valentini (fish_12)
13. Pomacentrus moluccensis (fish_13)
14. Zebrasoma scopas (fish_14)
15. Hemigymnus melapterus (fish_15)
16. Lutjanus fulvus (fish_16)
17. Scolopsis bilineata (fish_17)
18. Scaridae (fish_18)
19. Pempheris vanicolensis (fish_19)
20. Zanclus cornutus (fish_20)
21. Neoglyphidodon nigroris (fish_21)
22. Balistapus undulatus (fish_22)
23. Siganus fuscescens (fish_23)
