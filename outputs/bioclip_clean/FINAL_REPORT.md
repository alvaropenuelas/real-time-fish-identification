# BioCLIP-2 zero-shot on clarkii — final consolidated report

Closes the BioCLIP investigation on fish_clip_2 (ground truth: *Amphiprion clarkii*).
Full score distributions recovered; clean-crop sample expanded to n=6. No further
experiments proposed.

## STEP 1 — full distributions on the original 10 crops (with-clarkii, schemes A+B)

`full_scores_existing.csv` holds every label+score per run. clarkii's exact rank/score,
previously unknown for 16 of 20 runs, is now recovered:

| crop | scheme A: top1 (clarkii rank/score) | scheme B: top1 (clarkii rank/score) |
|---|---|---|
| easy_00 | Sea anemone (r4 / 0.005) | Sea anemone (r3 / 0.001) |
| easy_01 | A. percula (r4 / 0.070) | Sea anemone (r5 / 0.007) |
| easy_02 | A. ocellaris (r4 / 0.020) | Sea anemone (r5 / 0.005) |
| easy_03 | Sea anemone (r4 / 0.041) | Sea anemone (r6 / 0.002) |
| easy_04 | A. percula (r4 / 0.003) | Orange clownfish (r3 / 0.005) |
| hard_00 | A. percula (r4 / 0.004) | Sea anemone (**r2 / 0.246**) |
| hard_01 | A. percula (r4 / 0.012) | Sea anemone (r2 / 0.098) |
| hard_02 | Chromis chrysura (r8 / 0.034) | Pacific chromis (r7 / 0.009) |
| hard_03 | A. percula (**r2 / 0.328**) | **Clark's anemonefish (r1 / 0.978)** |
| hard_04 | Pomacentrus moluccensis (r4 / 0.044) | Lemon damselfish (r7 / 0.010) |

clarkii is top-1 in exactly **1 / 20** runs (hard_03 B). It reaches rank 2 in only 3 others.
Under scheme A it sits at rank 4 on almost every actual-fish crop, behind *percula* /
*ocellaris*.

## STEP 2 — additional clean crops (visually confirmed, independent tracks)

Track duration / detector confidence was NOT trusted (that produced the contaminated set).
Candidates were cut from many tracks and **inspected by eye**; only fish-dominant,
anemone-minor crops were kept. Four new clean crops from **distinct** tracks (34, 95, 71,
118) were confirmed. Near-duplicate frames of easy_04 (track 39, frames 178/186) were found
clean but **excluded** to avoid counting the same fish twice.

`full_scores_new_clean.csv`:

| crop | scheme A: top1 (clarkii r/score) | scheme B: top1 (clarkii r/score) |
|---|---|---|
| cand_t34_f0136  | A. percula (r4 / 0.009) | Sea anemone (r7 / 0.014) |
| cand_t95_f0307  | A. percula (r7 / 0.003) | Pacific chromis (r7 / 0.003) |
| cand_t71_f0244  | A. percula (r4 / 0.004) | Pacific chromis (r6 / 0.017) |
| cand_t118_f0396 | Pomacentrus moluccensis (r6 / 0.005) | Reticulated dascyllus (r7 / 0.001) |

clarkii top-1: **0 / 8**.

## Consolidated CLEAN_FISH crops (n = 6 independent)

| crop | track | A top1 | A clarkii | B top1 | B clarkii |
|---|---|---|---|---|---|
| easy_04 | 39 | A. percula | r4 / 0.003 | Orange clownfish | r3 / 0.005 |
| hard_00 | 107 | A. percula | r4 / 0.004 | Sea anemone | r2 / 0.246 |
| cand_t34_f0136 | 34 | A. percula | r4 / 0.009 | Sea anemone | r7 / 0.014 |
| cand_t95_f0307 | 95 | A. percula | r7 / 0.003 | Pacific chromis | r7 / 0.003 |
| cand_t71_f0244 | 71 | A. percula | r4 / 0.004 | Pacific chromis | r6 / 0.017 |
| cand_t118_f0396 | 118 | Pomacentrus moluccensis | r6 / 0.005 | Reticulated dascyllus | r7 / 0.001 |

**clarkii top-1 on clean crops: 0 / 12 runs (6 crops × 2 schemes).** Best clarkii ever
reaches on a clean crop is rank 2 (hard_00 B, 0.246); under scheme A it never beats rank 4
and its score never exceeds ~0.01.

## Verdict

On **n = 6 independent, visually-confirmed clean fish crops** (12 scheme runs), BioCLIP-2
zero-shot identified *Amphiprion clarkii* as top-1 **zero times**. It is confidently wrong
at the species level — scheme A collapses onto *Amphiprion percula* on five of six clean
crops; scheme B scatters across Sea anemone / Pacific chromis / Reticulated dascyllus. The
failure is therefore **not** an artifact of background-contaminated ghost-box crops: it
persists on clean fish. BioCLIP-2 zero-shot does **not** beat the closed-world EfficientNet
on this clip.

The single clarkii top-1 in the entire investigation (hard_03 scheme B, 0.978) was a MIXED
crop with the clownfish sitting in its **host anemone**. Whether the surrounding anemone
context cues BioCLIP toward the correct anemonefish label is a speculative one-off
hypothesis — **flagged and deliberately not pursued further.**
