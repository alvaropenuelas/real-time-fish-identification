# Visual audit of the 10 BioCLIP crops + category re-cut

Follow-up to the BioCLIP-2 zero-shot probe. No new BioCLIP runs — this regroups the
already-computed `outputs/bioclip_zeroshot_results.csv` by a by-eye category of each crop.
Ground truth for the whole clip: *Amphiprion clarkii*.

## STEP 1 — what each crop actually contains

| crop | track / size | category | what's in the image |
|---|---|---|---|
| easy_00 | 30 / 558×502 | BACKGROUND_DOMINANT | lavender haze + white anemone tentacles bottom-left; no fish |
| easy_01 | 1 / 473×274 | BACKGROUND_DOMINANT | out-of-focus pale mass, no fins/markings; ambiguous blur, no clear fish |
| easy_02 | 85 / 574×368 | MIXED | orange/pink anemone fills upper half; pale blurred fish body lower-left |
| easy_03 | 109 / 322×151 | BACKGROUND_DOMINANT | flat deep-blue water, nothing |
| easy_04 | 39 / 586×489 | CLEAN_FISH | anemonefish (yellow body, white/purple bands) dominant, dark background |
| hard_00 | 107 / 456×239 | CLEAN_FISH | close-up anemonefish flank (yellow + white band) fills the box |
| hard_01 | 29 / 903×300 | BACKGROUND_DOMINANT | white anemone tentacles dominate; tiny clownfish sliver top-left |
| hard_02 | 61 / 357×175 | BACKGROUND_DOMINANT | dark-blue haze + particles, no fish |
| hard_03 | 75 / 896×887 | MIXED | anemone center; clownfish bottom-left + green fish sliver |
| hard_04 | 83 / 1761×419 | BACKGROUND_DOMINANT | blue haze, rock/coral, particles, no clear fish |

Tally: **2 CLEAN_FISH, 2 MIXED, 6 BACKGROUND_DOMINANT.** The 6 background-dominant crops
are detector ghost boxes — consistent with the earlier domain-gap finding on this clip.

## STEP 2 — with-clarkii results re-cut by category

20 with-clarkii runs (10 crops × 2 schemes). Note: the saved table only stores top1/top2,
so clarkii's exact rank/score is known only when it lands in the top 2; otherwise rank ≥3
and score < the listed top2 (exact value not recoverable without a new forward pass).

| category | runs | clarkii top-1 | clarkii in top-2 | known clarkii rank/score |
|---|---:|---:|---:|---|
| CLEAN_FISH | 4 | **0/4** | 1/4 | hard_00 B: rank 2, 0.246; others rank ≥3 |
| MIXED | 4 | **1/4** | 2/4 | hard_03 B: rank 1, 0.978; hard_03 A: rank 2, 0.328 |
| BACKGROUND_DOMINANT | 12 | **0/12** | 1/12 | hard_01 B: rank 2, 0.098; others rank ≥3 |

What top-1 actually was on the CLEAN_FISH crops (where clarkii *should* win):

- easy_04: scheme A → *Amphiprion percula* (0.66); scheme B → Orange clownfish (0.53)
- hard_00: scheme A → *Amphiprion percula* (0.75); scheme B → Sea anemone (0.31), clarkii 2nd (0.25)

On clean, fish-dominant crops BioCLIP is confident but picks the **wrong *Amphiprion*
species** (percula / ocellaris / orange clownfish) — never clarkii. The single clarkii
top-1 in the whole experiment (hard_03 B, 0.978) is on a **MIXED** crop, not a clean one.

## STEP 3 — crop extraction: is padding added?

No. `scripts/test_bioclip_zeroshot.py` (extract_crops) cuts:

```python
x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
x1, y1 = max(0, x1), max(0, y1)
crop = frame[y1:y2, x1:x2]
```

**Zero margin/padding** — the raw YOLO box, only clamped to be non-negative. So the
anemone/background inside the crops comes from the **detector's boxes themselves** (loose /
ghost boxes), not from crop expansion. Padding is not the explanation for the background
bias.

## STEP 4 — revised verdict

The "BioCLIP-2 zero-shot does not beat the closed-world EfficientNet" conclusion **holds
even on CLEAN_FISH crops**: clarkii top-1 was **0/4** there, with BioCLIP confidently
naming the wrong anemonefish species. So the failure is **not merely a
background-contamination artifact** — it's fine-grained *Amphiprion* species confusion.

Two honest caveats the README must keep:

1. **Small clean sample.** Only 2 of 10 crops were CLEAN_FISH (4 runs). The clean-crop
   conclusion is suggestive, not statistically firm.
2. **Most crops were background.** 6/10 were BACKGROUND_DOMINANT detector false positives,
   where BioCLIP's "Sea anemone" calls are often *correct* — its open vocabulary can say
   "not a fish," which the closed-world EfficientNet cannot. The original aggregate was
   dragged down partly by these, so it slightly understated BioCLIP's honesty.

Correct README claim: *On this clip, BioCLIP-2 zero-shot failed to identify Clark's
anemonefish even on clean fish-dominant crops (0/4), confusing it with other Amphiprion
species; the clean-crop sample is small (2 crops), and 6/10 selected crops were
background-dominant detector false positives.* Not a state-of-the-art claim either way.

(To get exact clarkii rank/score on all 20 runs, the script would need to dump the full
softmax per crop — that requires re-running BioCLIP, which this follow-up deliberately did
not do.)
