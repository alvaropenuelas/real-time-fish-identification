# Clip ghost-box diagnosis — fish_clip_2

Box counts from `weights/detector_deepfish.pt` on the 4K clip at each conf.
VISUAL ONLY — threshold must be chosen on the held-out TEST set, not here.

| conf | total boxes | mean/frame | max/frame | frames with boxes |
|---:|---:|---:|---:|---:|
| 0.25 | 1363 | 3.28 | 9 | 416 |
| 0.5 | 773 | 1.86 | 6 | 382 |
| 0.65 | 440 | 1.06 | 3 | 307 |
| 0.8 | 109 | 0.26 | 1 | 109 |
