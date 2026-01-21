# Optional Ablations Results

## Mapper Capacity

We recommend varying mapper depth $n\in\{0,2,4,8\}$ (with $n{=}0$ being linear-only) and hidden width $h\in\{512,1024,2048\}$, holding all other settings fixed, to verify that performance is not narrowly tied to a single capacity choice.

| Config ($n$, $h$) | R@1 | R@5 | R@20 | MRR | MCES@1 (mean) |
|-------------------|-----|-----|------|-----|---------------|
| $(0, 512)$ | --- | --- | --- | --- | --- |
| $(0, 1024)$ | --- | --- | --- | --- | --- |
| $(0, 2048)$ | --- | --- | --- | --- | --- |
| $(2, 512)$ | 0.297 | 0.392 | 0.521 | 0.349 | 7.95 |
| $(2, 1024)$ | --- | --- | --- | --- | --- |
| $(2, 2048)$ | 0.312 | 0.411 | 0.539 | 0.365 | 7.64 |
| $(4, 512)$ | --- | --- | --- | --- | --- |
| $(4, 1024)$ | --- | --- | --- | --- | --- |
| $(4, 2048)$ | 0.312 | 0.411 | 0.541 | 0.365 | 7.64 |
| $(8, 512)$ | --- | --- | --- | --- | --- |
| $(8, 1024)$ | --- | --- | --- | --- | --- |
| $(8, 2048)$ | --- | --- | --- | --- | --- |

## Spectrum Unfreezing Depth

We recommend evaluating unfreezing schedules (frozen; last-1; last-2; last-4 blocks) to quantify the effect of limited spectrum-side adaptation on retrieval.

| Unfreeze Depth | R@1 | R@5 | R@20 | MRR | MCES@1 (mean) |
|----------------|-----|-----|------|-----|---------------|
| Frozen (0) | --- | --- | --- | --- | --- |
| Last-1 (1) | --- | --- | --- | --- | --- |
| Last-2 (2) | --- | --- | --- | --- | --- |
| Last-4 (4) | --- | --- | --- | --- | --- |
