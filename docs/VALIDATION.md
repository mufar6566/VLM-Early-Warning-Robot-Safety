# Artifact Validation

The portable artifact was validated against the original analysis workspace after repository restructuring.

## End-to-End Execution

The following analyses execute successfully from the artifact:

- RQ1 — Performance
- RQ2 — Uncertainty
- RQ3 — History-length effects / performance relationships
- RQ4 — Early-warning analysis

The analysis pipeline uses:

1. Final cleaned case-study predictions
2. `analysis/build_combined_predictions.py`
3. RQ1–RQ4 notebooks
4. Compact RQ4 frame metadata instead of the original multi-gigabyte raw prediction trees

## Result Equivalence

All scientific CSV outputs shared between the original analysis and the portable artifact were compared.

| Item | Count |
|---|---:|
| Compared outputs | 84 |
| Exact scientific matches | 84 |
| Different outputs | 0 |
| Artifact-only scientific outputs | 0 |

Machine-specific path columns and output-manifest bookkeeping were excluded from the comparison.

Numerical values were compared with:

- relative tolerance: `1e-9`
- absolute tolerance: `1e-12`
- NaN equality enabled

Textual scientific fields were compared exactly.

## RQ4 Metadata Reduction

The original RQ4 workflow accessed per-configuration raw prediction files only to recover frame timing, ground-truth state, and robot speed.

The portable artifact instead stores one canonical frame-metadata file per case study:

- HuRoN: 16,862 frames, 15 bags
- PAL: 9,249 frames, 3 bags
- CrowdBot: 8,801 frames, 7 bags

Total:

- 34,912 frames
- 25 bags
- zero missing robot-speed values

The compact metadata was validated frame-by-frame against the clean Qwen AppFr H02 canonical population before use.

## Combined Dataset

For every model × approach × history configuration:

- HuRoN: 16,862 frames
- PAL: 9,249 frames
- CrowdBot: 8,801 frames
- Combined: 34,912 frames
- Combined bags: 25

There are 96 combined experimental configurations:

- 3 VLMs
- 2 approaches
- 16 history lengths

## Validation Status

**PASS — 84/84 scientific result tables reproduced.**
