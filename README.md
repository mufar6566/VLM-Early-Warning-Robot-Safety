# VLM-Based Early Warning for Robot Navigation Safety

Research artifact for an empirical study of vision-language models (VLMs) for robot-navigation safety and early warning.

## Study

The artifact evaluates three VLMs:

- Qwen2.5-VL-7B-Instruct
- LLaVA-OneVision-7B
- InternVL3

across three case studies:

- HuRoN
- CrowdBot
- PAL

Two approaches are evaluated:

- AppFr: visual input
- AppOd: visual input with odometry and sensor context

Sixteen history lengths are used: H02, H04, ..., H32.

This gives 3 models × 2 approaches × 16 histories = 96 configurations per case study.

## Repository

- `case_studies/` — preprocessing/configuration/inference code, final predictions, metadata, and validation files
- `analysis/` — combined-data builder and RQ1–RQ4 analysis notebooks
- `results/` — generated CSV and LaTeX result tables
- `docs/VALIDATION.md` — artifact validation information
- `docs/DATA_AND_GROUND_TRUTH.md` — data provenance and exact ground-truth construction
- `docs/REPRODUCIBILITY.md` — experiment reconstruction and analysis instructions

## Reproduce the analysis

Install the analysis dependencies:

    pip install -r requirements.txt

Build the combined prediction dataset:

    python analysis/build_combined_predictions.py

Then execute the notebooks in order:

1. `analysis/rq1_performance.ipynb`
2. `analysis/rq2_uncertainty.ipynb`
3. `analysis/rq3_relationship.ipynb`
4. `analysis/rq4_history.ipynb`

RQ4 is executed after RQ3 because its final cross-RQ analysis uses RQ3 outputs.

## Prediction data

The repository contains 288 final cleaned prediction files:

- HuRoN: 96
- CrowdBot: 96
- PAL: 96

The generated combined dataset is intentionally excluded from Git because it can be rebuilt from these files.

## Raw prediction outputs

The original model prediction CSVs are available in the GitHub release:

**[Raw prediction outputs](https://github.com/mufar6566/VLM-Early-Warning-Robot-Safety/releases/tag/raw-predictions)**

- HuRoN: 1,440 raw prediction CSVs
- PAL: 288 raw prediction CSVs
- CrowdBot: 672 raw prediction CSVs
- Total: 2,400 raw prediction CSVs

The release also includes a file manifest and SHA-256 checksums for the archives.

## Validation

The portable artifact reproduces the original analysis results:

- Scientific result tables compared: 84
- Matched: 84
- Different: 0

See `docs/VALIDATION.md` for details.

## Raw data, ground truth, and model weights

Original ROS bags, raw image sequences, model weights, and model caches are not redistributed in this repository.

The repository preserves final cleaned predictions, ground-truth generation code, frame-level metadata, validation summaries, analysis code, and reported results.

See `docs/DATA_AND_GROUND_TRUTH.md` for the exact ground-truth procedures and `docs/REPRODUCIBILITY.md` for reconstruction instructions.
