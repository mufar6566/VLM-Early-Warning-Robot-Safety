# Reproducibility Guide

This repository is the archival research artifact for the VLM robot-navigation
safety study.

The goal is to preserve enough information to reproduce the reported analysis
and to reconstruct the inference experiment when the required source datasets
and model weights are available.

## Experimental matrix

Three VLMs are evaluated:

| Model | Artifact identifier |
|---|---|
| Qwen | Qwen2.5-VL-7B-Instruct |
| LLaVA | llava-hf/llava-onevision-qwen2-7b-ov-hf |
| InternVL | InternVL3-8B |

Two input approaches are used:

`AppFr`: visual/history input.

`AppOd`: visual/history input plus causal robot/sensor context.

History lengths are:

`H02, H04, H06, H08, H10, H12, H14, H16, H18, H20, H22, H24, H26, H28, H30, H32`.

This gives 96 configurations per case study and 288 final cleaned prediction
files across HuRoN, PAL, and CrowdBot.

## Model provenance

The archived configuration records Qwen2.5-VL-7B-Instruct from the Hugging Face
Qwen model cache. The HuRoN configuration records the snapshot hash:

`cc594898137f460bfe9f0759e9844b3ce807cfb5`

LLaVA uses:

`llava-hf/llava-onevision-qwen2-7b-ov-hf`

InternVL uses a locally stored `InternVL3-8B` model directory.

The exact upstream InternVL revision is not encoded in the current configuration
files. If the original local InternVL model directory still exists, preserve it
or record its upstream revision for long-term reproducibility.

Model weights themselves are not redistributed in this repository.

## Ground truth
## Ground truth

Ground-truth construction is documented in:

`docs/DATA_AND_GROUND_TRUTH.md`

Canonical generators are:

`case_studies/huron/preprocessing/make_frame_gt.py`

`case_studies/pal/preprocessing/make_frame_gt.py`

`case_studies/crowdbot/preprocessing/make_frame_gt.py`

## Inference code

The model-specific inference implementations are preserved under each case
study's `inference/` directory.

These files should be treated as archival experimental code. Historical file
paths and model-cache locations may reflect the original execution machines and
may need to be replaced with equivalent local paths when recreating an
experiment.

When reproducing the reported experiment, preserve the archived prompts,
classification definitions, history construction, probability extraction, and
causal sensor handling.

## Final prediction data

The repository includes 288 cleaned prediction CSV files:

| Case study | Prediction files |
|---|---:|
| HuRoN | 96 |
| PAL | 96 |
| CrowdBot | 96 |
| **Total** | **288** |

Expected rows in each case-study configuration are:

| Case study | Rows |
|---|---:|
| HuRoN | 16,862 |
| PAL | 9,249 |
| CrowdBot | 8,801 |

## Build the combined analysis dataset

From the repository root:

    python analysis/build_combined_predictions.py

Expected result:

96 combined configuration files.

Each combined configuration should contain:

34,912 rows and 25 bags.

The generated combined files are deliberately excluded from Git because they
are reproducible from the included case-study prediction files.

## Statistical analysis

Execute the notebooks in this order:

    analysis/rq1_performance.ipynb
    analysis/rq2_uncertainty.ipynb
    analysis/rq3_relationship.ipynb
    analysis/rq4_history.ipynb

RQ4 follows RQ3 because its final cross-RQ analysis uses RQ3 outputs.

## Result regression validation

The portable artifact was compared with the original analysis repository.

Scientific CSV result tables compared: 84.

Exact matches: 84.

Different: 0.

Additional details are recorded in:

`docs/VALIDATION.md`

## Long-term preservation

Do not rely on a single laptop as the only copy of raw ROS bags, locally stored
model weights, or non-public data.

Non-downloadable raw recordings and locally unique model directories should be
retained in durable institutional or project storage for long-term reproducibility.

A tagged GitHub release should be created after the archival documentation and
provenance records are finalized.
