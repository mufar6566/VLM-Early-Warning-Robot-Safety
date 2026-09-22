# Data and Ground-Truth Construction

This document records the data provenance and ground-truth procedures used in the
VLM robot-navigation safety experiments. It is intended both as supplementary
material for the paper and as a long-term reproducibility record.

## Case-study summary

| Case study | Bags | Frames | Safe | Potentially unsafe | Unsafe |
|---|---:|---:|---:|---:|---:|
| HuRoN | 15 | 16,862 | 15,273 | 1,378 | 211 |
| PAL | 3 | 9,249 | 5,518 | 2,769 | 962 |
| CrowdBot | 7 | 8,801 | 4,205 | 4,350 | 246 |
| **Combined** | **25** | **34,912** | **24,996** | **8,497** | **1,419** |

All three case studies are mapped to the common three-class vocabulary:

`safe`, `potentially_unsafe`, and `unsafe`.

The class label corresponds to the newest/current frame in each causal history.

## Raw data

Original ROS bags, extracted raw image sequences, model caches, and local
execution environments are not stored in this Git repository.

The repository instead preserves the final cleaned predictions, ground-truth
generation code where available, frame-level metadata, validation summaries,
analysis code, and final statistical results.

Before local raw-data copies are deleted, the original ROS bags or source
datasets should be retained in durable project/institutional storage. A
persistent dataset location or DOI should be added here when available.

## HuRoN

HuRoN uses the fisheye visual stream together with synchronized robot and
environment measurements. The ground-truth generator is:

`case_studies/huron/preprocessing/make_frame_gt.py`

The frame-level procedure uses bumper/contact status, proximity indicators,
pedestrian distance, odometry, and LiDAR evidence.

Sensor observations are matched to image frames using a nearest-time tolerance
of 0.15 s.

### HuRoN state labels

Unsafe:

`contact_active == True` OR closest pedestrian distance `< 0.60 m`.

Potentially unsafe:

proximity indicator active OR pedestrian distance in `[0.60, 1.20) m`.

Safe:

no unsafe/potentially-unsafe condition and closest pedestrian distance
`>= 1.20 m`.

LiDAR is retained as supporting evidence but is not used to determine the
HuRoN state label in the archived configuration (`USE_LASER_FOR_GT = False`).

### HuRoN early-warning annotation

For an unsafe event beginning at time `t_unsafe`:

`lead_time_to_unsafe = t_unsafe - current_frame_time`

The required stopping time is:

`T_required = T_reaction + speed / a_brake + T_margin`

Archived parameters:

| Parameter | Value |
|---|---:|
| Reaction time | 0.20 s |
| Safety margin | 0.50 s |
| Maximum warning window | 3.00 s |
| Moderate braking | 1.00 m/s^2 |
| Conservative braking | 0.50 m/s^2 |

Both braking profiles are computed. The conservative profile is retained as the
main/default warning target in the HuRoN GT generator.

## PAL

The canonical PAL ground-truth generator is:

`case_studies/pal/preprocessing/make_frame_gt.py`

SHA-256 of the preserved script:

`13ebe77f5b4e6fd35ab7c8be4d8d81bd1f68f839501b19c268b02b0ac809a8b8`

The same hash was obtained from both final PAL working copies before archival.
The older `make_pal_frame_gt_before_fix.py` had a different hash and is not the
canonical script.

PAL ground truth is based on the minimum valid front-scan distance associated
with each RGB frame.

### PAL state labels

Unsafe:

front obstacle distance `< 0.35 m`.

Potentially unsafe:

front obstacle distance in `[0.35, 0.80) m`.

Safe:

front obstacle distance `>= 0.80 m`.

A frame without a valid front-scan distance is marked unknown by the generator.

### PAL early-warning annotation

The archived PAL generator uses:

| Parameter | Value |
|---|---:|
| Reaction time | 0.20 s |
| Safety margin | 0.50 s |
| Maximum warning window | 3.00 s |
| Moderate braking | 1.00 m/s^2 |
| Conservative braking | 0.50 m/s^2 |

The final PAL evaluation contains 9,249 canonical frames across three bags.

## CrowdBot

The CrowdBot ground-truth generator is:

`case_studies/crowdbot/preprocessing/make_frame_gt.py`

Ground truth is derived from the front LiDAR stream and synchronized
odometry. Sensor matching is causal: for each image frame the latest available
sensor observation at or before that frame is used. Future sensor observations
are not used.

The archived default LiDAR and odometry matching tolerances are 0.15 s.

A short causal rolling median is applied to the front-LiDAR distance to suppress
isolated single-beam spikes. Raw distance values are also retained for audit.

### CrowdBot state labels

Unsafe:

front clearance `< 0.60 m`.

Potentially unsafe:

front clearance in `[0.60, 1.20) m`.

Safe:

front clearance `>= 1.20 m`.

Unknown:

no fresh valid front-LiDAR observation is available.

Stopping distance, closing speed, and TTC quantities are retained as diagnostic
variables. The archived code explicitly does not use these quantities to modify
`gt_state`.

Future unsafe-event annotations are constructed separately and therefore do not
change the current-frame classification target.

## Independence from VLM predictions

Ground-truth labels are generated from dataset-specific physical/sensor evidence
rather than from VLM outputs.

History windows are causal. The target is always the newest/current frame.

Future unsafe-event information is used only for subsequent early-warning
evaluation and is not provided to the VLM as classification input.

## Validation files

Dataset-specific integrity and label summaries are stored under:

`case_studies/huron/validation/`

`case_studies/pal/validation/`

`case_studies/crowdbot/validation/`

The final combined analysis contains 34,912 frames from 25 bags for each common
model/approach/history configuration.
