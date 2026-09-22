#!/usr/bin/env python3
from pathlib import Path
import os
import csv
import pandas as pd

ROOT = Path(os.environ.get("CROWDBOT_ROOT", "."))
GT_CSV = ROOT / "03_evaluation_inputs/crowdbot_all_bags_frame_gt_3class_ex3.csv"
EXTRACTED_ROOT = ROOT / "01_extracted_data/extracted"

MODELS = {
    "qwen": os.environ.get(
        "QWEN_MODEL_ID",
        "Qwen/Qwen2.5-VL-7B-Instruct"
    ),
    "internvl": os.environ.get(
        "INTERNVL_MODEL_ID",
        "InternVL3-8B"
    ),
    "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}

HISTORIES = list(range(2, 33, 2))
INPUT_MODES = ["appfr", "appod"]
FIELDS = [
    "run_id", "model_name", "model_id", "bag_id", "input_mode",
    "history_frames", "expected_rows", "max_samples", "gt_csv",
    "bag_dir", "out_csv",
]


def write_config(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not GT_CSV.exists():
        raise FileNotFoundError(GT_CSV)
    gt = pd.read_csv(GT_CSV)
    required = {"bag_id", "frame_uid", "frame_time_s", "image_path", "gt_state"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Evaluation CSV missing columns: {sorted(missing)}")
    if not gt["gt_state"].isin(["safe", "potentially_unsafe", "unsafe"]).all():
        raise ValueError("Evaluation CSV contains a non-three-class target")
    if gt["frame_uid"].duplicated().any():
        raise ValueError("Evaluation CSV contains duplicate frame_uid values")

    bag_counts = gt.groupby("bag_id").size().sort_index().to_dict()
    if len(bag_counts) != 7 or sum(bag_counts.values()) != 8865:
        raise ValueError(f"Unexpected CrowdBot population: {bag_counts}")

    config_dir = ROOT / "configs"
    output_dir = ROOT / "04_results"
    first_bag = next(iter(bag_counts))

    for model_name, model_id in MODELS.items():
        rows, smoke_rows = [], []
        for bag_id, expected_rows in bag_counts.items():
            bag_dir = EXTRACTED_ROOT / bag_id
            if not bag_dir.exists():
                raise FileNotFoundError(bag_dir)
            for input_mode in INPUT_MODES:
                for history in HISTORIES:
                    run_id = f"{model_name}_{bag_id}_{input_mode}_h{history:02d}"
                    out_csv = (
                        output_dir / model_name / input_mode / f"h{history:02d}"
                        / f"{run_id}.csv"
                    )
                    row = {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model_id": model_id,
                        "bag_id": bag_id,
                        "input_mode": input_mode,
                        "history_frames": history,
                        "expected_rows": expected_rows,
                        "max_samples": -1,
                        "gt_csv": str(GT_CSV),
                        "bag_dir": str(bag_dir),
                        "out_csv": str(out_csv),
                    }
                    rows.append(row)
                    if bag_id == first_bag and history == 2:
                        smoke = dict(row)
                        smoke["run_id"] += "_smoke8"
                        smoke["expected_rows"] = 8
                        smoke["max_samples"] = 8
                        smoke["out_csv"] = str(
                            output_dir / "smoke" / model_name
                            / f"{smoke['run_id']}.csv"
                        )
                        smoke_rows.append(smoke)

        write_config(config_dir / f"{model_name}_full.csv", rows)
        write_config(config_dir / f"{model_name}_smoke.csv", smoke_rows)
        print(f"{model_name}: full={len(rows)}, smoke={len(smoke_rows)}")


if __name__ == "__main__":
    main()
