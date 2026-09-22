from pathlib import Path
import os
import csv

ROOT = Path(os.environ.get("PAL_ROOT", "."))

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

BAGS = [
    "rosbag2_2025_07_11-13_23_07",
    "rosbag2_2025_07_11-13_37_37",
    "rosbag2_2025_07_11-13_55_25",
]

HISTORIES = list(range(2, 33, 2))
INPUT_MODES = ["visual", "sensor"]

fieldnames = [
    "run_id",
    "model_name",
    "model_id",
    "input_mode",
    "history_frames",
    "start_frame",
    "end_frame",
    "gt_csv",
    "bag_dir",
    "out_csv",
]

config_dir = ROOT / "configs"
output_dir = ROOT / "outputs"
config_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

for model_name, model_id in MODELS.items():
    rows = []

    for bag_name in BAGS:
        gt_csv = (
            ROOT
            / "ground_truth"
            / f"{bag_name}_frame_level_gt_front_scan_dynamic_warning.csv"
        )
        bag_dir = ROOT / "bags" / bag_name

        if not gt_csv.exists():
            raise FileNotFoundError(f"Missing GT file: {gt_csv}")

        if not bag_dir.exists():
            raise FileNotFoundError(f"Missing bag directory: {bag_dir}")

        for input_mode in INPUT_MODES:
            for history in HISTORIES:
                run_id = (
                    f"{model_name}_{bag_name}_"
                    f"{input_mode}_h{history:02d}"
                )

                out_csv = (
                    output_dir
                    / model_name
                    / input_mode
                    / f"h{history:02d}"
                    / f"{run_id}.csv"
                )

                out_csv.parent.mkdir(parents=True, exist_ok=True)

                rows.append({
                    "run_id": run_id,
                    "model_name": model_name,
                    "model_id": model_id,
                    "input_mode": input_mode,
                    "history_frames": history,
                    "start_frame": 0,
                    "end_frame": -1,
                    "gt_csv": str(gt_csv),
                    "bag_dir": str(bag_dir),
                    "out_csv": str(out_csv),
                })

    config_file = config_dir / f"{model_name}_full.csv"

    with config_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{model_name}: {len(rows)} runs -> {config_file}")
