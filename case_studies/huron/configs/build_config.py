from pathlib import Path
import os
import pandas as pd

ROOT = Path(os.environ.get("HURON_ROOT", "."))
EXP = ROOT / "04_experiments/all15_history_h2_h32"

GT_DIR = ROOT / "02_ground_truth/full15_bags"
BAG_DIR = ROOT / "01_extracted_data"

history_values = list(range(2, 34, 2))
input_modes = ["frame", "sensor"]

models = {
    "qwen": {
        "model_name": "Qwen",
        "model_id": os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
    },
    "internvl": {
        "model_name": "InternVL",
        "model_id": os.environ.get("INTERNVL_MODEL_ID", "InternVL3-8B"),
    },
    "llava": {
        "model_name": "LLaVA",
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    },
}

gt_files = sorted(GT_DIR.glob("*_frame_level_gt_fisheye_dynamic_warning.csv"))

if len(gt_files) != 15:
    raise RuntimeError(f"Expected 15 GT files, found {len(gt_files)}")

rows = []

for gt_csv in gt_files:
    bag_name = gt_csv.name.replace("_frame_level_gt_fisheye_dynamic_warning.csv", "")
    bag_dir = BAG_DIR / bag_name

    df = pd.read_csv(gt_csv)
    start_frame = int(df["frame_id"].min())
    end_frame = int(df["frame_id"].max())
    n_frames = len(df)

    if start_frame != 0 or end_frame != n_frames - 1:
        raise RuntimeError(f"Frame range mismatch for {bag_name}: rows={n_frames}, start={start_frame}, end={end_frame}")

    if not bag_dir.exists():
        raise RuntimeError(f"Missing extracted bag dir: {bag_dir}")

    for model_key, model_info in models.items():
        for input_mode in input_modes:
            for h in history_values:
                history_label = f"h{h}"
                run_id = f"all15_{model_key}_{bag_name}_{history_label}_{input_mode}"

                rows.append({
                    "run_id": run_id,
                    "bag_name": bag_name,
                    "bag_id": bag_name,
                    "model": model_key,
                    "model_name": model_info["model_name"],
                    "model_id": model_info["model_id"],
                    "input_mode": input_mode,
                    "input_type": input_mode,
                    "history": h,
                    "history_frames": h,
                    "history_label": history_label,
                    "max_video_frames": h,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "n_frames": n_frames,
                    "bag_dir": str(bag_dir),
                    "gt_csv": str(gt_csv),
                    "out_csv": str(EXP / "predictions" / model_key / bag_name / f"{run_id}.csv"),
                    "eval_dir": str(EXP / "metrics" / model_key / bag_name / run_id),
                })

cfg = pd.DataFrame(rows)

config_dir = EXP / "configs"
config_dir.mkdir(parents=True, exist_ok=True)

cfg.to_csv(config_dir / "all15_all_config.csv", index=False)

for model_key in models:
    sub = cfg[cfg["model"] == model_key].copy()
    sub.to_csv(config_dir / f"all15_{model_key}_config.csv", index=False)

print("Saved configs in:", config_dir)
print("All rows:", len(cfg))
print("\nBy model/input:")
print(cfg.groupby(["model", "input_mode"]).size())

print("\nBy bag:")
print(cfg.groupby("bag_name").size())

print("\nTotal frames across 15 bags:", cfg.drop_duplicates("bag_name")["n_frames"].sum())
print("Total frame-level predictions:", int((cfg["n_frames"]).sum()))

print("\nPreview:")
print(cfg[["run_id", "model", "input_mode", "history_frames", "start_frame", "end_frame", "n_frames", "out_csv"]].head(10).to_string(index=False))
