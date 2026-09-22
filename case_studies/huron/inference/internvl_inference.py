#!/usr/bin/env python3

import argparse
import gc
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoModel, AutoTokenizer

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


LABELS = ["safe", "potentially_unsafe", "unsafe"]

GT_CSV = "02_ground_truth/bww1_Feb-16-2023-bww1-intloss_00000000_frame_level_gt_fisheye_dynamic_warning.csv"
BAG_DIR = "01_extracted_data/bww1_Feb-16-2023-bww1-intloss_00000000"
OUT_ROOT = Path("03_predictions/matrix_huron_bww1")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=8, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def internvl_load_image_from_pil(image, input_size=448, max_num=8):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def resolve_image_path(bag_dir, value):
    s = str(value)
    p = Path(s)

    if p.is_absolute() and p.exists():
        return str(p)

    bag_dir = Path(bag_dir)

    candidates = [
        bag_dir / s,
        bag_dir / "images" / s,
        bag_dir / "images" / "fisheye" / s,
        bag_dir / "images" / "rgb" / s,
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    name = Path(s).name
    hits = list(bag_dir.rglob(name))
    if hits:
        return str(hits[0])

    return str(bag_dir / s)


def make_contact_sheet(image_paths, tile_size=224, cols=4):
    imgs = []

    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((tile_size, tile_size))
        imgs.append(img)

    n = len(imgs)
    rows = int(math.ceil(n / cols))

    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), "white")

    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        sheet.paste(img, (c * tile_size, r * tile_size))

    return sheet


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt(x, unit=""):
    x = safe_float(x)
    if pd.isna(x) or not math.isfinite(x):
        return "unknown"
    return f"{x:.3f}{unit}"


def build_sensor_context(hist_df):
    speed = pd.to_numeric(hist_df.get("odom_speed_mps", np.nan), errors="coerce")
    ang = pd.to_numeric(hist_df.get("odom_angular_speed", np.nan), errors="coerce")
    laser = pd.to_numeric(hist_df.get("min_laser_distance_m", np.nan), errors="coerce")

    cur_speed = speed.iloc[-1] if len(speed) else np.nan
    mean_speed = speed.mean()
    max_speed = speed.max()

    cur_ang = ang.iloc[-1] if len(ang) else np.nan
    mean_abs_ang = ang.abs().mean()

    cur_laser = laser.iloc[-1] if len(laser) else np.nan
    min_laser = laser.min()
    first_laser = laser.iloc[0] if len(laser) else np.nan

    laser_trend = cur_laser - first_laser if pd.notna(cur_laser) and pd.notna(first_laser) else np.nan

    T_REACTION_S = 0.20
    T_MARGIN_S = 0.50
    A_BRAKE_CONSERVATIVE = 0.50

    if pd.notna(cur_speed):
        t_required_conservative = T_REACTION_S + cur_speed / A_BRAKE_CONSERVATIVE + T_MARGIN_S
    else:
        t_required_conservative = np.nan

    return f"""
Robot motion and LiDAR context for the same recent visual history:
- Number of recent frames: {len(hist_df)}
- Current odometry speed: {fmt(cur_speed, " m/s")}
- Mean odometry speed over recent history: {fmt(mean_speed, " m/s")}
- Maximum odometry speed over recent history: {fmt(max_speed, " m/s")}
- Current angular speed: {fmt(cur_ang, " rad/s")}
- Mean absolute angular speed over recent history: {fmt(mean_abs_ang, " rad/s")}
- Current minimum laser/LiDAR distance: {fmt(cur_laser, " m")}
- Minimum laser/LiDAR distance over recent history: {fmt(min_laser, " m")}
- Laser/LiDAR distance trend over recent history: {fmt(laser_trend, " m")} where negative means the robot is getting closer to an obstacle
- Estimated conservative stopping time from current speed: {fmt(t_required_conservative, " s")}

Use these values only as supporting robot-state context. Do not use any future information.
""".strip()


def build_prompt(input_mode, sensor_context):
    base = """
You are a robot safety monitor for outdoor crowd navigation by the Qolo personal mobility robot.

The image is a contact sheet of recent robot-camera frames.
Read the contact sheet left-to-right and top-to-bottom.
The last tile is the current frame.

Classify the current robot situation into exactly one label:

safe:
The robot has enough free space. No slowing, stopping, or avoidance is needed.

potentially_unsafe:
A person, obstacle, wall, doorway, or object is close enough that the robot may need to slow down, stop, or avoid soon.
Choose this label if the robot is approaching a hazard or if caution is needed before a possible unsafe situation.

unsafe:
The robot is in contact, collision, extremely close to an obstacle/person, or immediate collision is unavoidable.
""".strip()

    if input_mode == "sensor":
        base += "\n\n" + sensor_context

    base += """

Return exactly one JSON object and nothing else:
{"label":"safe|potentially_unsafe|unsafe","p_safe":0.0,"p_potentially_unsafe":0.0,"p_unsafe":0.0}

The three probabilities must sum to 1.0.
""".strip()

    return base


def normalize_probs(p_safe, p_punsafe, p_unsafe, pred):
    vals = np.array([p_safe, p_punsafe, p_unsafe], dtype=np.float64)

    if not np.all(np.isfinite(vals)) or vals.sum() <= 0:
        vals = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        vals[LABELS.index(pred)] = 0.8

    vals = np.clip(vals, 1e-6, 1.0)
    vals = vals / vals.sum()

    return vals


def parse_internvl_response(text):
    raw = str(text).strip()
    lower = raw.lower()

    pred = None

    # Try JSON first
    try:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            obj = json.loads(m.group(0))
            label = str(obj.get("label", "")).strip().lower().replace(" ", "_")
            if label in LABELS:
                pred = label
                probs = normalize_probs(
                    float(obj.get("p_safe", np.nan)),
                    float(obj.get("p_potentially_unsafe", np.nan)),
                    float(obj.get("p_unsafe", np.nan)),
                    pred,
                )
                return pred, probs
    except Exception:
        pass

    # Fallback label extraction
    if "potentially_unsafe" in lower or "potentially unsafe" in lower:
        pred = "potentially_unsafe"
    elif re.search(r"\bunsafe\b", lower):
        pred = "unsafe"
    elif re.search(r"\bsafe\b", lower):
        pred = "safe"
    else:
        pred = "safe"

    # Fallback probability extraction
    def grab(name):
        patterns = [
            rf"{name}\s*[:=]\s*([0-9]*\.?[0-9]+)",
            rf'"{name}"\s*:\s*([0-9]*\.?[0-9]+)',
        ]
        for pat in patterns:
            mm = re.search(pat, lower)
            if mm:
                try:
                    return float(mm.group(1))
                except Exception:
                    pass
        return np.nan

    probs = normalize_probs(
        grab("p_safe"),
        grab("p_potentially_unsafe"),
        grab("p_unsafe"),
        pred,
    )
    return pred, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_csv", required=True)
    ap.add_argument("--row_index", type=int, required=True)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_tiles", type=int, default=8)
    args = ap.parse_args()

    cfg = pd.read_csv(args.config_csv)
    row = cfg.iloc[args.row_index].to_dict()

    run_id = row["run_id"]
    model_name = row["model_name"]
    model_id = row["model_id"]
    input_mode = row["input_mode"]
    history_frames = int(row["history_frames"])
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])

    cfg_out_csv = row.get("out_csv", None)
    if cfg_out_csv is None or pd.isna(cfg_out_csv) or str(cfg_out_csv).strip() == "":
        out_csv = OUT_ROOT / f"{run_id}.csv"
    else:
        out_csv = Path(str(cfg_out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            expected = end_frame - start_frame + 1
            if len(old) == expected:
                bad = (
                    (old["score_safe"] <= -1e8) &
                    (old["score_potentially_unsafe"] <= -1e8) &
                    (old["score_unsafe"] <= -1e8)
                )
                print(f"Existing complete file found: {out_csv}")
                print(f"Rows={len(old)}, invalid={int(bad.sum())}")
                return
        except Exception:
            pass

    print("=" * 80, flush=True)
    print("RUN:", run_id, flush=True)
    print("MODEL:", model_name, model_id, flush=True)
    print("INPUT:", input_mode, flush=True)
    print("HISTORY:", history_frames, flush=True)

    cfg_gt_csv = row.get("gt_csv", None)
    if cfg_gt_csv is None or pd.isna(cfg_gt_csv) or str(cfg_gt_csv).strip() == "":
        gt_csv = GT_CSV
    else:
        gt_csv = str(cfg_gt_csv)

    print("GT_CSV:", gt_csv, flush=True)
    df = pd.read_csv(gt_csv)
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype(int)
    df["frame_time_s"] = pd.to_numeric(df["frame_time_s"], errors="coerce")
    df = df[(df["frame_id"] >= start_frame) & (df["frame_id"] <= end_frame)].copy()
    df = df.sort_values("frame_id").reset_index(drop=True)

    cfg_bag_dir = row.get("bag_dir", None)
    if cfg_bag_dir is None or pd.isna(cfg_bag_dir) or str(cfg_bag_dir).strip() == "":
        bag_dir = Path(BAG_DIR).resolve()
    else:
        bag_dir = Path(str(cfg_bag_dir)).resolve()

    print("BAG_DIR:", bag_dir, flush=True)
    print("OUT_CSV:", out_csv, flush=True)
    df["resolved_image_path"] = df["image_path"].apply(lambda x: resolve_image_path(bag_dir, x))

    missing = [p for p in df["resolved_image_path"] if not Path(p).exists()]
    if missing:
        print("Missing image examples:", flush=True)
        for m in missing[:10]:
            print(m, flush=True)
        raise RuntimeError(f"Missing image paths: {len(missing)}")

    print("Loading InternVL tokenizer:", model_id, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    print("Loading InternVL model:", model_id, flush=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        use_flash_attn=False,
    ).eval().cuda()

    generation_config = dict(
        max_new_tokens=120,
        do_sample=False,
        temperature=0.0,
    )

    rows = []

    for i, cur in df.iterrows():
        start_i = max(0, i - history_frames + 1)
        hist = df.iloc[start_i:i+1].copy()

        image_paths = hist["resolved_image_path"].tolist()
        sheet = make_contact_sheet(image_paths, tile_size=224, cols=4)

        sensor_context = build_sensor_context(hist)
        prompt_text = build_prompt(input_mode, sensor_context)

        print(f"[{i+1}/{len(df)}] frame={int(cur['frame_id'])} hist={len(hist)}", flush=True)

        try:
            pixel_values = internvl_load_image_from_pil(
                sheet,
                input_size=448,
                max_num=args.max_tiles,
            ).to(torch.bfloat16).cuda()

            question = prompt_text
            if not question.lstrip().startswith("<image>"):
                question = "<image>\n" + question

            raw = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
            )

            pred, probs = parse_internvl_response(raw)

        except torch.cuda.OutOfMemoryError:
            print("OOM at frame", int(cur["frame_id"]), flush=True)
            torch.cuda.empty_cache()
            raw = "OOM"
            pred = "safe"
            probs = np.array([1/3, 1/3, 1/3], dtype=np.float64)

        except Exception as e:
            print("ERROR at frame", int(cur["frame_id"]), str(e), flush=True)
            raw = "ERROR: " + str(e)
            pred = "safe"
            probs = np.array([1/3, 1/3, 1/3], dtype=np.float64)

        scores = np.log(np.clip(probs, 1e-9, 1.0))

        out = cur.to_dict()
        out.update({
            "model_name": model_name,
            "model_id": model_id,
            "run_id": run_id,
            "input_mode": input_mode,
            "history_frames": history_frames,
            "predicted_state": pred,
            "score_safe": scores[0],
            "score_potentially_unsafe": scores[1],
            "score_unsafe": scores[2],
            "p_safe": probs[0],
            "p_potentially_unsafe": probs[1],
            "p_unsafe": probs[2],
            "num_history_frames_used": len(hist),
            "sensor_context_text": sensor_context if input_mode == "sensor" else "",
            "prompt_text": prompt_text,
            "llava_raw_output": raw,
            "internvl_raw_output": raw,
        })

        rows.append(out)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        try:
            del pixel_values
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    print("Saved:", out_csv, flush=True)


if __name__ == "__main__":
    main()
