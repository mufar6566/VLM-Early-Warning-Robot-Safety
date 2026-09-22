#!/usr/bin/env python3

import argparse
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoProcessor


LABELS = ["safe", "potentially_unsafe", "unsafe"]
WARNING_CLASSES = {"potentially_unsafe", "unsafe"}


GT_CSV = "02_ground_truth/bww1_Feb-16-2023-bww1-intloss_00000000_frame_level_gt_fisheye_dynamic_warning.csv"
BAG_DIR = "01_extracted_data/bww1_Feb-16-2023-bww1-intloss_00000000"
OUT_ROOT = Path("03_predictions/matrix_huron_bww1")


def get_device_from_model(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hf_model(model_id):
    """
    Load supported VLMs.
    Qwen2.5-VL needs its specific HF class.
    InternVL will be handled separately if generic AutoModel works.
    """
    import transformers

    # Qwen2.5-VL specific loader
    if "Qwen2.5-VL" in model_id or "Qwen2_5" in model_id:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            print("Trying loader: Qwen2_5_VLForConditionalGeneration", flush=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            model.eval()
            print("Loaded with Qwen2_5_VLForConditionalGeneration", flush=True)
            return model
        except Exception as e:
            print("Qwen2_5_VLForConditionalGeneration failed:", e, flush=True)

    model = None
    last_err = None

    class_names = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModel",
        "AutoModelForCausalLM",
    ]

    for cls_name in class_names:
        try:
            cls = getattr(transformers, cls_name)
        except Exception as e:
            last_err = e
            continue

        try:
            print(f"Trying loader: {cls_name}", flush=True)
            model = cls.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            model.eval()
            print(f"Loaded with {cls_name}", flush=True)
            return model
        except Exception as e:
            print(f"Loader failed {cls_name}: {e}", flush=True)
            last_err = e

    raise RuntimeError(f"Could not load model {model_id}. Last error: {last_err}")


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

Answer with exactly one label only:
safe
potentially_unsafe
unsafe
""".strip()

    return base


def make_inputs(processor, image, prompt_text, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = prompt_text

    try:
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
        )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    return inputs, text


@torch.no_grad()
def avg_logprob_candidate(model, tokenizer, base_inputs, candidate):
    device = base_inputs["input_ids"].device

    cand_ids = tokenizer(
        candidate,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    if cand_ids.numel() == 0:
        return -1e9

    base_ids = base_inputs["input_ids"]
    base_mask = base_inputs.get("attention_mask", torch.ones_like(base_ids))

    full_ids = torch.cat([base_ids, cand_ids], dim=1)
    full_mask = torch.cat([base_mask, torch.ones_like(cand_ids)], dim=1)

    model_inputs = {}
    for k, v in base_inputs.items():
        if k in ["input_ids", "attention_mask"]:
            continue
        model_inputs[k] = v

    model_inputs["input_ids"] = full_ids
    model_inputs["attention_mask"] = full_mask

    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits

    base_len = base_ids.shape[1]
    vals = []

    for j in range(cand_ids.shape[1]):
        token_id = cand_ids[0, j]
        pos = base_len + j - 1
        logp = torch.log_softmax(logits[0, pos, :], dim=-1)[token_id]
        vals.append(float(logp.detach().cpu()))

    return float(np.mean(vals))



def is_qwen_model(model):
    try:
        mt = getattr(model.config, "model_type", "")
        return "qwen2_5_vl" in str(mt).lower() or "qwen" in model.__class__.__name__.lower()
    except Exception:
        return False


@torch.no_grad()
def score_labels_qwen_first_token(model, processor, inputs):
    """
    Qwen2.5-VL can fail when we append candidate tokens manually because its
    visual-token position/mask logic is strict. Instead, score the first
    possible answer token for each label from the next-token distribution.
    This gives usable class probabilities without invalid score collapse.
    """
    tokenizer = processor.tokenizer

    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    label_variants = {
        "safe": ["safe", " safe", "\nsafe"],
        "potentially_unsafe": [
            "potentially_unsafe",
            " potentially_unsafe",
            "\npotentially_unsafe",
            "potentially unsafe",
            " potentially unsafe",
            "\npotentially unsafe",
        ],
        "unsafe": ["unsafe", " unsafe", "\nunsafe"],
    }

    scores = {}

    for label, variants in label_variants.items():
        vals = []
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False).input_ids
            if len(ids) > 0:
                vals.append(float(log_probs[ids[0]].detach().cpu()))
        scores[label] = max(vals) if vals else -1e9

    arr = np.array([scores[x] for x in LABELS], dtype=np.float64)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs = probs / probs.sum()

    pred = LABELS[int(np.argmax(probs))]
    return pred, scores, probs


def score_labels(model, processor, inputs):
    # Qwen-specific robust scoring
    if is_qwen_model(model):
        try:
            return score_labels_qwen_first_token(model, processor, inputs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            raise RuntimeError(
                f"Qwen first-token scoring failed: {e}"
            ) from e

    # Original sequence scoring for non-Qwen models
    tokenizer = processor.tokenizer
    scores = {}

    for label in LABELS:
        try:
            scores[label] = avg_logprob_candidate(model, tokenizer, inputs, label)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            scores[label] = -1e9
        except Exception as e:
            print("Candidate scoring error:", label, str(e), flush=True)
            scores[label] = -1e9

    arr = np.array([scores[x] for x in LABELS], dtype=np.float64)

    if np.all(arr <= -1e8):
        raise RuntimeError("All candidate label scores are invalid.")

    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs = probs / probs.sum()

    pred = LABELS[int(np.argmax(probs))]
    return pred, scores, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_csv", required=True)
    ap.add_argument("--row_index", type=int, required=True)
    ap.add_argument("--skip_existing", action="store_true")
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
                if "prediction_valid" in old.columns:
                    bad = ~old["prediction_valid"].fillna(False).astype(bool)
                else:
                    bad = (
                        old["score_safe"].isna()
                        | old["score_potentially_unsafe"].isna()
                        | old["score_unsafe"].isna()
                        | (
                            (old["score_safe"] <= -1e8)
                            & (old["score_potentially_unsafe"] <= -1e8)
                            & (old["score_unsafe"] <= -1e8)
                        )
                    )
                invalid_count = int(bad.sum())
                print(f"Existing file found: {out_csv}")
                print(f"Rows={len(old)}, invalid={invalid_count}")

                if invalid_count == 0:
                    print("Existing file is complete and valid; skipping.")
                    return

                print("Existing file contains invalid predictions; rerunning.")
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

    if end_frame < 0:
        end_frame = int(df["frame_id"].max())

    df = df[
        (df["frame_id"] >= start_frame)
        & (df["frame_id"] <= end_frame)
    ].copy()
    df = df.sort_values("frame_id").reset_index(drop=True)

    print(
        f"Frames to process: {len(df)} "
        f"(start_frame={start_frame}, end_frame={end_frame})",
        flush=True,
    )

    if df.empty:
        raise RuntimeError(
            f"No frames selected after filtering "
            f"(start_frame={start_frame}, end_frame={end_frame})"
        )

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

    print("Loading processor:", model_id, flush=True)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading model:", model_id, flush=True)
    model = load_hf_model(model_id)
    device = get_device_from_model(model)

    rows = []

    for i, cur in df.iterrows():
        start_i = max(0, i - history_frames + 1)
        hist = df.iloc[start_i:i+1].copy()

        image_paths = hist["resolved_image_path"].tolist()
        sheet = make_contact_sheet(image_paths, tile_size=224, cols=4)

        sensor_context = build_sensor_context(hist)
        prompt_text = build_prompt(input_mode, sensor_context)

        print(f"[{i+1}/{len(df)}] frame={int(cur['frame_id'])} hist={len(hist)}", flush=True)

        prediction_valid = True
        inference_error_type = ""
        inference_error_message = ""
        probability_source = "qwen_first_token_logits"

        try:
            inputs, chat_text = make_inputs(processor, sheet, prompt_text, device)
            pred, scores, probs = score_labels(model, processor, inputs)

            probs = np.asarray(probs, dtype=np.float64)

            if (
                pred not in LABELS
                or probs.shape != (len(LABELS),)
                or not np.all(np.isfinite(probs))
                or np.any(probs < 0.0)
                or not np.isclose(probs.sum(), 1.0, atol=1e-6)
            ):
                raise RuntimeError(
                    f"Invalid prediction output: pred={pred}, probs={probs}"
                )

        except torch.cuda.OutOfMemoryError as e:
            print("OOM at frame", int(cur["frame_id"]), flush=True)
            torch.cuda.empty_cache()

            prediction_valid = False
            inference_error_type = "cuda_out_of_memory"
            inference_error_message = str(e)
            probability_source = ""

            pred = np.nan
            scores = {x: np.nan for x in LABELS}
            probs = np.full(len(LABELS), np.nan, dtype=np.float64)

        except Exception as e:
            print("ERROR at frame", int(cur["frame_id"]), str(e), flush=True)

            prediction_valid = False
            inference_error_type = type(e).__name__
            inference_error_message = str(e)
            probability_source = ""

            pred = np.nan
            scores = {x: np.nan for x in LABELS}
            probs = np.full(len(LABELS), np.nan, dtype=np.float64)

        out = cur.to_dict()
        out.update({
            "model_name": model_name,
            "model_id": model_id,
            "run_id": run_id,
            "input_mode": input_mode,
            "history_frames": history_frames,
            "predicted_state": pred,
            "prediction_valid": prediction_valid,
            "inference_error_type": inference_error_type,
            "inference_error_message": inference_error_message,
            "probability_source": probability_source,
            "score_safe": scores["safe"],
            "score_potentially_unsafe": scores["potentially_unsafe"],
            "score_unsafe": scores["unsafe"],
            "p_safe": probs[0],
            "p_potentially_unsafe": probs[1],
            "p_unsafe": probs[2],
            "num_history_frames_used": len(hist),
            "sensor_context_text": sensor_context if input_mode == "sensor" else "",
            "prompt_text": prompt_text,
            "llava_raw_output": "",
        })

        rows.append(out)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        try:
            del inputs
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Saved:", out_csv, flush=True)


if __name__ == "__main__":
    main()
