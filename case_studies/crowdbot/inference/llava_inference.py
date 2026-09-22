from pathlib import Path
import argparse
import json
import re
import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


CLASSES = ["safe", "potentially_unsafe", "unsafe"]


# ============================================================
# Prompt
# ============================================================

def _fmt_sensor(v, unit=""):
    try:
        if pd.isna(v):
            return "unknown"
        return f"{float(v):.3f}{unit}"
    except Exception:
        return "unknown"


def build_sensor_context(hist_df):
    lines = [
        "Causal sensor context paired one-to-one with the video frames:",
        "Frames are listed oldest to newest; the final line is current.",
    ]
    for j, (_, r) in enumerate(hist_df.iterrows(), start=1):
        role = "current/newest" if j == len(hist_df) else "history"
        lines.append(
            f"- F{j:02d} ({role}): "
            f"speed={_fmt_sensor(r.get('robot_speed_mps'), ' m/s')}, "
            f"angular_speed={_fmt_sensor(r.get('angular_speed_radps'), ' rad/s')}, "
            f"front_lidar_min={_fmt_sensor(r.get('front_min_distance_raw_m'), ' m')}"
        )
    lines.append("No future sensor observation is included.")
    return "\n".join(lines)


def build_prompt(input_mode="appfr", sensor_context=""):
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

Return exactly one label only:
safe
potentially_unsafe
unsafe
""".strip()

    if input_mode == "appod":
        base += "\n\n" + sensor_context

    return base


# ============================================================
# Paths and video creation
# ============================================================

def resolve_image_path(image_path_str, bag_dir, image_view="fisheye"):
    """
    GT image_path may be an old relative path like:
    results/huron_extracted_bags/.../images/fisheye/fisheye_000001_t_0.098.jpg

    This function maps it to the actual extracted bag folder.
    """

    p = Path(str(image_path_str))

    if p.is_absolute() and p.exists():
        return p

    # Try relative to current working directory.
    if p.exists():
        return p.resolve()

    # Try relative to bag root.
    p2 = bag_dir / p
    if p2.exists():
        return p2.resolve()

    # Try by basename inside bag_dir/images/fisheye.
    p3 = bag_dir / "images" / image_view / p.name
    if p3.exists():
        return p3.resolve()

    # Try recursive search by basename under images.
    matches = list((bag_dir / "images").rglob(p.name))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(f"Could not resolve image path: {image_path_str}")


def make_history_video(image_paths, out_video_path, fps=10.0, resize_width=None):
    """
    Create a temporary MP4 video from cumulative image history.
    """

    if len(image_paths) == 0:
        raise ValueError("No image paths provided.")

    first = Image.open(image_paths[0]).convert("RGB")
    w, h = first.size

    if resize_width is not None and resize_width > 0 and w > resize_width:
        scale = resize_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (new_w, new_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_video_path}")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        if (new_w, new_h) != img.size:
            img = img.resize((new_w, new_h))
        frame_rgb = np.array(img)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()

    return out_video_path


# ============================================================
# LLaVA input
# ============================================================

def load_history_as_video_array(image_paths, resize_width=None):
    """
    Load history frames directly from JPG/PNG files as an in-memory video array.

    This avoids torchcodec/torchvision video decoding errors on eX3.
    Output shape: [num_frames, height, width, 3], dtype uint8.
    """

    frames = []

    target_size = None

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")

        if resize_width is not None and resize_width > 0:
            w, h = img.size
            if w > resize_width:
                scale = resize_width / float(w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h))

        # All frames must have same size.
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size)

        frames.append(np.array(img, dtype=np.uint8))

    if len(frames) == 0:
        raise ValueError("No history frames loaded.")

    return np.stack(frames, axis=0)


def make_inputs(processor, model, image_paths, prompt, resize_width=None):
    """
    Build LLaVA-OneVision inputs from in-memory video frames.

    Important:
    We do NOT pass an mp4 path here. Passing an mp4 path makes Transformers
    call torchcodec, which fails on eX3 because libtorchcodec/libnvrtc is not
    available/compatible.
    """

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    video_array = load_history_as_video_array(
        image_paths=image_paths,
        resize_width=resize_width,
    )

    # Try to preserve the exact number of history frames.
    try:
        inputs = processor(
            text=[chat_text],
            videos=[video_array],
            return_tensors="pt",
            padding=True,
            videos_kwargs={
                "num_frames": video_array.shape[0],
            },
        )
    except TypeError:
        # Fallback for processor versions that do not accept videos_kwargs.
        inputs = processor(
            text=[chat_text],
            videos=[video_array],
            return_tensors="pt",
            padding=True,
        )

    inputs = inputs.to(model.device, dtype=torch.float16)

    return inputs


# ============================================================
# Candidate label scoring
# ============================================================

def candidate_token_ids(tokenizer, label):
    """
    Try both label and newline-ended label.
    We use the better likelihood later.
    """
    variants = [
        label,
        label + "\n",
    ]

    ids_list = []
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if len(ids) > 0:
            ids_list.append(ids)

    # fallback
    if not ids_list:
        ids_list = [tokenizer(label, add_special_tokens=False).input_ids]

    return ids_list


@torch.inference_mode()
def sequence_avg_logprob(model, inputs, cand_ids):
    """
    Score candidate answer tokens conditioned on the visual prompt.

    We compute average log probability over candidate tokens:
      log p(candidate | prompt, video) / number_of_tokens

    Length normalization is important because 'potentially_unsafe' can be
    more than one token.
    """

    prompt_ids = inputs["input_ids"]
    prompt_mask = inputs["attention_mask"]

    cand = torch.tensor([cand_ids], dtype=torch.long, device=prompt_ids.device)
    cand_mask = torch.ones_like(cand, dtype=prompt_mask.dtype, device=prompt_mask.device)

    full_ids = torch.cat([prompt_ids, cand], dim=1)
    full_mask = torch.cat([prompt_mask, cand_mask], dim=1)

    model_inputs = {}
    for k, v in inputs.items():
        if k in ["input_ids", "attention_mask"]:
            continue
        model_inputs[k] = v

    outputs = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
        **model_inputs,
    )

    logits = outputs.logits
    prompt_len = prompt_ids.shape[1]

    total = 0.0

    for j, tok in enumerate(cand_ids):
        # token at full_ids[prompt_len + j] is predicted by logits[prompt_len + j - 1]
        pos = prompt_len + j - 1
        log_probs = torch.log_softmax(logits[0, pos, :], dim=-1)
        total += float(log_probs[tok].detach().cpu())

    return total / max(1, len(cand_ids))


@torch.inference_mode()
def score_labels(model, processor, inputs):
    tokenizer = processor.tokenizer

    label_scores = {}

    for label in CLASSES:
        variant_scores = []
        for ids in candidate_token_ids(tokenizer, label):
            try:
                variant_scores.append(sequence_avg_logprob(model, inputs, ids))
            except Exception:
                traceback.print_exc()

        if len(variant_scores) == 0:
            raise RuntimeError(
                f"No valid candidate score computed for label {label!r}"
            )

        label_scores[label] = max(variant_scores)

    scores = np.array([label_scores[c] for c in CLASSES], dtype=np.float64)

    # softmax
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / probs.sum()

    return {
        "p_safe": float(probs[0]),
        "p_potentially_unsafe": float(probs[1]),
        "p_unsafe": float(probs[2]),
        "score_safe": float(label_scores["safe"]),
        "score_potentially_unsafe": float(label_scores["potentially_unsafe"]),
        "score_unsafe": float(label_scores["unsafe"]),
        "predicted_state": CLASSES[int(np.argmax(probs))],
    }


# ============================================================
# Optional raw generation
# ============================================================

def parse_label_from_text(text):
    s = str(text).strip().lower()

    # order matters because "unsafe" contains "safe"
    if "potentially_unsafe" in s or "potentially unsafe" in s or "warning" in s:
        return "potentially_unsafe"
    if re.search(r"\bunsafe\b", s):
        return "unsafe"
    if re.search(r"\bsafe\b", s):
        return "safe"

    return ""


@torch.inference_mode()
def generate_raw_output(model, processor, inputs, max_new_tokens=16):
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    gen_ids = output_ids[:, input_len:]
    text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_csv",
        default=None,
        help="Optional experiment configuration CSV.",
    )

    parser.add_argument(
        "--row_index",
        type=int,
        default=None,
        help="Zero-based row index in --config_csv.",
    )
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument(
        "--gt_csv",
        default="crowdbot_all_bags_frame_gt_3class_ex3.csv",
        help="CrowdBot three-class frame-level evaluation CSV.",
    )

    parser.add_argument(
        "--bag_dir",
        default=".",
        help="CrowdBot extracted-data root.",
    )

    parser.add_argument(
        "--out_csv",
        default="outputs/llava/manual.csv",
        help="Output prediction CSV.",
    )

    parser.add_argument(
        "--model_id",
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="HF model ID or local model path.",
    )

    parser.add_argument(
        "--image_view",
        default="fisheye",
        help="Image view used in GT and image folder.",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame_id.",
    )

    parser.add_argument(
        "--max_frame_id",
        type=int,
        default=120,
        help="Stop at this frame_id inclusive. Use -1 for all frames.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run every N frames.",
    )

    parser.add_argument(
        "--video_fps",
        type=float,
        default=10.0,
        help="FPS used when creating temporary history video.",
    )

    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=0,
        help=(
            "0 means use all history frames up to current frame. "
            "If GPU memory fails, set e.g. 64 to uniformly sample history."
        ),
    )

    parser.add_argument(
        "--resize_width",
        type=int,
        default=0,
        help="Resize video frames to this width. 0 keeps original size.",
    )

    parser.add_argument(
        "--include_speed",
        action="store_true",
        help="Include current odometry speed in the prompt.",
    )

    parser.add_argument(
        "--generate_raw",
        action="store_true",
        help="Also generate raw text label. Slower.",
    )

    parser.add_argument(
        "--temp_video_dir",
        default="../../03_predictions/temp_llava_history_videos",
        help="Temporary cumulative history videos.",
    )

    args = parser.parse_args()

    config_row = None

    if args.config_csv is not None:
        if args.row_index is None:
            raise ValueError("--row_index is required when --config_csv is used.")

        config_path = Path(args.config_csv).expanduser().resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config CSV not found: {config_path}")

        config_df = pd.read_csv(config_path)

        if args.row_index < 0 or args.row_index >= len(config_df):
            raise IndexError(
                f"row_index={args.row_index} is outside config range "
                f"0..{len(config_df) - 1}"
            )

        config_row = config_df.iloc[args.row_index]

        def config_value(name, default=None):
            value = config_row.get(name, default)
            if pd.isna(value):
                return default
            if isinstance(value, str) and value.strip() == "":
                return default
            return value

        args.gt_csv = str(config_value("gt_csv", args.gt_csv))
        args.bag_dir = str(config_value("bag_dir", args.bag_dir))
        args.out_csv = str(config_value("out_csv", args.out_csv))
        args.model_id = str(config_value("model_id", args.model_id))

        args.start_frame = int(
            config_value("start_frame", args.start_frame)
        )

        args.max_frame_id = int(
            config_value("end_frame", args.max_frame_id)
        )

        # Experiment history H02, H04, ..., H32 maps directly
        # to the latest N chronological frames.
        args.max_video_frames = int(
            config_value("history_frames", args.max_video_frames)
        )

        input_mode = str(config_value("input_mode", "appfr")).strip().lower()
        if input_mode not in {"appfr", "appod"}:
            raise ValueError(
                f"Unsupported input_mode={input_mode!r}; "
                f"expected 'appfr' or 'appod'."
            )
        bag_id = str(config_value("bag_id"))
        expected_rows = int(config_value("expected_rows"))
        max_samples = int(config_value("max_samples", -1))

        print("Config CSV:", config_path, flush=True)
        print("Config row index:", args.row_index, flush=True)
        print(
            "Config run ID:",
            config_value("run_id", f"row_{args.row_index}"),
            flush=True,
        )
        print("Config input mode:", input_mode, flush=True)
        print(
            "Config history frames:",
            args.max_video_frames,
            flush=True,
        )

    bag_dir = Path(args.bag_dir).expanduser().resolve()
    gt_csv = Path(args.gt_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    temp_video_dir = Path(args.temp_video_dir).expanduser().resolve()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            valid = old.get("prediction_valid", pd.Series(False, index=old.index))
            if len(old) == expected_rows and valid.fillna(False).astype(bool).all():
                print(f"Existing complete output; skipping: {out_csv}")
                return
        except Exception:
            pass

    print("\n============================================================")
    print("LLaVA-OneVision cumulative-history safety inference")
    print("============================================================")
    print("Bag dir:", bag_dir)
    print("GT CSV:", gt_csv)
    print("Output CSV:", out_csv)
    print("Model:", args.model_id)
    print("Bag:", bag_id)
    print("Max video frames:", args.max_video_frames)
    print("Input mode:", input_mode)
    print("Generate raw:", args.generate_raw)

    df = pd.read_csv(gt_csv)
    required = {"bag_id", "frame_uid", "frame_index", "frame_time_s", "image_path", "gt_state"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"GT is missing required columns: {sorted(missing_cols)}")
    df = df[df["bag_id"].astype(str).eq(bag_id)].copy()
    df["frame_id"] = pd.to_numeric(df["frame_index"], errors="coerce").astype(int)
    df = df.sort_values(["frame_time_s", "frame_id"]).reset_index(drop=True)
    if max_samples > 0:
        df = df.head(max_samples).copy()
    run_df = df

    print("Frames to run:", len(run_df))

    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
    processor.tokenizer.padding_side = "left"

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Resolve all image paths once.
    all_image_paths = []
    for _, row in df.iterrows():
        all_image_paths.append(
            resolve_image_path(row["image_path"], bag_dir, image_view=args.image_view)
        )

    results = []

    for position, (_, row) in enumerate(run_df.iterrows()):
        n = position + 1
        frame_id = int(row["frame_id"])
        frame_time_s = float(row["frame_time_s"])

        # Cumulative history = all frames from start up to current frame.
        history_df = df.iloc[: position + 1]
        history_paths = all_image_paths[: position + 1]

        if args.max_video_frames > 0 and len(history_paths) > args.max_video_frames:
            # Use the latest N frames only: current frame + most recent previous frames.
            history_paths_used = history_paths[-args.max_video_frames:]
            history_df_used = history_df.iloc[-args.max_video_frames:]
            sampling_note = f"last_{args.max_video_frames}_from_{len(history_paths)}"
        else:
            history_paths_used = history_paths
            history_df_used = history_df
            sampling_note = "all_history_frames"

        num_frames_for_processor = len(history_paths_used)

        # No temporary mp4 decoding is used after the patch.
        # We keep this field only for bookkeeping.
        video_path = ""

        sensor_context = build_sensor_context(history_df_used)
        prompt = build_prompt(input_mode=input_mode, sensor_context=sensor_context)

        try:
            inputs = make_inputs(
                processor=processor,
                model=model,
                image_paths=history_paths_used,
                prompt=prompt,
                resize_width=args.resize_width if args.resize_width > 0 else None,
            )

            scores = score_labels(model, processor, inputs)

            raw_output = ""
            raw_label = ""
            if args.generate_raw:
                raw_output = generate_raw_output(model, processor, inputs)
                raw_label = parse_label_from_text(raw_output)

            result = row.to_dict()
            result.update({
                "run_id": str(config_value("run_id", f"row_{args.row_index}")),
                "model_name": "llava",
                "model_id": args.model_id,
                "prediction_valid": True,
                "inference_error_type": "",
                "inference_error_message": "",
                "probability_source": "llava_sequence_logprob",
                "num_history_frames_total": len(history_paths),
                "num_history_frames_used": len(history_paths_used),
                "history_frame_uids": "|".join(history_df_used["frame_uid"].astype(str)),
                "input_mode": input_mode,
                "sensor_context_text": sensor_context if input_mode == "appod" else "",
                "history_sampling": sampling_note,
                "history_video_path": str(video_path),
                "prompt_text": prompt,
                "llava_raw_output": raw_output,
                "llava_raw_label": raw_label,
            })
            result.update(scores)

            results.append(result)

            print(
                f"[{n}/{len(run_df)}] frame={frame_id:04d} "
                f"t={frame_time_s:.3f}s "
                f"gt={row['gt_state']} "
                f"pred={scores['predicted_state']} "
                f"p=[{scores['p_safe']:.3f}, {scores['p_potentially_unsafe']:.3f}, {scores['p_unsafe']:.3f}] "
                f"history={len(history_paths_used)}/{len(history_paths)}"
            )

        except Exception as e:
            traceback.print_exc()

            result = row.to_dict()
            result.update({
                "run_id": str(config_value("run_id", f"row_{args.row_index}")),
                "model_name": "llava",
                "model_id": args.model_id,
                "prediction_valid": False,
                "inference_error_type": type(e).__name__,
                "inference_error_message": str(e),
                "probability_source": "",
                "num_history_frames_total": len(history_paths),
                "num_history_frames_used": len(history_paths_used),
                "history_frame_uids": "|".join(history_df_used["frame_uid"].astype(str)),
                "input_mode": input_mode,
                "sensor_context_text": sensor_context if input_mode == "appod" else "",
                "history_sampling": sampling_note,
                "history_video_path": str(video_path),
                "prompt_text": prompt,
                "llava_raw_output": "",
                "llava_raw_label": "",
                "predicted_state": "ERROR",
                "p_safe": np.nan,
                "p_potentially_unsafe": np.nan,
                "p_unsafe": np.nan,
                "score_safe": np.nan,
                "score_potentially_unsafe": np.nan,
                "score_unsafe": np.nan,
                "error": str(e),
            })
            results.append(result)

        # Save after every frame so progress is not lost.
        pd.DataFrame(results).to_csv(out_csv, index=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
