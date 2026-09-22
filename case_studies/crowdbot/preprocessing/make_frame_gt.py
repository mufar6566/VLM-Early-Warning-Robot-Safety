#!/usr/bin/env python3
"""Create canonical frame-level CrowdBot safety ground truth.

This script deliberately knows nothing about APPFR, APPOD, or history length.
Those evaluation inputs must join to this table by ``frame_uid`` and use the
target of the newest frame in each causal history window.

The GT is sensor-derived (front LiDAR + odometry), so it is independent of the
VLM image inputs.  Only the front scan is used by default because the extracted
RGB stream is a forward/left camera; rear/global minima are not necessarily
visible to an image-only model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def causal_merge(left, right, left_time, right_time, tolerance):
    left = left.sort_values(left_time).reset_index(drop=True)
    right = right.sort_values(right_time).reset_index(drop=True)
    return pd.merge_asof(
        left,
        right,
        left_on=left_time,
        right_on=right_time,
        # APPOD must never receive a sensor observation recorded after the
        # corresponding image frame.
        direction="backward",
        tolerance=tolerance,
    )


def finite_numeric(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.where(np.isfinite(x))


def prepare_frames(path, bag_id):
    df = pd.read_csv(path)
    required = {"time_sec", "frame_index", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    out = pd.DataFrame({
        "dataset": "crowdbot",
        "bag_id": bag_id,
        "frame_index": pd.to_numeric(df["frame_index"], errors="coerce"),
        "frame_time_s": finite_numeric(df["time_sec"]),
        "frame_rel_time_s": finite_numeric(df.get("rel_time_sec", np.nan)),
        "image_path": df["image_path"].fillna("").astype(str),
    })
    out = out.dropna(subset=["frame_index", "frame_time_s"])
    out = out[out["image_path"].str.len() > 0].copy()
    out["frame_index"] = out["frame_index"].astype(int)
    out["frame_uid"] = (
        out["bag_id"].astype(str) + "_frame_"
        + out["frame_index"].astype(str).str.zfill(6)
    )
    return out.sort_values("frame_time_s").reset_index(drop=True)


def prepare_front_lidar(path, topic, smooth_samples):
    df = pd.read_csv(path)
    required = {"time_sec", "topic", "min_distance_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = df[df["topic"].astype(str).eq(topic)].copy()
    df["lidar_time_s"] = finite_numeric(df["time_sec"])
    df["front_min_distance_raw_m"] = finite_numeric(df["min_distance_m"])
    if "range_min" in df:
        rmin = finite_numeric(df["range_min"])
        df.loc[df["front_min_distance_raw_m"] < rmin, "front_min_distance_raw_m"] = np.nan

    df = df.dropna(subset=["lidar_time_s"]).sort_values("lidar_time_s")
    # A short causal median suppresses isolated single-beam spikes without
    # using future sensor values. Raw values remain available for audit.
    df["front_min_distance_m"] = df["front_min_distance_raw_m"].rolling(
        smooth_samples, min_periods=1
    ).median()

    # Positive closing speed means the closest front obstacle is approaching.
    dt = df["lidar_time_s"].diff()
    dd = df["front_min_distance_m"].diff()
    closing = -(dd / dt)
    closing = closing.where((dt > 0) & (closing >= 0))
    df["lidar_closing_speed_mps"] = closing.rolling(5, min_periods=2).median()
    return df[[
        "lidar_time_s", "front_min_distance_raw_m", "front_min_distance_m",
        "lidar_closing_speed_mps",
    ]]


def prepare_odometry(path):
    if not path.exists():
        return pd.DataFrame(columns=["odom_time_s", "robot_speed_mps", "angular_speed_radps"])
    df = pd.read_csv(path)
    if "time_sec" not in df:
        raise ValueError(f"{path}: missing time_sec")
    return pd.DataFrame({
        "odom_time_s": finite_numeric(df["time_sec"]),
        "robot_speed_mps": finite_numeric(df.get("linear_speed", np.nan)).abs(),
        "angular_speed_radps": finite_numeric(df.get("angular_speed", np.nan)),
    }).dropna(subset=["odom_time_s"]).sort_values("odom_time_s")


def assign_labels(df, args):
    """Assign canonical state from front clearance only.

    TTC and stopping distance are retained as secondary dynamic diagnostics.
    They do not change gt_state because TTC derived from the derivative of a
    scan-wide minimum can switch obstacle identity, and TF-derived speed can
    contain discontinuity spikes.
    """
    out = df.copy()
    d = out["front_min_distance_m"]
    v = out["robot_speed_mps"].fillna(0.0).clip(lower=0.0)
    closing = out["lidar_closing_speed_mps"]

    out["stopping_distance_m"] = (
        v * args.reaction_time_s
        + (v ** 2) / (2.0 * args.braking_deceleration_mps2)
        + args.stopping_margin_m
    )
    out["ttc_s"] = (d / closing).where(closing >= args.min_closing_speed_mps)
    unknown = d.isna()
    unsafe = ~unknown & d.lt(args.unsafe_distance_m)
    potential = (
        ~unknown
        & d.ge(args.unsafe_distance_m)
        & d.lt(args.caution_distance_m)
    )

    out["gt_state"] = "safe"
    out["gt_reason"] = f"front_clearance_ge_{args.caution_distance_m:.2f}m"
    out.loc[potential, "gt_state"] = "potentially_unsafe"
    out.loc[potential, "gt_reason"] = (
        f"front_clearance_{args.unsafe_distance_m:.2f}_to_"
        f"{args.caution_distance_m:.2f}m"
    )
    out.loc[unsafe, "gt_state"] = "unsafe"
    out.loc[unsafe, "gt_reason"] = f"front_clearance_lt_{args.unsafe_distance_m:.2f}m"
    out.loc[unknown, "gt_state"] = "unknown"
    out.loc[unknown, "gt_reason"] = "no_fresh_valid_front_lidar_match"

    # Secondary diagnostics only: never use these columns as VLM targets or
    # include them in APPFR prompts. APPOD may receive causal raw sensor fields,
    # but must not receive these derived risk flags.
    out["stopping_risk_flag"] = (
        ~unknown & d.le(out["stopping_distance_m"])
    ).astype(int)
    out["ttc_potential_flag"] = out["ttc_s"].le(args.potential_ttc_s).fillna(False).astype(int)
    out["ttc_unsafe_flag"] = out["ttc_s"].le(args.unsafe_ttc_s).fillna(False).astype(int)

    out["warning_required"] = out["gt_state"].isin(["potentially_unsafe", "unsafe"]).astype(int)
    out["evidence_quality"] = np.where(unknown, "missing", "fresh")
    return out


def add_future_event_annotations(df, horizon_s):
    """Add analysis targets without changing the current-frame safety state."""
    out = df.copy()
    out["unsafe_event_id"] = -1
    out["next_unsafe_time_s"] = np.nan
    out["lead_time_to_unsafe_s"] = np.nan
    out["warning_actionable"] = 0

    unsafe = out["gt_state"].eq("unsafe").to_numpy()
    starts = np.flatnonzero(unsafe & np.r_[True, ~unsafe[:-1]])
    times = out["frame_time_s"].to_numpy(float)
    for event_id, idx in enumerate(starts):
        event_t = times[idx]
        end = idx
        while end + 1 < len(out) and unsafe[end + 1]:
            end += 1
        out.loc[idx:end, "unsafe_event_id"] = event_id

        candidates = (times < event_t) & (times >= event_t - horizon_s) & (~unsafe)
        # Do not overwrite frames already assigned to a nearer unsafe event.
        candidates &= out["next_unsafe_time_s"].isna().to_numpy()
        out.loc[candidates, "unsafe_event_id"] = event_id
        out.loc[candidates, "next_unsafe_time_s"] = event_t
        out.loc[candidates, "lead_time_to_unsafe_s"] = event_t - times[candidates]

    # Actionable means a warning is required now and positive lead time remains.
    out["warning_actionable"] = (
        out["warning_required"].eq(1)
        & out["lead_time_to_unsafe_s"].gt(0)
    ).astype(int)
    return out


def process_bag(bag_dir, out_dir, args):
    bag_id = bag_dir.name
    frames = prepare_frames(bag_dir / "camera_left_color_index.csv", bag_id)
    lidar = prepare_front_lidar(
        bag_dir / "lidar_min_distance.csv", args.front_lidar_topic, args.lidar_smooth_samples
    )
    odom = prepare_odometry(bag_dir / "tf_qolo_odometry.csv")

    out = causal_merge(frames, lidar, "frame_time_s", "lidar_time_s", args.lidar_tolerance_s)
    if len(odom):
        out = causal_merge(out, odom, "frame_time_s", "odom_time_s", args.odom_tolerance_s)
    else:
        out["odom_time_s"] = np.nan
        out["robot_speed_mps"] = np.nan
        out["angular_speed_radps"] = np.nan

    out["lidar_dt_s"] = (out["frame_time_s"] - out["lidar_time_s"]).abs()
    out["odom_dt_s"] = (out["frame_time_s"] - out["odom_time_s"]).abs()
    out = assign_labels(out, args)
    out = add_future_event_annotations(out, args.early_warning_horizon_s)

    out["gt_version"] = "crowdbot_frame_gt_v2_distance_primary"
    out["target_definition"] = "current_newest_frame"
    columns = [
        "dataset", "bag_id", "frame_uid", "frame_index", "frame_time_s",
        "frame_rel_time_s", "image_path", "front_min_distance_raw_m",
        "front_min_distance_m", "lidar_time_s", "lidar_dt_s",
        "lidar_closing_speed_mps", "robot_speed_mps", "angular_speed_radps",
        "odom_time_s", "odom_dt_s", "stopping_distance_m", "ttc_s",
        "stopping_risk_flag", "ttc_potential_flag", "ttc_unsafe_flag",
        "gt_state", "gt_reason", "warning_required", "warning_actionable",
        "unsafe_event_id", "next_unsafe_time_s", "lead_time_to_unsafe_s",
        "evidence_quality", "gt_version", "target_definition",
    ]
    out = out[columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bag_id}_crowdbot_frame_gt.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path} ({len(out)} frames)")
    print(out["gt_state"].value_counts(dropna=False).to_string())
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--extracted-dir", type=Path, default=Path("01_extracted_data/extracted"))
    p.add_argument("--out-dir", type=Path, default=Path("02_ground_truth/frame_level"))
    p.add_argument(
        "--bags",
        nargs="*",
        default=None,
        help="Bag directory names. If omitted, discover every valid extracted bag.",
    )
    p.add_argument("--front-lidar-topic", default="/front_lidar/scan")
    p.add_argument("--lidar-tolerance-s", type=float, default=0.15)
    p.add_argument("--odom-tolerance-s", type=float, default=0.15)
    p.add_argument("--lidar-smooth-samples", type=int, default=3)
    p.add_argument("--unsafe-distance-m", type=float, default=0.60)
    p.add_argument("--caution-distance-m", type=float, default=1.20)
    p.add_argument("--reaction-time-s", type=float, default=0.50)
    p.add_argument("--braking-deceleration-mps2", type=float, default=0.80)
    p.add_argument("--stopping-margin-m", type=float, default=0.30)
    p.add_argument("--min-closing-speed-mps", type=float, default=0.10)
    p.add_argument("--unsafe-ttc-s", type=float, default=0.75)
    p.add_argument("--potential-ttc-s", type=float, default=3.00)
    p.add_argument("--early-warning-horizon-s", type=float, default=3.00)
    return p.parse_args()


def main():
    args = parse_args()
    if args.lidar_smooth_samples < 1 or args.lidar_smooth_samples % 2 == 0:
        raise ValueError("--lidar-smooth-samples must be a positive odd integer")

    if args.bags:
        bag_ids = args.bags
    else:
        required = {
            "camera_left_color_index.csv",
            "lidar_min_distance.csv",
            "tf_qolo_odometry.csv",
        }
        bag_ids = sorted(
            p.name for p in args.extracted_dir.iterdir()
            if p.is_dir() and required.issubset({x.name for x in p.iterdir()})
        )
        if not bag_ids:
            raise RuntimeError(
                f"No valid extracted bag directories found under {args.extracted_dir}"
            )
        print(f"Auto-discovered {len(bag_ids)} extracted bags")

    all_frames = []
    for bag_id in bag_ids:
        bag_dir = args.extracted_dir / bag_id
        if not bag_dir.exists():
            raise FileNotFoundError(bag_dir)
        all_frames.append(process_bag(bag_dir, args.out_dir, args))

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = args.out_dir / "crowdbot_all_bags_frame_gt.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved {combined_path} ({len(combined)} frames)")


if __name__ == "__main__":
    main()
