#!/usr/bin/env python3

from pathlib import Path
import argparse
import math
import re
import numpy as np
import pandas as pd


# ============================================================
# PAL GT parameters
# ============================================================

LASER_MIN_VALID_M = 0.10

UNSAFE_DIST_M = 0.35
POTENTIALLY_UNSAFE_DIST_M = 0.80

# Dynamic warning parameters, same idea as HuRoN
T_REACTION_S = 0.20
T_MARGIN_S = 0.50
A_BRAKE_MODERATE = 1.00
A_BRAKE_CONSERVATIVE = 0.50
MAX_WARNING_TIME_S = 3.00

# Nearest join tolerances
SCAN_TIME_TOL_S = 0.20
ODOM_TIME_TOL_S = 0.20
CMD_TIME_TOL_S = 0.20


# ============================================================
# Helper functions
# ============================================================

def normalize_col(c):
    return str(c).strip().lower()


def find_time_col(df):
    candidates = [
        "time_s", "timestamp_s", "stamp_s", "t_s",
        "time", "timestamp", "stamp", "sec", "secs",
        "frame_time_s", "image_time_s",
    ]
    lower_map = {normalize_col(c): c for c in df.columns}

    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    # Prefer columns containing time/stamp
    for c in df.columns:
        lc = normalize_col(c)
        if "time" in lc or "stamp" in lc:
            return c

    # Last fallback: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise ValueError(f"Could not find time column. Columns: {list(df.columns)}")


def find_image_col(df):
    candidates = [
        "image_path", "rgb_path", "path", "filename", "file",
        "image_file", "frame_path",
    ]
    lower_map = {normalize_col(c): c for c in df.columns}

    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    for c in df.columns:
        lc = normalize_col(c)
        if "image" in lc and ("path" in lc or "file" in lc or "name" in lc):
            return c

    return None


def find_frame_col(df):
    candidates = ["frame_id", "frame", "image_id", "idx", "index"]
    lower_map = {normalize_col(c): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def parse_numeric_list_cell(x):
    """
    Parse a cell like '[1.2, 0.3, inf]' or '1.2 0.3 inf' into floats.
    """
    if pd.isna(x):
        return []
    if isinstance(x, (int, float, np.number)):
        return [float(x)]

    s = str(x)
    vals = []
    for token in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+|inf|nan", s.lower()):
        try:
            vals.append(float(token))
        except Exception:
            pass
    return vals


def row_min_valid(values, min_valid=LASER_MIN_VALID_M):
    vals = []
    for v in values:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f) and f >= min_valid:
            vals.append(f)
    if not vals:
        return np.nan
    return float(np.min(vals))


def build_front_scan_min(scan_csv):
    """
    Build dataframe with columns:
      scan_time_s, front_min_distance_m

    Handles two common cases:
    1) wide format: each row has many numeric range columns
    2) long format: rows have time + range column
    """
    df = pd.read_csv(scan_csv)
    time_col = find_time_col(df)

    # Normalize time to float seconds
    out_time = pd.to_numeric(df[time_col], errors="coerce")

    lower_cols = {normalize_col(c): c for c in df.columns}

    # Case A: explicit min/range column
    # PAL extracted scan files contain these important columns:
    #   raw_min_range
    #   clean_min_range_gt_0.10
    # Prefer clean_min_range_gt_0.10 because it already removes readings <= 0.10 m.
    explicit_min_cols = [
        "clean_min_range_gt_0.10",
        "raw_min_range",
        "front_clean_min_range_m",
        "front_min_range_m",
        "min_range_m",
        "min_range",
        "range_m",
        "range",
        "distance_m",
        "distance",
    ]

    range_col = None
    for c in explicit_min_cols:
        if c in lower_cols:
            range_col = lower_cols[c]
            break

    # Long format: time + one range column
    if range_col is not None:
        tmp = pd.DataFrame({
            "scan_time_s": out_time,
            "range_value": pd.to_numeric(df[range_col], errors="coerce"),
        })
        tmp = tmp.dropna(subset=["scan_time_s", "range_value"])
        tmp = tmp[np.isfinite(tmp["range_value"])]
        tmp = tmp[tmp["range_value"] >= LASER_MIN_VALID_M]
        if tmp.empty:
            return pd.DataFrame(columns=["scan_time_s", "front_min_distance_m"])
        g = tmp.groupby("scan_time_s", as_index=False)["range_value"].min()
        g = g.rename(columns={"range_value": "front_min_distance_m"})
        return g.sort_values("scan_time_s").reset_index(drop=True)

    # Case B: wide numeric range columns
    ignore_words = [
        "time", "stamp", "sec", "nsec", "angle", "increment",
        "min", "max", "header", "seq", "frame", "id"
    ]

    numeric_cols = []
    for c in df.columns:
        lc = normalize_col(c)
        if c == time_col:
            continue
        if any(w in lc for w in ignore_words):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    # If no numeric wide columns, try string list columns
    rows = []
    if numeric_cols:
        for t, (_, r) in zip(out_time, df.iterrows()):
            vals = [r[c] for c in numeric_cols]
            m = row_min_valid(vals)
            rows.append((t, m))
    else:
        # Try any object column containing list-like ranges
        object_cols = [c for c in df.columns if c != time_col]
        for t, (_, r) in zip(out_time, df.iterrows()):
            vals = []
            for c in object_cols:
                vals.extend(parse_numeric_list_cell(r[c]))
            m = row_min_valid(vals)
            rows.append((t, m))

    out = pd.DataFrame(rows, columns=["scan_time_s", "front_min_distance_m"])
    out = out.dropna(subset=["scan_time_s"])
    out = out.sort_values("scan_time_s").reset_index(drop=True)
    return out


def build_speed_df(csv_path, preferred="odom"):
    """
    Build dataframe:
      speed_time_s, speed_xy
    Works with odometry.csv or cmd_vel.csv if common column names exist.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["speed_time_s", "speed_xy"])

    df = pd.read_csv(csv_path)
    time_col = find_time_col(df)
    t = pd.to_numeric(df[time_col], errors="coerce")

    lower_map = {normalize_col(c): c for c in df.columns}

    # Try explicit speed columns first
    speed_candidates = [
        "speed_xy", "avg_speed_xy", "linear_speed", "speed",
        "v", "vel", "velocity", "cmd_speed", "avg_cmd_speed_xy"
    ]
    for c in speed_candidates:
        if c in lower_map:
            sp = pd.to_numeric(df[lower_map[c]], errors="coerce").abs()
            return pd.DataFrame({"speed_time_s": t, "speed_xy": sp}).dropna().sort_values("speed_time_s")

    # Try vx/vy or linear x/y
    x_candidates = [
        "vx", "linear_x", "linear.x", "twist_linear_x",
        "odom_linear_x", "cmd_vel_linear_x", "x"
    ]
    y_candidates = [
        "vy", "linear_y", "linear.y", "twist_linear_y",
        "odom_linear_y", "cmd_vel_linear_y", "y"
    ]

    x_col = None
    y_col = None

    for c in x_candidates:
        if c in lower_map:
            x_col = lower_map[c]
            break

    for c in y_candidates:
        if c in lower_map:
            y_col = lower_map[c]
            break

    if x_col is not None and y_col is not None:
        vx = pd.to_numeric(df[x_col], errors="coerce")
        vy = pd.to_numeric(df[y_col], errors="coerce")
        sp = np.sqrt(vx * vx + vy * vy)
    elif x_col is not None:
        sp = pd.to_numeric(df[x_col], errors="coerce").abs()
    else:
        sp = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({"speed_time_s": t, "speed_xy": sp})
    out = out.dropna(subset=["speed_time_s"])
    out = out.sort_values("speed_time_s").reset_index(drop=True)
    return out


def nearest_merge(left, right, left_time, right_time, tolerance):
    if right.empty:
        return left.copy()

    l = left.sort_values(left_time).reset_index(drop=True)
    r = right.sort_values(right_time).reset_index(drop=True)

    return pd.merge_asof(
        l,
        r,
        left_on=left_time,
        right_on=right_time,
        direction="nearest",
        tolerance=tolerance,
    )


def label_from_distance(d):
    if pd.isna(d):
        return "unknown", "no_valid_front_scan_distance"

    if d < UNSAFE_DIST_M:
        return "unsafe", f"front_distance_lt_{UNSAFE_DIST_M:.2f}m"

    if d < POTENTIALLY_UNSAFE_DIST_M:
        return "potentially_unsafe", f"front_distance_{UNSAFE_DIST_M:.2f}_to_{POTENTIALLY_UNSAFE_DIST_M:.2f}m"

    return "safe", f"front_distance_ge_{POTENTIALLY_UNSAFE_DIST_M:.2f}m"


def add_dynamic_warning_targets(df):
    df = df.copy()

    df["t_required_moderate_s"] = T_REACTION_S + (df["speed_mps"].fillna(0.0) / A_BRAKE_MODERATE) + T_MARGIN_S
    df["t_required_conservative_s"] = T_REACTION_S + (df["speed_mps"].fillna(0.0) / A_BRAKE_CONSERVATIVE) + T_MARGIN_S

    df["lead_time_to_unsafe_s"] = np.nan
    df["warning_target_moderate"] = 0
    df["warning_target_conservative"] = 0
    df["warning_type_moderate"] = "none"
    df["warning_type_conservative"] = "none"
    df["unsafe_event_id"] = -1
    df["unsafe_event_start_s"] = np.nan
    df["warning_window_start_s"] = np.nan
    df["warning_window_end_s"] = np.nan

    unsafe_idx = df.index[df["gt_state"] == "unsafe"].tolist()
    if not unsafe_idx:
        df["warning_target"] = df["warning_target_conservative"]
        df["warning_type"] = df["warning_type_conservative"]
        return df

    # Build contiguous unsafe events
    events = []
    current = [unsafe_idx[0]]
    for idx in unsafe_idx[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            events.append(current)
            current = [idx]
    events.append(current)

    for event_id, inds in enumerate(events):
        start_i = inds[0]
        unsafe_start_t = float(df.loc[start_i, "frame_time_s"])

        window_start_t = unsafe_start_t - MAX_WARNING_TIME_S
        window_end_t = unsafe_start_t

        # Frames before unsafe event, inside warning window
        mask = (
            (df["frame_time_s"] < unsafe_start_t) &
            (df["frame_time_s"] >= window_start_t)
        )

        candidate_indices = df.index[mask].tolist()

        for i in candidate_indices:
            lead = unsafe_start_t - float(df.loc[i, "frame_time_s"])
            df.loc[i, "lead_time_to_unsafe_s"] = lead
            df.loc[i, "unsafe_event_id"] = event_id
            df.loc[i, "unsafe_event_start_s"] = unsafe_start_t
            df.loc[i, "warning_window_start_s"] = window_start_t
            df.loc[i, "warning_window_end_s"] = window_end_t

            if lead >= float(df.loc[i, "t_required_moderate_s"]):
                df.loc[i, "warning_target_moderate"] = 1
                df.loc[i, "warning_type_moderate"] = "actionable_moderate"

            if lead >= float(df.loc[i, "t_required_conservative_s"]):
                df.loc[i, "warning_target_conservative"] = 1
                df.loc[i, "warning_type_conservative"] = "actionable_conservative"

    df["warning_target"] = df["warning_target_conservative"]
    df["warning_type"] = df["warning_type_conservative"]

    return df


def resolve_image_path(bag_dir, image_value):
    if image_value is None or pd.isna(image_value):
        return ""

    s = str(image_value)
    p = Path(s)

    if p.is_absolute():
        return str(p)

    candidate = bag_dir / s
    if candidate.exists():
        return str(candidate)

    candidate = bag_dir / "images" / s
    if candidate.exists():
        return str(candidate)

    return str(bag_dir / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag_dir", required=True, help="Path to one extracted PAL bag folder")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--speed_scale", type=float, default=1.0, help="Multiply speed by this scale if needed")
    args = ap.parse_args()

    bag_dir = Path(args.bag_dir).expanduser().resolve()
    csv_dir = bag_dir / "csv"

    image_csv = csv_dir / "image_index.csv"
    if not image_csv.exists():
        raise FileNotFoundError(image_csv)

    # Prefer front raw scan for PAL
    scan_candidates = [
        csv_dir / "scan_front_raw.csv",
        csv_dir / "scan.csv",
    ]
    scan_csv = None
    for c in scan_candidates:
        if c.exists():
            scan_csv = c
            break

    if scan_csv is None:
        raise FileNotFoundError("No scan_front_raw.csv or scan.csv found")

    print("Bag:", bag_dir.name)
    print("Image CSV:", image_csv)
    print("Scan CSV:", scan_csv)

    img = pd.read_csv(image_csv)
    img_time_col = find_time_col(img)
    img_frame_col = find_frame_col(img)
    img_path_col = find_image_col(img)

    out = pd.DataFrame()

    if img_frame_col:
        out["frame_id"] = pd.to_numeric(img[img_frame_col], errors="coerce").fillna(np.arange(len(img))).astype(int)
    else:
        out["frame_id"] = np.arange(len(img), dtype=int)

    out["frame_time_s"] = pd.to_numeric(img[img_time_col], errors="coerce")

    if img_path_col:
        out["image_path"] = [resolve_image_path(bag_dir, v) for v in img[img_path_col]]
    else:
        out["image_path"] = ""

    out = out.dropna(subset=["frame_time_s"]).sort_values("frame_time_s").reset_index(drop=True)
    out["bag_name"] = bag_dir.name

    scan_min = build_front_scan_min(scan_csv)

    out = nearest_merge(
        out,
        scan_min,
        left_time="frame_time_s",
        right_time="scan_time_s",
        tolerance=SCAN_TIME_TOL_S,
    )

    # Speed: prefer odometry, fallback cmd_vel
    odom_speed = build_speed_df(csv_dir / "odometry.csv", preferred="odom")
    cmd_speed = build_speed_df(csv_dir / "cmd_vel.csv", preferred="cmd")

    if not odom_speed.empty and odom_speed["speed_xy"].notna().any():
        speed = odom_speed
        speed_source = "odometry"
        tol = ODOM_TIME_TOL_S
    else:
        speed = cmd_speed
        speed_source = "cmd_vel"
        tol = CMD_TIME_TOL_S

    if not speed.empty:
        out = nearest_merge(
            out,
            speed,
            left_time="frame_time_s",
            right_time="speed_time_s",
            tolerance=tol,
        )
        out["speed_source"] = speed_source
    else:
        out["speed_time_s"] = np.nan
        out["speed_xy"] = np.nan
        out["speed_source"] = "missing"

    out["speed_mps"] = pd.to_numeric(out.get("speed_xy", np.nan), errors="coerce") * args.speed_scale

    labels = out["front_min_distance_m"].apply(label_from_distance)
    out["gt_state"] = [x[0] for x in labels]
    out["gt_reason"] = [x[1] for x in labels]

    out = add_dynamic_warning_targets(out)

    # Arrange columns
    ordered = [
        "bag_name",
        "frame_id",
        "frame_time_s",
        "image_path",
        "front_min_distance_m",
        "scan_time_s",
        "speed_source",
        "speed_time_s",
        "speed_xy",
        "speed_mps",
        "gt_state",
        "gt_reason",
        "t_required_moderate_s",
        "t_required_conservative_s",
        "lead_time_to_unsafe_s",
        "warning_target_moderate",
        "warning_target_conservative",
        "warning_type_moderate",
        "warning_type_conservative",
        "unsafe_event_id",
        "unsafe_event_start_s",
        "warning_window_start_s",
        "warning_window_end_s",
        "warning_target",
        "warning_type",
    ]

    out = out[[c for c in ordered if c in out.columns]]

    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser().resolve()
    else:
        pal_root = bag_dir.parents[1]
        out_dir = pal_root / "02_ground_truth"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{bag_dir.name}_frame_level_gt_front_scan_dynamic_warning.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("Rows:", len(out))

    print("\nGT counts:")
    print(out["gt_state"].value_counts(dropna=False).to_string())

    print("\nWarning target counts:")
    print("moderate:")
    print(out["warning_target_moderate"].value_counts(dropna=False).to_string())
    print("conservative:")
    print(out["warning_target_conservative"].value_counts(dropna=False).to_string())

    print("\nDistance summary:")
    print(out["front_min_distance_m"].describe().to_string())

    print("\nFirst rows:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
