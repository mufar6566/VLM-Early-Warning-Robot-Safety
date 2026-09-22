from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Use one visual stream only.
# Options: "fisheye", "panorama", "spherical"
IMAGE_VIEW = "fisheye"

# Timestamp tolerance for matching image frame with nearest sensor values.
NEAREST_TOL_S = 0.15

# HuRoN safety distance thresholds.
UNSAFE_PED_DIST_M = 0.6
POTENTIALLY_UNSAFE_PED_DIST_M = 1.2

# Laser is kept as evidence only for now.
USE_LASER_FOR_GT = False

# Ignore LiDAR readings below this value to reduce noise/self-detection artifacts.
# In HuRoN laserscan.csv, range_min is around 0.15 m.
LASER_MIN_VALID_M = 0.15

# Early-warning setting.
# A warning is considered only inside the final MAX_WARNING_TIME_S before unsafe.
MAX_WARNING_TIME_S = 3.0

# Dynamic actionable-warning formula:
# T_required = T_reaction + speed_mps / a_brake + T_margin
T_REACTION_S = 0.2
T_MARGIN_S = 0.5

# Two braking profiles.
A_BRAKE_MODERATE = 1.0       # m/s^2
A_BRAKE_CONSERVATIVE = 0.5   # m/s^2

# HuRoN odometry speed_xy appears to be in cm/s-like scale.
# Example: raw speed 11.7 -> 0.117 m/s.
ODOM_SPEED_SCALE_TO_MPS = 0.01


# ============================================================
# PATH HANDLING
# ============================================================

cwd = Path(".").resolve()

# Script can run from bag root or from csv folder.
if (cwd / "csv").exists():
    BAG_DIR = cwd
    CSV_DIR = BAG_DIR / "csv"
elif cwd.name == "csv":
    CSV_DIR = cwd
    BAG_DIR = cwd.parent
else:
    raise RuntimeError("Run this script from the bag root folder or from its csv folder.")

BAG_NAME = BAG_DIR.name

# Expected structure:
# HuRon/
# ├── 01_extracted_data/
# │   └── bag_folder/
# └── 02_ground_truth/
OUT_DIR = BAG_DIR.parents[1] / "02_ground_truth" / "full15_bags"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / f"{BAG_NAME}_frame_level_gt_{IMAGE_VIEW}_dynamic_warning.csv"


# ============================================================
# BASIC HELPERS
# ============================================================

def read_csv(name):
    path = CSV_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"\nReading: {path}")
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    return df


def find_time_col(df):
    candidates = [
        "time_s",
        "timestamp_s",
        "relative_time_s",
        "elapsed_time_s",
        "stamp_s",
        "t_s",
        "time",
        "timestamp",
        "stamp",
        "t",
    ]

    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in lower:
            return lower[c]

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise ValueError(f"No time column found in: {df.columns.tolist()}")


def make_time_s(df, ref_start=None):
    time_col = find_time_col(df)
    t = pd.to_numeric(df[time_col], errors="coerce")

    med = t.dropna().median()

    # Convert ns/us to seconds if needed.
    if med > 1e12:
        t = t / 1e9
    elif med > 1e9:
        t = t / 1e6

    # If absolute ROS/Unix time, subtract image reference start.
    if ref_start is not None and t.dropna().median() > 100000:
        t = t - ref_start

    # If already relative but starts from offset > 10 sec, normalize.
    elif t.dropna().min() > 10:
        t = t - t.dropna().min()

    return t.astype(float)


def nearest_row(df, t, tol=NEAREST_TOL_S):
    if df is None or len(df) == 0:
        return None

    idx = (df["time_s_norm"] - t).abs().idxmin()
    row = df.loc[idx]
    dt = abs(float(row["time_s_norm"]) - t)

    if dt > tol:
        return None

    return row


def rows_near_time(df, t, tol=NEAREST_TOL_S):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    return df[
        (df["time_s_norm"] >= t - tol)
        & (df["time_s_norm"] <= t + tol)
    ]


def bool_value(x):
    if pd.isna(x):
        return False

    if isinstance(x, str):
        return x.strip().lower() in ["1", "true", "yes", "pressed", "active"]

    return bool(x)


# ============================================================
# SENSOR FEATURE EXTRACTION
# ============================================================

def get_bumper_status(row):
    """
    Correct bumper/proximity logic.

    Contact:
      - is_left_pressed
      - is_right_pressed

    Proximity:
      - is_light_center_left
      - is_light_center_right
      - is_light_front_left
      - is_light_front_right
      - is_light_left
      - is_light_right

    Important:
      We do NOT use numeric light_signal_* columns as boolean flags.
    """

    if row is None:
        return False, False

    contact_cols = [
        "is_left_pressed",
        "is_right_pressed",
    ]

    proximity_cols = [
        "is_light_center_left",
        "is_light_center_right",
        "is_light_front_left",
        "is_light_front_right",
        "is_light_left",
        "is_light_right",
    ]

    contact_active = any(
        bool_value(row[c]) for c in contact_cols if c in row.index
    )

    proximity_active = any(
        bool_value(row[c]) for c in proximity_cols if c in row.index
    )

    return contact_active, proximity_active


def get_closest_pedestrian_distance(rows):
    if rows is None or len(rows) == 0:
        return np.inf

    # Preferred column in HuRoN pedestrians_pose.csv.
    if "distance_xy" in rows.columns:
        d = pd.to_numeric(rows["distance_xy"], errors="coerce")
        if not d.dropna().empty:
            return float(d.min())

    # Fallback: any distance-like column.
    dist_cols = [
        c for c in rows.columns
        if "distance" in c.lower() or c.lower() in ["dist", "ped_distance"]
    ]

    if dist_cols:
        d = pd.to_numeric(rows[dist_cols[0]], errors="coerce")
        if not d.dropna().empty:
            return float(d.min())

    # Fallback: compute sqrt(x^2+y^2).
    if "x" in rows.columns and "y" in rows.columns:
        x = pd.to_numeric(rows["x"], errors="coerce")
        y = pd.to_numeric(rows["y"], errors="coerce")
        d = np.sqrt(x**2 + y**2)
        if not d.dropna().empty:
            return float(d.min())

    return np.inf


def get_min_laser_distance(rows):
    """
    Laser evidence only.

    Use actual measured min_range, not sensor range_min.
    Ignore values below LASER_MIN_VALID_M to reduce noise/self-detection artifacts.
    """

    if rows is None or len(rows) == 0:
        return np.nan

    if "min_range" in rows.columns:
        vals = pd.to_numeric(rows["min_range"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        vals = vals[vals >= LASER_MIN_VALID_M]
        if len(vals):
            return float(vals.min())

    fallback_cols = [
        c for c in rows.columns
        if any(k in c.lower() for k in ["min_valid", "min_distance"])
    ]

    if fallback_cols:
        vals = pd.to_numeric(rows[fallback_cols[0]], errors="coerce")
        vals = vals[np.isfinite(vals)]
        vals = vals[vals >= LASER_MIN_VALID_M]
        if len(vals):
            return float(vals.min())

    return np.nan


def get_odometry_features(rows):
    """
    Use odometry columns:
      - speed_xy as raw speed
      - angular_z as yaw/angular speed

    We also convert raw speed to m/s using ODOM_SPEED_SCALE_TO_MPS.
    """

    if rows is None or len(rows) == 0:
        return np.nan, np.nan, np.nan

    speed_raw = np.nan
    speed_mps = np.nan
    angular_speed = np.nan

    if "speed_xy" in rows.columns:
        vals = pd.to_numeric(rows["speed_xy"], errors="coerce")
        if not vals.dropna().empty:
            speed_raw = float(vals.mean())
            speed_mps = speed_raw * ODOM_SPEED_SCALE_TO_MPS

    elif "linear_x" in rows.columns and "linear_y" in rows.columns:
        vx = pd.to_numeric(rows["linear_x"], errors="coerce")
        vy = pd.to_numeric(rows["linear_y"], errors="coerce")
        speed = np.sqrt(vx**2 + vy**2)
        if not speed.dropna().empty:
            speed_raw = float(speed.mean())
            speed_mps = speed_raw * ODOM_SPEED_SCALE_TO_MPS

    if "angular_z" in rows.columns:
        vals = pd.to_numeric(rows["angular_z"], errors="coerce")
        if not vals.dropna().empty:
            angular_speed = float(vals.mean())

    return speed_raw, speed_mps, angular_speed


# ============================================================
# GT LABELING
# ============================================================

def assign_gt(contact_active, proximity_active, closest_ped_m, min_laser_m):
    """
    Frame-level HuRoN GT rule.

    Unsafe:
      bumper/contact pressed OR closest pedestrian < 0.6 m

    Potentially unsafe:
      boolean proximity active OR pedestrian distance in [0.6, 1.2)

    Safe:
      no contact/proximity and pedestrian distance >= 1.2 m

    Laser:
      evidence only unless USE_LASER_FOR_GT=True.
    """

    if contact_active or closest_ped_m < UNSAFE_PED_DIST_M:
        return "unsafe"

    if USE_LASER_FOR_GT:
        if np.isfinite(min_laser_m) and min_laser_m < UNSAFE_PED_DIST_M:
            return "unsafe"

    if proximity_active or (
        UNSAFE_PED_DIST_M <= closest_ped_m < POTENTIALLY_UNSAFE_PED_DIST_M
    ):
        return "potentially_unsafe"

    if USE_LASER_FOR_GT:
        if np.isfinite(min_laser_m) and (
            UNSAFE_PED_DIST_M <= min_laser_m < POTENTIALLY_UNSAFE_PED_DIST_M
        ):
            return "potentially_unsafe"

    return "safe"


def make_reason(contact_active, proximity_active, closest_ped_m, min_laser_m):
    reasons = []

    if contact_active:
        reasons.append("bumper/contact pressed")

    if closest_ped_m < UNSAFE_PED_DIST_M:
        reasons.append(
            f"closest pedestrian {closest_ped_m:.3f} m < {UNSAFE_PED_DIST_M:.2f} m"
        )

    if USE_LASER_FOR_GT:
        if np.isfinite(min_laser_m) and min_laser_m < UNSAFE_PED_DIST_M:
            reasons.append(
                f"laser min_range {min_laser_m:.3f} m < {UNSAFE_PED_DIST_M:.2f} m"
            )

    if proximity_active:
        reasons.append("boolean proximity/light active")

    if UNSAFE_PED_DIST_M <= closest_ped_m < POTENTIALLY_UNSAFE_PED_DIST_M:
        reasons.append(
            f"closest pedestrian {closest_ped_m:.3f} m in "
            f"[{UNSAFE_PED_DIST_M:.2f}, {POTENTIALLY_UNSAFE_PED_DIST_M:.2f}) m"
        )

    if USE_LASER_FOR_GT:
        if np.isfinite(min_laser_m) and (
            UNSAFE_PED_DIST_M <= min_laser_m < POTENTIALLY_UNSAFE_PED_DIST_M
        ):
            reasons.append(
                f"laser min_range {min_laser_m:.3f} m in "
                f"[{UNSAFE_PED_DIST_M:.2f}, {POTENTIALLY_UNSAFE_PED_DIST_M:.2f}) m"
            )

    if not reasons:
        reasons.append(
            "no contact/proximity and closest pedestrian >= "
            f"{POTENTIALLY_UNSAFE_PED_DIST_M:.2f} m"
        )

    return "; ".join(reasons)


# ============================================================
# DYNAMIC EARLY-WARNING GT
# ============================================================

def compute_t_required(speed_mps, a_brake):
    """
    T_required = T_reaction + speed_mps / a_brake + T_margin
    """

    if pd.isna(speed_mps):
        speed_mps = 0.0

    return T_REACTION_S + (speed_mps / a_brake) + T_MARGIN_S


def add_warning_targets(df):
    """
    Dynamic actionable-warning GT.

    For each unsafe event starting at t_unsafe:

      lead_time_to_unsafe = t_unsafe - frame_time

      T_required = T_reaction + speed_mps / a_brake + T_margin

    A warning is actionable if:
      1. It occurs before unsafe.
      2. It is inside the final MAX_WARNING_TIME_S before unsafe.
      3. lead_time_to_unsafe >= T_required.

    We compute two profiles:
      - moderate braking: a_brake = 1.0 m/s^2
      - conservative braking: a_brake = 0.5 m/s^2
    """

    df = df.copy()

    df["t_required_moderate_s"] = df["odom_speed_mps"].apply(
        lambda v: compute_t_required(v, A_BRAKE_MODERATE)
    )

    df["t_required_conservative_s"] = df["odom_speed_mps"].apply(
        lambda v: compute_t_required(v, A_BRAKE_CONSERVATIVE)
    )

    df["lead_time_to_unsafe_s"] = np.nan

    df["warning_target_moderate"] = 0
    df["warning_target_conservative"] = 0

    df["warning_type_moderate"] = "none"
    df["warning_type_conservative"] = "none"

    df["unsafe_event_id"] = -1
    df["unsafe_event_start_s"] = np.nan
    df["warning_window_start_s"] = np.nan
    df["warning_window_end_s"] = np.nan

    unsafe = df["gt_state"].eq("unsafe").to_numpy()

    event_id = 0
    i = 0

    while i < len(df):
        if not unsafe[i]:
            i += 1
            continue

        start_i = i

        while i < len(df) and unsafe[i]:
            i += 1

        end_i = i - 1

        t_unsafe = float(df.loc[start_i, "frame_time_s"])
        warning_window_start = t_unsafe - MAX_WARNING_TIME_S
        warning_window_end = t_unsafe

        # Mark unsafe event frames.
        df.loc[start_i:end_i, "unsafe_event_id"] = event_id
        df.loc[start_i:end_i, "unsafe_event_start_s"] = t_unsafe

        # Candidate frames before unsafe, inside max warning window.
        candidate_mask = (
            (df["frame_time_s"] >= warning_window_start)
            & (df["frame_time_s"] < t_unsafe)
            & (~df["gt_state"].eq("unsafe"))
        )

        df.loc[candidate_mask, "lead_time_to_unsafe_s"] = (
            t_unsafe - df.loc[candidate_mask, "frame_time_s"]
        )

        df.loc[candidate_mask, "unsafe_event_id"] = event_id
        df.loc[candidate_mask, "unsafe_event_start_s"] = t_unsafe
        df.loc[candidate_mask, "warning_window_start_s"] = warning_window_start
        df.loc[candidate_mask, "warning_window_end_s"] = warning_window_end

        moderate_mask = (
            candidate_mask
            & (df["lead_time_to_unsafe_s"] >= df["t_required_moderate_s"])
            & (df["lead_time_to_unsafe_s"] <= MAX_WARNING_TIME_S)
        )

        conservative_mask = (
            candidate_mask
            & (df["lead_time_to_unsafe_s"] >= df["t_required_conservative_s"])
            & (df["lead_time_to_unsafe_s"] <= MAX_WARNING_TIME_S)
        )

        df.loc[moderate_mask, "warning_target_moderate"] = 1
        df.loc[moderate_mask, "warning_type_moderate"] = "actionable_warning"

        df.loc[conservative_mask, "warning_target_conservative"] = 1
        df.loc[conservative_mask, "warning_type_conservative"] = "actionable_warning"

        event_id += 1

    # Backward-compatible main warning columns.
    # Use conservative as the main/default because it is stricter.
    df["warning_target"] = df["warning_target_conservative"]
    df["warning_type"] = df["warning_type_conservative"]

    return df


# ============================================================
# MAIN
# ============================================================

print("\n============================================================")
print("Frame-level GT generation with dynamic actionable warning")
print("============================================================")
print("Bag:", BAG_NAME)
print("Bag dir:", BAG_DIR)
print("CSV dir:", CSV_DIR)
print("Output:", OUT_CSV)
print("Image view:", IMAGE_VIEW)
print("Nearest tolerance:", NEAREST_TOL_S)
print("Unsafe pedestrian distance:", UNSAFE_PED_DIST_M)
print("Potentially unsafe pedestrian distance:", POTENTIALLY_UNSAFE_PED_DIST_M)
print("Use laser for GT:", USE_LASER_FOR_GT)
print("Laser min valid:", LASER_MIN_VALID_M)
print("Max warning window:", MAX_WARNING_TIME_S)
print("Reaction time:", T_REACTION_S)
print("Safety margin:", T_MARGIN_S)
print("Moderate braking a:", A_BRAKE_MODERATE)
print("Conservative braking a:", A_BRAKE_CONSERVATIVE)
print("Odometry speed scale to m/s:", ODOM_SPEED_SCALE_TO_MPS)


# ----------------------------
# Load CSVs
# ----------------------------

image_df = read_csv("image_index.csv")
bumper_df = read_csv("bumper.csv")
ped_df = read_csv("pedestrians_pose.csv")
laser_df = read_csv("laserscan.csv")

odometry_path = CSV_DIR / "odometry.csv"
odometry_df = read_csv("odometry.csv") if odometry_path.exists() else None


# ----------------------------
# Build time columns
# ----------------------------

image_time_col = find_time_col(image_df)
raw_img_time = pd.to_numeric(image_df[image_time_col], errors="coerce")

if raw_img_time.dropna().median() > 1e12:
    image_ref_start = raw_img_time.dropna().min() / 1e9
elif raw_img_time.dropna().median() > 1e9:
    image_ref_start = raw_img_time.dropna().min() / 1e6
else:
    image_ref_start = None

image_df["frame_time_s"] = make_time_s(image_df, ref_start=image_ref_start)
bumper_df["time_s_norm"] = make_time_s(bumper_df, ref_start=image_ref_start)
ped_df["time_s_norm"] = make_time_s(ped_df, ref_start=image_ref_start)
laser_df["time_s_norm"] = make_time_s(laser_df, ref_start=image_ref_start)

if odometry_df is not None:
    odometry_df["time_s_norm"] = make_time_s(odometry_df, ref_start=image_ref_start)


# ----------------------------
# Find image path column
# ----------------------------

image_path_cols = [
    c for c in image_df.columns
    if any(k in c.lower() for k in ["path", "file", "filename", "image"])
]

if not image_path_cols:
    raise RuntimeError("No image path column found in image_index.csv.")

img_col = image_path_cols[0]


# ----------------------------
# Keep only selected image view
# ----------------------------

before_count = len(image_df)

image_df = image_df[
    image_df[img_col].astype(str).str.contains(IMAGE_VIEW, case=False, na=False)
].copy()

image_df = image_df.sort_values("frame_time_s").reset_index(drop=True)

after_count = len(image_df)

print("\nImage filtering:")
print(f"  Before filtering: {before_count}")
print(f"  After {IMAGE_VIEW}-only filtering: {after_count}")

if after_count == 0:
    raise RuntimeError(f"No images found for IMAGE_VIEW='{IMAGE_VIEW}'.")


# ----------------------------
# Generate frame-level GT
# ----------------------------

rows = []

for frame_id, frame in image_df.iterrows():
    t = float(frame["frame_time_s"])

    bumper_row = nearest_row(bumper_df, t)
    ped_rows = rows_near_time(ped_df, t)
    laser_rows = rows_near_time(laser_df, t)
    odom_rows = rows_near_time(odometry_df, t) if odometry_df is not None else pd.DataFrame()

    contact_active, proximity_active = get_bumper_status(bumper_row)
    closest_ped_m = get_closest_pedestrian_distance(ped_rows)
    min_laser_m = get_min_laser_distance(laser_rows)
    odom_speed_raw, odom_speed_mps, odom_angular_speed = get_odometry_features(odom_rows)

    gt_state = assign_gt(
        contact_active=contact_active,
        proximity_active=proximity_active,
        closest_ped_m=closest_ped_m,
        min_laser_m=min_laser_m,
    )

    gt_reason = make_reason(
        contact_active=contact_active,
        proximity_active=proximity_active,
        closest_ped_m=closest_ped_m,
        min_laser_m=min_laser_m,
    )

    rows.append({
        "bag_name": BAG_NAME,
        "frame_id": frame_id,
        "frame_time_s": t,
        "image_view": IMAGE_VIEW,
        "image_path": str(frame[img_col]),

        "contact_active": contact_active,
        "proximity_active": proximity_active,
        "closest_pedestrian_distance_m": (
            closest_ped_m if np.isfinite(closest_ped_m) else np.nan
        ),
        "min_laser_distance_m": min_laser_m,

        "odom_speed_raw": odom_speed_raw,
        "odom_speed_mps": odom_speed_mps,
        "odom_angular_speed": odom_angular_speed,

        "gt_state": gt_state,
        "gt_reason": gt_reason,
    })


gt = pd.DataFrame(rows)
gt = gt.sort_values("frame_time_s").reset_index(drop=True)
gt = add_warning_targets(gt)

gt.to_csv(OUT_CSV, index=False)


# ----------------------------
# Summary
# ----------------------------

print("\n============================================================")
print("Saved:", OUT_CSV)
print("============================================================")

print("\nGT state counts:")
print(gt["gt_state"].value_counts(dropna=False))

print("\nWarning target counts, moderate:")
print(gt["warning_target_moderate"].value_counts(dropna=False))

print("\nWarning target counts, conservative/main:")
print(gt["warning_target_conservative"].value_counts(dropna=False))

print("\nContact active count:")
print(gt["contact_active"].value_counts(dropna=False))

print("\nProximity active count:")
print(gt["proximity_active"].value_counts(dropna=False))

print("\nLaser min_range summary:")
print(gt["min_laser_distance_m"].describe())

print("\nOdometry raw speed summary:")
print(gt["odom_speed_raw"].describe())

print("\nOdometry speed m/s summary:")
print(gt["odom_speed_mps"].describe())

print("\nT required moderate summary:")
print(gt["t_required_moderate_s"].describe())

print("\nT required conservative summary:")
print(gt["t_required_conservative_s"].describe())

print("\nUnsafe rows:")
unsafe_rows = gt[gt["gt_state"] == "unsafe"]
if len(unsafe_rows):
    print(
        unsafe_rows[
            [
                "frame_id",
                "frame_time_s",
                "image_path",
                "contact_active",
                "proximity_active",
                "closest_pedestrian_distance_m",
                "min_laser_distance_m",
                "odom_speed_mps",
                "odom_angular_speed",
                "gt_reason",
            ]
        ].to_string(index=False)
    )
else:
    print("No unsafe rows found.")

print("\nActionable warning rows, moderate:")
warning_rows_m = gt[gt["warning_target_moderate"] == 1]
if len(warning_rows_m):
    print(
        warning_rows_m[
            [
                "frame_id",
                "frame_time_s",
                "gt_state",
                "lead_time_to_unsafe_s",
                "t_required_moderate_s",
                "unsafe_event_id",
                "unsafe_event_start_s",
                "warning_window_start_s",
                "warning_window_end_s",
            ]
        ].head(40).to_string(index=False)
    )
else:
    print("No moderate actionable warning rows found.")

print("\nActionable warning rows, conservative:")
warning_rows_c = gt[gt["warning_target_conservative"] == 1]
if len(warning_rows_c):
    print(
        warning_rows_c[
            [
                "frame_id",
                "frame_time_s",
                "gt_state",
                "lead_time_to_unsafe_s",
                "t_required_conservative_s",
                "unsafe_event_id",
                "unsafe_event_start_s",
                "warning_window_start_s",
                "warning_window_end_s",
            ]
        ].head(40).to_string(index=False)
    )
else:
    print("No conservative actionable warning rows found.")

print("\nFirst 20 rows:")
print(gt.head(20).to_string(index=False))

print("\nDone.")
