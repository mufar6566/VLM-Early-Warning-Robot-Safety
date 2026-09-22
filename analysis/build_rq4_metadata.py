from pathlib import Path
import argparse

import numpy as np
import pandas as pd


EXPECTED = {
    "huron": {"rows": 16862, "bags": 15},
    "pal": {"rows": 9249, "bags": 3},
    "crowdbot": {"rows": 8801, "bags": 7},
}

ALIASES = {
    "huron": {
        "bag_name": ["bag_name"],
        "frame_id": ["frame_id"],
        "frame_time_s": ["frame_time_s", "time_s"],
        "ground_truth": ["gt_state", "ground_truth", "ground_truth_label"],
        "speed_mps_raw": [
            "odom_speed_mps",
            "robot_speed_mps",
            "speed_mps",
            "speed_m_s",
        ],
    },
    "pal": {
        "bag_name": ["bag_name"],
        "frame_id": ["frame_id"],
        "frame_time_s": ["frame_time_s", "time_s"],
        "ground_truth": ["gt_state", "ground_truth", "ground_truth_label"],
        "speed_mps_raw": [
            "odom_speed_mps",
            "odom_speed_raw",
            "robot_speed_mps",
            "speed_mps",
        ],
    },
    "crowdbot": {
        "bag_name": ["bag_id", "bag_name"],
        "frame_id": ["frame_uid", "frame_id", "frame_index"],
        "frame_time_s": ["frame_time_s", "time_s"],
        "ground_truth": ["gt_state", "ground_truth", "ground_truth_label"],
        "speed_mps_raw": [
            "robot_speed_mps",
            "odom_speed_mps",
            "speed_mps",
        ],
    },
}

APPFR_MODES = {
    "huron": {"frame", "visual", "appfr"},
    "pal": {"visual", "frame", "appfr"},
    "crowdbot": {"appfr", "visual", "frame"},
}


def normalize_label(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(r"potentially_+unsafe", "potentially_unsafe", regex=True)
    )


def normalize_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"(?<=\d)\.0$", "", regex=True)
    )


def choose_column(columns, choices):
    return next((c for c in choices if c in columns), None)


def find_canonical_files(root, case_study):
    candidates = []

    for path in sorted(root.rglob("*.csv")):
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue

        selected = {}
        valid = True

        for target, choices in ALIASES[case_study].items():
            source = choose_column(columns, choices)
            if source is None:
                valid = False
                break
            selected[target] = source

        if not valid:
            continue

        required_meta = ["model_name", "input_mode", "history_frames"]
        if not all(c in columns for c in required_meta):
            continue

        try:
            first = pd.read_csv(
                path,
                nrows=1,
                usecols=required_meta,
                low_memory=False,
            )
        except Exception:
            continue

        if first.empty:
            continue

        model = str(first.iloc[0]["model_name"]).strip().lower()
        mode = str(first.iloc[0]["input_mode"]).strip().lower()

        try:
            history = int(float(first.iloc[0]["history_frames"]))
        except Exception:
            continue

        if "qwen" not in model:
            continue

        if history != 2:
            continue

        if mode not in APPFR_MODES[case_study]:
            continue

        candidates.append((path, selected))

    return candidates


def build_case(case_study, source_root, artifact_root):
    candidates = find_canonical_files(source_root, case_study)

    expected_bags = EXPECTED[case_study]["bags"]

    print(f"\n===== {case_study.upper()} =====")
    print("Canonical Qwen AppFr H02 files found:", len(candidates))

    for path, _ in candidates:
        print(" ", path.relative_to(source_root))

    assert len(candidates) == expected_bags, (
        f"{case_study}: expected {expected_bags} canonical bag files, "
        f"found {len(candidates)}"
    )

    frames = []

    for path, mapping in candidates:
        source_columns = list(dict.fromkeys(mapping.values()))

        raw = pd.read_csv(
            path,
            usecols=source_columns,
            low_memory=False,
        )

        raw = raw.rename(
            columns={source: target for target, source in mapping.items()}
        )

        raw["case_study"] = case_study
        raw["bag_name"] = raw["bag_name"].astype("string").str.strip()
        raw["global_bag_id"] = (
            case_study + "__" + raw["bag_name"].astype(str)
        )

        raw["frame_id"] = normalize_id(raw["frame_id"])
        raw["ground_truth"] = normalize_label(raw["ground_truth"])

        raw["frame_time_s"] = pd.to_numeric(
            raw["frame_time_s"],
            errors="coerce",
        )

        raw["speed_mps_raw"] = pd.to_numeric(
            raw["speed_mps_raw"],
            errors="coerce",
        ).abs()

        raw["source_file"] = str(path.relative_to(source_root))
        raw["speed_source_column"] = mapping["speed_mps_raw"]

        frames.append(raw)

    data = pd.concat(frames, ignore_index=True)

    data = data.sort_values(
        ["global_bag_id", "frame_time_s", "frame_id"]
    ).reset_index(drop=True)

    assert len(data) == EXPECTED[case_study]["rows"], (
        f"{case_study}: expected {EXPECTED[case_study]['rows']} rows, "
        f"found {len(data)}"
    )

    assert data["global_bag_id"].nunique() == expected_bags
    assert not data.duplicated(["global_bag_id", "frame_id"]).any()
    assert data["frame_time_s"].notna().all()

    assert set(data["ground_truth"]) <= {
        "safe",
        "potentially_unsafe",
        "unsafe",
    }

    missing_before = int(data["speed_mps_raw"].isna().sum())

    data["speed_mps"] = (
        data.groupby("global_bag_id", sort=False)["speed_mps_raw"]
        .transform(lambda x: x.interpolate(limit_direction="both"))
    )

    missing_after = int(data["speed_mps"].isna().sum())

    assert missing_after == 0, (
        f"{case_study}: {missing_after} speed values remain missing"
    )

    # Validate against the exact clean Qwen AppFr H02 prediction population.
    clean_file = (
        artifact_root
        / "case_studies"
        / case_study
        / "predictions"
        / "qwen"
        / "qwen_appfr_h02.csv"
    )

    clean = pd.read_csv(clean_file, low_memory=False)

    clean["bag_name"] = clean["bag_name"].astype("string").str.strip()
    clean["global_bag_id"] = (
        case_study + "__" + clean["bag_name"].astype(str)
    )
    clean["frame_id"] = normalize_id(clean["frame_id"])
    clean["ground_truth"] = normalize_label(clean["ground_truth"])
    clean["frame_time_s"] = pd.to_numeric(
        clean["frame_time_s"],
        errors="coerce",
    )

    left = data[
        [
            "global_bag_id",
            "frame_id",
            "frame_time_s",
            "ground_truth",
        ]
    ].sort_values(
        ["global_bag_id", "frame_id"]
    ).reset_index(drop=True)

    right = clean[
        [
            "global_bag_id",
            "frame_id",
            "frame_time_s",
            "ground_truth",
        ]
    ].sort_values(
        ["global_bag_id", "frame_id"]
    ).reset_index(drop=True)

    assert len(left) == len(right)
    assert left[
        ["global_bag_id", "frame_id", "ground_truth"]
    ].equals(
        right[["global_bag_id", "frame_id", "ground_truth"]]
    )

    assert np.allclose(
        left["frame_time_s"],
        right["frame_time_s"],
        atol=1e-6,
        rtol=0,
    )

    output_dir = (
        artifact_root
        / "case_studies"
        / case_study
        / "metadata"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "rq4_frame_metadata.csv"

    final_columns = [
        "case_study",
        "bag_name",
        "global_bag_id",
        "frame_id",
        "frame_time_s",
        "ground_truth",
        "speed_mps_raw",
        "speed_mps",
        "speed_source_column",
        "source_file",
    ]

    data[final_columns].to_csv(output, index=False)

    print("Rows:", len(data))
    print("Bags:", data["global_bag_id"].nunique())
    print("Missing speed before interpolation:", missing_before)
    print("Missing speed after interpolation:", missing_after)
    print("Saved:", output)

    return {
        "case_study": case_study,
        "rows": len(data),
        "bags": data["global_bag_id"].nunique(),
        "missing_speed_before": missing_before,
        "missing_speed_after": missing_after,
        "output": str(output.relative_to(artifact_root)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-analysis-root",
        required=True,
        type=Path,
        help="Original VLM_EW_Robot_Safety_DT/Analysis directory.",
    )

    args = parser.parse_args()

    source = args.source_analysis_root.expanduser().resolve()
    artifact = Path(__file__).resolve().parents[1]

    roots = {
        "huron": source / "HuRoN",
        "pal": source / "PAL" / "Pal_Pred",
        "crowdbot": source / "CrowdBot",
    }

    for case_study, root in roots.items():
        assert root.is_dir(), f"Missing source folder: {root}"

    rows = [
        build_case(case, root, artifact)
        for case, root in roots.items()
    ]

    summary = pd.DataFrame(rows)

    summary_path = artifact / "analysis" / "rq4_metadata_summary.csv"
    summary.to_csv(summary_path, index=False)

    assert summary["rows"].sum() == 34912
    assert summary["bags"].sum() == 25
    assert summary["missing_speed_after"].eq(0).all()

    print("\nFINAL RQ4 METADATA VERIFICATION PASSED")
    print(summary.to_string(index=False))
    print("\nTotal rows:", summary["rows"].sum())
    print("Total bags:", summary["bags"].sum())


if __name__ == "__main__":
    main()
