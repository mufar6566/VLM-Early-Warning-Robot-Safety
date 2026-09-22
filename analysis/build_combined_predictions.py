from pathlib import Path

import numpy as np
import pandas as pd


# Repository root
HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE.parent, *HERE.parents]
    if (p / "case_studies").is_dir()
)

MODELS = ["qwen", "llava", "internvl"]
APPROACHES = ["appfr", "appod"]
HISTORIES = range(2, 33, 2)

EXPECTED_ROWS = {
    "huron": 16862,
    "pal": 9249,
    "crowdbot": 8801,
}

EXPECTED_BAGS = {
    "huron": 15,
    "pal": 3,
    "crowdbot": 7,
}

EXPECTED_COMBINED_ROWS = sum(EXPECTED_ROWS.values())
EXPECTED_COMBINED_BAGS = sum(EXPECTED_BAGS.values())

CASE_ROOTS = {
    "huron": ROOT / "case_studies" / "huron" / "predictions",
    "pal": ROOT / "case_studies" / "pal" / "predictions",
    "crowdbot": ROOT / "case_studies" / "crowdbot" / "predictions",
}

OUTPUT_ROOT = ROOT / "analysis" / "combined"
CLEAN_ROOT = OUTPUT_ROOT / "clean"
CLEAN_ROOT.mkdir(parents=True, exist_ok=True)

summary_rows = []

for model in MODELS:
    model_output = CLEAN_ROOT / model
    model_output.mkdir(parents=True, exist_ok=True)

    for approach in APPROACHES:
        for history in HISTORIES:

            parts = []

            for case_study, source_root in CASE_ROOTS.items():
                source = (
                    source_root
                    / model
                    / f"{model}_{approach}_h{history:02d}.csv"
                )

                assert source.is_file(), f"Missing prediction file: {source}"

                df = pd.read_csv(source, low_memory=False)

                assert len(df) == EXPECTED_ROWS[case_study], (
                    f"{source}: expected {EXPECTED_ROWS[case_study]} rows, "
                    f"found {len(df)}"
                )

                assert df["bag_name"].nunique() == EXPECTED_BAGS[case_study], (
                    f"{source}: incorrect bag count"
                )

                assert not df.duplicated(
                    ["bag_name", "frame_id"]
                ).any(), f"{source}: duplicate frames"

                df.insert(0, "case_study", case_study)
                df.insert(
                    2,
                    "global_bag_id",
                    case_study + "__" + df["bag_name"].astype(str),
                )

                df["model"] = model
                df["approach"] = approach
                df["history_frames"] = history

                parts.append(df)

            combined = pd.concat(
                parts,
                ignore_index=True,
                sort=False,
            )

            assert len(combined) == EXPECTED_COMBINED_ROWS
            assert combined["global_bag_id"].nunique() == EXPECTED_COMBINED_BAGS

            assert not combined.duplicated(
                ["global_bag_id", "frame_id"]
            ).any()

            assert combined["predicted_label"].isin(
                ["safe", "potentially_unsafe", "unsafe"]
            ).all()

            probabilities = combined[
                [
                    "prob_safe",
                    "prob_potentially_unsafe",
                    "prob_unsafe",
                ]
            ].to_numpy(float)

            assert np.isfinite(probabilities).all()

            assert np.isclose(
                probabilities.sum(axis=1),
                1.0,
                atol=1e-5,
            ).all()

            output = (
                model_output
                / f"{model}_{approach}_h{history:02d}.csv"
            )

            combined = combined.sort_values(
                ["case_study", "bag_name", "frame_id"]
            ).reset_index(drop=True)

            combined.to_csv(output, index=False)

            summary_rows.append(
                {
                    "model": model,
                    "approach": approach,
                    "history": f"H{history:02d}",
                    "rows": len(combined),
                    "bags": combined["global_bag_id"].nunique(),
                    "huron_rows": EXPECTED_ROWS["huron"],
                    "pal_rows": EXPECTED_ROWS["pal"],
                    "crowdbot_rows": EXPECTED_ROWS["crowdbot"],
                    "output_file": str(output.relative_to(ROOT)),
                }
            )

            print(
                f"Saved {model} {approach} H{history:02d}: "
                f"{len(combined):,} rows"
            )

summary = pd.DataFrame(summary_rows)

assert len(summary) == 96
assert summary["rows"].eq(EXPECTED_COMBINED_ROWS).all()
assert summary["bags"].eq(EXPECTED_COMBINED_BAGS).all()

summary.to_csv(
    OUTPUT_ROOT / "merge_summary.csv",
    index=False,
)

print()
print("FINAL VERIFICATION PASSED")
print(f"Combined files: {len(summary)}")
print(f"Rows per configuration: {EXPECTED_COMBINED_ROWS:,}")
print(f"Bags per configuration: {EXPECTED_COMBINED_BAGS}")
print(f"Output: {CLEAN_ROOT}")
