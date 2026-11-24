"""Recount every transition from the per-image predictions without PyTorch."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = ["little_or_no_damage", "mild_damage", "severe_damage"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = read_csv(RESULTS / "predictions.csv")
    conditions = read_csv(RESULTS / "conditions.csv")
    summary = read_json(RESULTS / "summary.json")
    manifest_path = ROOT / "split_manifest.json"
    if not manifest_path.exists():
        manifest_path = ROOT / "data" / "manifest.json"
    manifest = read_json(manifest_path)
    assert len(rows) == len({r["image_id"] for r in rows}) == 529
    manifest_test = {
        r["image_id"]: r for r in manifest if r["split"] == "test" and r["included"]
    }
    assert set(manifest_test) == {r["image_id"] for r in rows}
    assert len(manifest) == 3526
    for row in rows:
        original = manifest_test[row["image_id"]]
        assert LABELS[int(row["true"])] == original["label"]
        assert row["tweet_id"] == original["tweet_id"]
    retained_counts = {}
    original_counts = {}
    for part in ["train", "dev", "test"]:
        for included, record in [(False, original_counts), (True, retained_counts)]:
            group = [
                r
                for r in manifest
                if r["split"] == part and (r["included"] or not included)
            ]
            record[part] = [sum(r["label"] == label for r in group) for label in LABELS]
    assert retained_counts == summary["counts"]
    overlap = {}
    for key in ["tweet_id", "image_id", "sha256", "pixel_sha256"]:
        orig = [
            {r[key] for r in manifest if r["split"] == part}
            for part in ["train", "dev", "test"]
        ]
        kept = [
            {r[key] for r in manifest if r["split"] == part and r["included"]}
            for part in ["train", "dev", "test"]
        ]
        overlap[key] = {"original_train_test": len(orig[0] & orig[2])}
        assert not (kept[0] & kept[1] or kept[0] & kept[2] or kept[1] & kept[2])
    cm = [
        [
            sum(int(r["true"]) == i and int(r["clean"]) == j for r in rows)
            for j in range(3)
        ]
        for i in range(3)
    ]
    assert cm == summary["clean"]["confusion"]
    f1 = [
        2 * cm[i][i] / max(1, sum(cm[i]) + sum(row[i] for row in cm)) for i in range(3)
    ]
    assert abs(sum(f1) / 3 - summary["clean"]["macro_f1"]) < 1e-8
    cohort = [r for r in rows if r["true"] == r["clean"] == "2"]
    assert len(cohort) == summary["eligible"] == cm[2][2]
    assert (
        len(conditions)
        == len({(r["eps_over_255"], r["method"]) for r in conditions})
        == 12
    )
    for c in conditions:
        key = f"{c['method']}_{c['eps_over_255']}"
        counts = Counter(int(r[key]) for r in cohort)
        assert [counts[n] for n in range(3)] == [
            int(c[k]) for k in ("to_little_none", "to_mild", "stayed_severe")
        ]
        assert int(c["eligible"]) == len(cohort)
        assert int(c["downgraded"]) == counts[0] + counts[1]
        assert (
            abs(
                float(c["downgraded_pct"]) - 100 * (counts[0] + counts[1]) / len(cohort)
            )
            < 1e-8
        )
        assert (
            float(c["ms_per_image"]) > 0
            and float(c["measured_linf"]) <= int(c["eps_over_255"]) / 255 + 2e-6
        )
    pairs = Counter((r["fgsm_4"] != "2", r["bim_4"] != "2") for r in cohort)
    audit = {
        "split_original": original_counts,
        "split_used": retained_counts,
        "overlap": overlap,
        "both_downgraded": pairs[True, True],
        "fgsm_only": pairs[True, False],
        "bim_only": pairs[False, True],
        "neither": pairs[False, False],
        "checkpoint_sha256": hashlib.file_digest(
            (RESULTS / "model.pt").open("rb"), "sha256"
        ).hexdigest(),
        "verified_test_images": len(rows),
        "verified_conditions": len(conditions),
    }
    print(json.dumps(audit, indent=2))
    (RESULTS / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
