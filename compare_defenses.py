"""Summarize matched image predictions for the three fixed classifiers (standard library only)."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("standard", "noise", "fgsm")
LABELS = ("little_or_none", "mild", "severe")
CONDITIONS = ["clean"] + [f"{method}_{eps}" for eps in (1, 2, 4, 8) for method in ("sign_noise", "fgsm", "bim")]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    folders = {arm: ROOT / "defense" / arm for arm in ARMS}
    folders["standard"] = ROOT / "results"
    indexes = {}
    metadata = {}
    for arm, folder in folders.items():
        rows = read_csv(folder / "predictions.csv")
        indexes[arm] = {r["image_id"]: r for r in rows}
        assert len(rows) == len(indexes[arm]) == 529
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        hist = json.loads((folder / "training.json").read_text(encoding="utf-8"))
        best = max(hist, key=lambda row: row["macro_f1"])
        assert best["epoch"] == summary["selected_epoch"] and len(hist) == 10
        metadata[arm] = dict(epoch=best["epoch"], validation_macro_f1=best["macro_f1"], train_seconds=sum(x["seconds"] for x in hist) if "seconds" in best else None)
    reference = indexes["standard"]
    for arm in ARMS:
        assert set(indexes[arm]) == set(reference)
        assert all(all(indexes[arm][i][k] == r[k] for k in ("true","tweet_id","event")) for i,r in reference.items())
    severe = [i for i, r in reference.items() if r["true"] == "2"]
    common = [i for i in severe if all(indexes[arm][i]["clean"] == "2" for arm in ARMS)]
    assert len(severe) == 332 and common
    out = ROOT / "comparison"
    out.mkdir(exist_ok=True)
    metrics, pairs = [], []
    for arm in ARMS:
        idx = indexes[arm]
        for cond in CONDITIONS:
            cm = Counter((int(r["true"]), int(r[cond])) for r in idx.values())
            f1, recall = [], []
            for k in range(3):
                total = sum(cm[k,j] for j in range(3))
                predicted = sum(cm[j,k] for j in range(3))
                f1.append(2*cm[k,k]/(total+predicted))
                recall.append(cm[k,k]/total)
            common_counts = Counter(idx[i][cond] for i in common)
            correct_on_severe = cm[2,2]
            added = sum(idx[i][cond] == "2" and reference[i][cond] != "2" for i in severe)
            lost = sum(idx[i][cond] != "2" and reference[i][cond] == "2" for i in severe)
            assert correct_on_severe == sum(reference[i][cond] == "2" for i in severe) + added - lost
            metrics.append(dict(model=arm, condition=cond, total=529, accuracy=sum(cm[k,k] for k in range(3))/529, macro_f1=sum(f1)/3,
                little_none_recall=recall[0], mild_recall=recall[1], severe_recall=recall[2], severe_correct=correct_on_severe, severe_total=len(severe),
                common_severe_n=len(common), common_stayed=common_counts["2"], common_to_mild=common_counts["1"], common_to_little_none=common_counts["0"],
                gained_vs_standard=added, lost_vs_standard=lost))
        for i in reference:
            pairs.append(dict(model=arm,image_id=i,truth=idx[i]["true"],common_clean_severe=int(i in common), **{c:idx[i][c] for c in CONDITIONS}))
    write_csv(out/"metrics.csv", metrics)
    write_csv(out/"paired_predictions.csv", pairs)
    (out/"training_summary.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    for row in metrics:
        if row["condition"] in ("clean","sign_noise_4","fgsm_4","bim_4"):
            print(row["model"],row["condition"],row["severe_correct"],f'{row["macro_f1"]:.4f}',row["common_stayed"],row["common_to_mild"],row["common_to_little_none"])
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
