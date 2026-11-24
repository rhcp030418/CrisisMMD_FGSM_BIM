"""Fixed, image-only CrisisMMD benchmark. Run prepare.py before this script."""

import argparse
import csv
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import ResNet18_Weights, resnet18
from prepare import CACHE, LABELS, ROOT, REV

OUT = ROOT / "results"
SEED = 20261123
BATCH = 32
EPOCHS = 10
BUDGETS = (1, 2, 4, 8)
BIM_STEPS = 5


class DamageNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(LABELS))
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, rgb):
        return self.backbone((rgb - self.mean) / self.std)


def scores(y, pred):
    k = len(LABELS)
    cm = np.bincount(k * np.asarray(y) + np.asarray(pred), minlength=k * k).reshape(
        k, k
    )
    tp = cm.diagonal()
    f1 = 2 * tp / np.maximum(1, cm.sum(0) + cm.sum(1))
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "f1": f1.tolist(),
        "recall": (tp / np.maximum(1, cm.sum(1))).tolist(),
        "confusion": cm.tolist(),
    }


def inputs(pixels, ix):
    return pixels[ix].float().div_(255)


@torch.no_grad()
def predict(model, pixels, indices):
    model.eval()
    return (
        torch.cat([model(inputs(pixels, ix)).argmax(1) for ix in indices.split(BATCH)])
        .cpu()
        .numpy()
    )


def fit(model, pixels, y, subsets):
    train, dev = subsets["train"], subsets["dev"]
    weights = torch.bincount(y[train], minlength=len(LABELS)).float().rsqrt()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=2e-5
    )
    best = -1
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = train[torch.randperm(len(train), device=train.device)]
        for ix in order.split(BATCH):
            rgb = inputs(pixels, ix)
            flip = torch.rand(len(ix), device=rgb.device) < 0.5
            rgb[flip] = rgb[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(rgb), y[ix], weight=weights)
            loss.backward()
            optimizer.step()
        val = scores(y[dev].cpu().numpy(), predict(model, pixels, dev))
        history.append(
            {"epoch": epoch, "macro_f1": val["macro_f1"], "accuracy": val["accuracy"]}
        )
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(
                {"epoch": epoch, "weights": model.state_dict()}, OUT / "model.pt"
            )
        scheduler.step()
        print(
            f"Epoch {epoch}/{EPOCHS} validation macro-F1 {val['macro_f1']:.4f}",
            flush=True,
        )
    save_json("training.json", history)


def gradient_sign(model, rgb, labels):
    current = rgb.detach().requires_grad_(True)
    loss = F.cross_entropy(model(current), labels)
    return torch.autograd.grad(loss, current)[0].detach().sign()


def fgsm(model, original, labels, eps):
    return (original + eps * gradient_sign(model, original, labels)).clamp(0, 1)


def bim(model, original, labels, eps):
    current = original
    lower, upper = (original - eps).clamp(0, 1), (original + eps).clamp(0, 1)
    for _ in range(BIM_STEPS):
        current = current + (eps / BIM_STEPS) * gradient_sign(model, current, labels)
        current = torch.maximum(lower, torch.minimum(upper, current)).detach()
    return current


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, rgb, y, method, budget, repeat):
    device = rgb.device
    eps = budget / 255
    rng = torch.Generator(device=device).manual_seed(SEED)
    results = []
    max_delta = torch.zeros((), device=device)
    low = torch.ones((), device=device)
    high = torch.zeros((), device=device)
    duration = 0
    for original, labels in zip(rgb.split(BATCH), y.split(BATCH)):
        synchronize(device)
        start = time.perf_counter()
        if method == "sign_noise":
            direction = (
                torch.randint(0, 2, original.shape, device=device, generator=rng).to(
                    original.dtype
                )
                * 2
                - 1
            )
            changed = (original + eps * direction).clamp(0, 1)
        elif method == "fgsm":
            changed = fgsm(model, original, labels, eps)
        else:
            changed = bim(model, original, labels, eps)
        with torch.no_grad():
            pred = model(changed).argmax(1)
        synchronize(device)
        duration += time.perf_counter() - start
        if repeat == 0:
            max_delta = torch.maximum(max_delta, (changed - original).abs().max())
            low = torch.minimum(low, changed.min())
            high = torch.maximum(high, changed.max())
        results.append(pred)
    assert float(max_delta) <= eps + 2e-6 and float(low) >= 0 and float(high) <= 1
    return (
        torch.cat(results).cpu().numpy(),
        duration * 1000 / len(rgb),
        float(max_delta),
    )


def save_json(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(reuse):
    OUT.mkdir(exist_ok=True)
    torch.hub.set_dir(str(CACHE / "torch" / "hub"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    pixels = (
        torch.from_numpy(np.load(CACHE / "pixels.npy"))
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    y = torch.tensor([LABELS.index(r["label"]) for r in rows], device=device)
    subsets = {
        s: torch.tensor(
            [i for i, r in enumerate(rows) if r["included"] and r["split"] == s],
            device=device,
        )
        for s in ("train", "dev", "test")
    }
    model = DamageNet(pretrained=not reuse).to(device)
    if not reuse:
        fit(model, pixels, y, subsets)
    ckpt = torch.load(OUT / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["weights"])
    model.eval().requires_grad_(False)
    test = subsets["test"]
    rgb = inputs(pixels, test)
    labels = y[test]
    actual = labels.cpu().numpy()
    clean = predict(model, pixels, test)
    eligible = (actual == 2) & (clean == 2)
    assert eligible.any()
    ids = test.cpu().tolist()
    detailed = [
        {
            "image_id": rows[i]["image_id"],
            "tweet_id": rows[i]["tweet_id"],
            "event": rows[i]["event_name"],
            "true": int(a),
            "clean": int(p),
        }
        for i, a, p in zip(ids, actual, clean)
    ]
    summary = []
    # Warm up kernels, including the smaller final batch, before measuring conditions.
    for n in [BATCH, len(test) % BATCH or BATCH]:
        for f in [fgsm, bim]:
            with torch.enable_grad():
                _ = f(model, rgb[:n], labels[:n], 1 / 255)
    for budget in BUDGETS:
        for method in ("sign_noise", "fgsm", "bim"):
            times = []
            for repeat in range(3):
                pred, ms, delta = measure(model, rgb, labels, method, budget, repeat)
                times.append(ms)
                if repeat == 0:
                    first = pred.copy()
                    actual_delta = delta
                else:
                    assert np.array_equal(first, pred)
            for row, value in zip(detailed, first):
                row[f"{method}_{budget}"] = int(value)
            none, mild, severe = np.bincount(first[eligible], minlength=3).tolist()
            result = {
                "eps_over_255": budget,
                "method": method,
                "eligible": int(eligible.sum()),
                "to_little_none": none,
                "to_mild": mild,
                "stayed_severe": severe,
                "downgraded": none + mild,
                "downgraded_pct": 100 * (none + mild) / eligible.sum(),
                "ms_per_image": statistics.median(times),
                "measured_linf": actual_delta,
                "all_test_macro_f1": scores(actual, first)["macro_f1"],
            }
            summary.append(result)
            print(
                f"{budget}/255 {method} severe -> mild {mild}, little/no {none}, stayed {severe}",
                flush=True,
            )
    counts = {
        s: np.bincount(y[ix].cpu().numpy(), minlength=3).tolist()
        for s, ix in subsets.items()
    }
    save_csv("predictions.csv", detailed)
    save_csv("conditions.csv", summary)
    info = {
        "dataset_revision": REV,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "python": sys.version.split()[0],
        "system": platform.system(),
        "selected_epoch": ckpt["epoch"],
        "counts": counts,
        "clean": scores(actual, clean),
        "eligible": int(eligible.sum()),
        "budget_numerators": BUDGETS,
        "bim_steps": BIM_STEPS,
        "batch": BATCH,
    }
    save_json("summary.json", info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Evaluate the supplied fitted checkpoint.",
    )
    main(parser.parse_args().reuse_model)
