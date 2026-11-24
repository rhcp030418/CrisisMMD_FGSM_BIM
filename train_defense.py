"""Train and evaluate the two defense models with the fixed CrisisMMD benchmark."""
import argparse
import json
import random
import time
from pathlib import Path

import experiment as bench
import numpy as np
import torch
import verify
from torch.nn import functional as F
from prepare import CACHE, LABELS

ROOT = Path(__file__).resolve().parent
TRAIN_EPS = 4 / 255


def train(arm, out):
    random.seed(bench.SEED)
    np.random.seed(bench.SEED)
    torch.manual_seed(bench.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.hub.set_dir(str(CACHE / "torch" / "hub"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_rng = torch.Generator(device=device).manual_seed(bench.SEED)
    rows = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    pixels = torch.from_numpy(np.load(CACHE / "pixels.npy")).permute(0, 3, 1, 2).contiguous().to(device)
    labels = torch.tensor([LABELS.index(r["label"]) for r in rows], device=device)
    def split_ids(split):
        return torch.tensor([i for i, r in enumerate(rows) if r["included"] and r["split"] == split], device=device)
    train_ids, dev_ids = split_ids("train"), split_ids("dev")
    assert (len(train_ids), len(dev_ids)) == (2261, 508)
    model = bench.DamageNet(pretrained=True).to(device)
    weights = torch.bincount(labels[train_ids], minlength=len(LABELS)).float().rsqrt()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=bench.EPOCHS, eta_min=2e-5)
    best = -1
    history = []
    for epoch in range(1, bench.EPOCHS + 1):
        start = time.perf_counter()
        model.train()
        order = train_ids[torch.randperm(len(train_ids), device=device)]
        for ids in order.split(bench.BATCH):
            rgb = bench.inputs(pixels, ids)
            flip = torch.rand(len(ids), device=device) < 0.5
            rgb[flip] = rgb[flip].flip(-1)
            y = labels[ids]
            if arm == "noise":
                signs = torch.randint(0, 2, rgb.shape, device=device, generator=noise_rng).to(rgb.dtype).mul_(2).sub_(1)
                augmented = (rgb + TRAIN_EPS * signs).clamp_(0, 1)
            else:
                # Match the classroom exercise: generate the training view in inference mode.
                model.eval()
                augmented = bench.fgsm(model, rgb, y, TRAIN_EPS).detach()
                model.train()
            optimizer.zero_grad(set_to_none=True)
            clean_loss = F.cross_entropy(model(rgb), y, weight=weights)
            augmented_loss = F.cross_entropy(model(augmented), y, weight=weights)
            ((clean_loss + augmented_loss) / 2).backward()
            optimizer.step()
        metrics = bench.scores(labels[dev_ids].cpu().numpy(), bench.predict(model, pixels, dev_ids))
        elapsed = time.perf_counter() - start
        history.append(dict(epoch=epoch, macro_f1=metrics["macro_f1"], accuracy=metrics["accuracy"], seconds=elapsed))
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"epoch": epoch, "weights": model.state_dict()}, out / "model.pt")
        scheduler.step()
        print(f"{arm}: {epoch}/{bench.EPOCHS}, validation macro-F1={metrics['macro_f1']:.4f}, {elapsed:.1f}s", flush=True)
    (out / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("noise", "fgsm"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "defense" / args.model
    out.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        train(args.model, out)
    bench.OUT = out
    bench.main(True)
    verify.ROOT = ROOT
    verify.RESULTS = out
    verify.main()


if __name__ == "__main__":
    main()
