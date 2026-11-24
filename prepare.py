"""Prepare the pinned CrisisMMD damage benchmark. Usage: python prepare.py"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data"
REPO = "QCRI/CrisisMMD"
REV = "10a5626ba112ab9c50369c473a2062bcf205d708"
BASE = "https://huggingface.co"
LABELS = ["little_or_no_damage", "mild_damage", "severe_damage"]
SIZE = 160


def request(client, method, url, **kwargs):
    for attempt in range(5):
        try:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def chunks(sequence, size):
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def download(client, row, url):
    out = CACHE / "images" / (row["image_id"] + ".jpg")
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == row["sha256"]:
        return
    content = request(client, "GET", url).content
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("SHA-256 mismatch: " + row["image_id"])
    tmp = out.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(out)


def get_images(rows):
    # The Hub paths-info and Git LFS batch APIs resolve multiple public objects per request.
    with (
        httpx.Client(follow_redirects=True, timeout=60) as client,
        ThreadPoolExecutor(max_workers=12) as pool,
    ):
        for batch_no, group in enumerate(chunks(rows, 80)):
            paths = [r["image_path"] for r in group]
            metadata = request(
                client,
                "POST",
                f"{BASE}/api/datasets/{REPO}/paths-info/{REV}",
                data={"paths": paths},
            ).json()
            files = {x["path"]: x for x in metadata}
            for row in group:
                row["sha256"] = files[row["image_path"]]["lfs"]["oid"]
            objects = {item["lfs"]["oid"]: item["size"] for item in files.values()}
            batch = request(
                client,
                "POST",
                f"{BASE}/datasets/{REPO}.git/info/lfs/objects/batch",
                headers={
                    "Accept": "application/vnd.git-lfs+json",
                    "Content-Type": "application/vnd.git-lfs+json",
                },
                json={
                    "operation": "download",
                    "transfers": ["basic"],
                    "objects": [
                        {"oid": oid, "size": size} for oid, size in objects.items()
                    ],
                },
            ).json()
            urls = {
                x["oid"]: x["actions"]["download"]["href"] for x in batch["objects"]
            }
            futures = [
                pool.submit(download, client, row, urls[row["sha256"]]) for row in group
            ]
            for future in futures:
                future.result()
            print(
                f"Verified {min((batch_no + 1) * 80, len(rows))}/{len(rows)} images",
                flush=True,
            )


def main():
    (CACHE / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(follow_redirects=True, timeout=40) as client:
        for part in ["train", "dev", "test"]:
            records = request(
                client,
                "GET",
                f"{BASE}/datasets/{REPO}/resolve/{REV}/damage/{part}.json",
            ).json()
            for item in records:
                rows.append(
                    {
                        k: str(item[k])
                        for k in (
                            "event_name",
                            "image_id",
                            "tweet_id",
                            "image_path",
                            "label",
                        )
                    }
                    | {"split": part}
                )
    if len(rows) != 3526 or len({r["image_id"] for r in rows}) != 3526:
        raise ValueError("Unexpected benchmark manifest")
    get_images(rows)
    pixels = np.empty((len(rows), SIZE, SIZE, 3), np.uint8)
    for i, row in enumerate(rows):
        with Image.open(CACHE / "images" / (row["image_id"] + ".jpg")) as image:
            rgb = (
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            )
            pixels[i] = np.asarray(rgb)
        row["pixel_sha256"] = hashlib.sha256(pixels[i].tobytes()).hexdigest()

    # Connected components group all linked post IDs or identical model inputs.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("tweet_id", "sha256", "pixel_sha256"):
            previous = seen.setdefault((key, row[key]), i)
            parent[find(i)] = find(previous)
    priority = {"train": 0, "dev": 1, "test": 2}
    retained = {}
    for i, row in enumerate(rows):
        group = find(i)
        retained[group] = max(retained.get(group, -1), priority[row["split"]])
    for i, row in enumerate(rows):
        row["included"] = priority[row["split"]] == retained[find(i)]
    for key in ("tweet_id", "sha256", "pixel_sha256"):
        groups = [
            {r[key] for r in rows if r["included"] and r["split"] == part}
            for part in ("train", "dev", "test")
        ]
        assert not (
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )
    assert sum(r["included"] and r["split"] == "test" for r in rows) == 529
    np.save(CACHE / "pixels.npy", pixels)
    (CACHE / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    for part in ("train", "dev", "test"):
        print(
            part,
            [
                sum(
                    r["included"] and r["split"] == part and r["label"] == c
                    for r in rows
                )
                for c in LABELS
            ],
        )


if __name__ == "__main__":
    main()
