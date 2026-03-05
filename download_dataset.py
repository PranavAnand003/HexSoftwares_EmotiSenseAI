"""
Download the dair-ai/emotion dataset from HuggingFace and save as
train.txt / val.txt / test.txt in the ./data directory.

Run ONCE before training:
    python download_dataset.py
"""

import os

def download_with_datasets():
    from datasets import load_dataset

    print("📥 Downloading dair-ai/emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")

    os.makedirs("data", exist_ok=True)

    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

    for split in ["train", "validation", "test"]:
        split_name = "val" if split == "validation" else split
        out_path = os.path.join("data", f"{split_name}.txt")
        data = dataset[split]
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                text  = str(item["text"]).replace(";", ",")
                label = label_map[item["label"]]
                f.write(f"{text};{label}\n")
        print(f"  ✅ Saved {len(data):,} examples → {out_path}")

    print("\n🎉 Dataset ready! Run: python train_model.py")


def download_with_requests():
    """Fallback: download raw files directly from HuggingFace CDN."""
    import urllib.request

    BASE = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/data/"
    FILES = {
        "train": "train.jsonl",
        "val":   "validation.jsonl",
        "test":  "test.jsonl",
    }

    os.makedirs("data", exist_ok=True)

    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

    import json
    for split, fname in FILES.items():
        url = BASE + fname
        local = os.path.join("data", fname)
        print(f"📥 Downloading {url} ...")
        urllib.request.urlretrieve(url, local)
        out_path = os.path.join("data", f"{split}.txt")
        with open(local, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                item = json.loads(line.strip())
                text  = str(item["text"]).replace(";", ",")
                label = label_map[item["label"]]
                fout.write(f"{text};{label}\n")
        os.remove(local)
        print(f"  ✅ Saved → {out_path}")

    print("\n🎉 Dataset ready! Run: python train_model.py")


if __name__ == "__main__":
    try:
        download_with_datasets()
    except ImportError:
        print("⚠️  `datasets` library not found, trying direct download...")
        download_with_requests()
    except Exception as e:
        print(f"⚠️  datasets library failed ({e}), trying direct download...")
        download_with_requests()
