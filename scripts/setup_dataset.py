import shutil
from pathlib import Path

def main() -> None:
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"
    cache_dir = data_dir / "cache"

    for d in [raw_dir, processed_dir, splits_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (raw_dir / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / split).mkdir(parents=True, exist_ok=True)

    mappings = {
        data_dir / "train_images" / "train_images": raw_dir / "train",
        data_dir / "val_images" / "val_images": raw_dir / "val",
        data_dir / "test_images" / "test_images": raw_dir / "test"
    }

    for src, dst in mappings.items():
        if src.exists():
            print(f"Moving images from {src} to {dst}...")
            for f in src.glob("*.png"):
                shutil.move(str(f), str(dst / f.name))

    csv_mappings = {
        data_dir / "train.csv": splits_dir / "train.csv",
        data_dir / "valid.csv": splits_dir / "val.csv",
        data_dir / "test.csv": splits_dir / "test.csv"
    }

    for src, dst in csv_mappings.items():
        if src.exists():
            print(f"Moving CSV from {src} to {dst}...")
            shutil.move(str(src), str(dst))

    for old_dir in ["train_images", "val_images", "test_images"]:
        path = data_dir / old_dir
        if path.exists():
            print(f"Cleaning up {path}...")
            shutil.rmtree(path)

    print("Dataset reorganization complete.")

if __name__ == "__main__":
    main()
