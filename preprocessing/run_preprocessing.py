import sys
from pathlib import Path

# Add project root to sys.path dynamically
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import argparse
from preprocessing.pipeline import preprocess_batch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch preprocess APTOS dataset splits.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Split to preprocess (default: all)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image square resolution (default: 512)"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        in_dir = raw_dir / split
        out_dir = processed_dir / split
        
        if not in_dir.exists():
            print(f"Directory {in_dir} does not exist. Skipping.")
            continue

        print(f"\nProcessing split: {split}")
        summary = preprocess_batch(str(in_dir), str(out_dir), args.size)
        print(f"Split {split} summary: {summary}")

if __name__ == "__main__":
    main()
