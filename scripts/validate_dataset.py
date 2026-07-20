import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd

def validate_split(name: str, csv_path: Path, img_dir: Path) -> bool:
    print(f"Validating {name} split...")
    if not csv_path.exists():
        print(f"ERROR: CSV path does not exist: {csv_path}")
        return False
    if not img_dir.exists():
        print(f"ERROR: Image directory does not exist: {img_dir}")
        return False

    df = pd.read_csv(csv_path)
    required_cols = {"id_code", "diagnosis"}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: CSV missing columns. Found {df.columns}. Required: {required_cols}")
        return False

    missing_files = []
    invalid_labels = []
    
    for idx, row in df.iterrows():
        id_code = row["id_code"]
        diagnosis = row["diagnosis"]
        img_path = img_dir / f"{id_code}.png"
        
        if not img_path.exists():
            missing_files.append(id_code)
        
        if not (0 <= diagnosis <= 4):
            invalid_labels.append((id_code, diagnosis))

    if missing_files:
        print(f"ERROR: {len(missing_files)} missing image files. Sample: {missing_files[:5]}")
    if invalid_labels:
        print(f"ERROR: {len(invalid_labels)} invalid labels. Sample: {invalid_labels[:5]}")

    if not missing_files and not invalid_labels:
        print(f"SUCCESS: {name} split validated successfully. Count: {len(df)} samples.")
        return True
    return False

def main() -> None:
    data_dir = Path("data")
    splits_dir = data_dir / "splits"
    raw_dir = data_dir / "raw"

    splits = [
        ("Train", splits_dir / "train.csv", raw_dir / "train"),
        ("Validation", splits_dir / "val.csv", raw_dir / "val"),
        ("Test", splits_dir / "test.csv", raw_dir / "test"),
    ]

    all_passed = True
    for name, csv_p, img_d in splits:
        if not validate_split(name, csv_p, img_d):
            all_passed = False

    if all_passed:
        print("\nAll dataset files validated successfully.")
    else:
        print("\nDataset validation FAILED.")

if __name__ == "__main__":
    main()
