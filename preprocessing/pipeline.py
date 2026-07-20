import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor

from preprocessing.image_quality import run_all_checks, QualityResult
from preprocessing.crop import remove_black_borders, circular_crop, resize_to_square
from preprocessing.enhance import ben_graham, apply_clahe

def preprocess_image(
    image_path: str,
    output_size: int = 512,
) -> tuple[np.ndarray | None, QualityResult]:
    quality = run_all_checks(image_path)
    if not quality.passed:
        return None, quality

    image = cv2.imread(image_path)
    if image is None:
        return None, QualityResult(passed=False, reason="Could not read file")

    image = remove_black_borders(image)
    image = circular_crop(image)
    image = resize_to_square(image, output_size)
    image = ben_graham(image)
    image = apply_clahe(image)

    return image, quality

def preprocess_batch(
    input_dir: str,
    output_dir: str,
    output_size: int = 512,
) -> dict[str, int]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
    
    processed = 0
    failed = 0
    skipped = 0
    failures = {}

    def process_file(f: Path) -> tuple[str, str, str]:
        out_file = output_path / f"{f.stem}.png"
        if out_file.exists():
            return "skipped", f.name, None
        try:
            image, quality = preprocess_image(str(f), output_size)
            if quality.passed and image is not None:
                cv2.imwrite(str(out_file), image)
                return "processed", f.name, None
            else:
                return "failed", f.name, quality.reason or "Unknown quality error"
        except Exception as e:
            return "failed", f.name, str(e)

    # Parallelize image preprocessing using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files), desc=f"Preprocessing {input_path.name}"))

    for status, name, reason in results:
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            failures[name] = reason

    stats = {
        "input_dir": input_dir,
        "total_files": len(files),
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "failures": failures,
        "timestamp": time.time()
    }
    
    stats_file = Path("data/cache") / f"stats_{input_path.name}.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as sf:
        json.dump(stats, sf, indent=4)

    return {"processed": processed, "failed": failed, "skipped": skipped}
