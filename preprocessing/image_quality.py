import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class QualityResult:
    passed: bool
    reason: Optional[str] = None

def check_image_loadable(image_path: str) -> QualityResult:
    image = cv2.imread(image_path)
    if image is None:
        return QualityResult(passed=False, reason="Corrupt or unloadable image")
    return QualityResult(passed=True)

def check_minimum_resolution(
    image: np.ndarray,
    min_width: int = 256,
    min_height: int = 256,
) -> QualityResult:
    h, w = image.shape[:2]
    if w < min_width or h < min_height:
        return QualityResult(passed=False, reason=f"Resolution too low: {w}x{h}")
    return QualityResult(passed=True)

def check_blur(image: np.ndarray, threshold: float = 100.0) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < threshold:
        return QualityResult(passed=False, reason=f"Image is blurry (variance: {variance:.2f})")
    return QualityResult(passed=True)

def check_fundus_coverage(
    image: np.ndarray,
    min_coverage: float = 0.3,
) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coverage = np.count_nonzero(thresh) / thresh.size
    if coverage < min_coverage:
        return QualityResult(passed=False, reason=f"Retinal disc coverage too low: {coverage:.2f}")
    return QualityResult(passed=True)

def run_all_checks(image_path: str) -> QualityResult:
    res = check_image_loadable(image_path)
    if not res.passed:
        return res

    image = cv2.imread(image_path)

    res = check_minimum_resolution(image)
    if not res.passed:
        return res

    res = check_fundus_coverage(image)
    if not res.passed:
        return res

    return QualityResult(passed=True)
