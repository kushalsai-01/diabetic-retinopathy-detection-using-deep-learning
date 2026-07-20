import cv2
import numpy as np

def ben_graham(
    image: np.ndarray,
    sigmaX: int = 10,
    alpha: float = 4.0,
    beta: float = -4.0,
    gamma: float = 128.0,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(image, alpha, blurred, beta, gamma)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
