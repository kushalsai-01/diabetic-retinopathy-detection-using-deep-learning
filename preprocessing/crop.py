import cv2
import numpy as np

def remove_black_borders(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    if w == 0 or h == 0:
        return image
    return image[y:y+h, x:x+w]

def circular_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    cv2.circle(mask, center, radius, 255, -1)
    
    return cv2.bitwise_and(image, image, mask=mask)

def resize_to_square(image: np.ndarray, size: int = 512) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
