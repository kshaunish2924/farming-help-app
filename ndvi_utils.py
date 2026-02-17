# ndvi_utils.py
import numpy as np
from PIL import Image

def load_rgb_image(image_path: str, max_size: int = 800) -> np.ndarray:
    """
    Loads an image as RGB float32 array in range [0,1].
    Optionally resizes large images to keep processing fast/safe.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize to keep it fast + memory safe
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)))

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def ndvi_like_index(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Simulated NDVI using only RGB:
    NDVI-like = (G - R) / (G + R + eps)
    Output roughly in [-1, 1].
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    idx = (g - r) / (g + r + eps)
    return idx

def vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Simple heuristic mask: keep pixels that are likely vegetation.
    Helps ignore background/soil.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # “green-ish” pixels
    mask = (g > r) & (g > b) & (g > 0.15)
    return mask

def ndvi_metrics(rgb: np.ndarray) -> dict:
    idx = ndvi_like_index(rgb)
    mask = vegetation_mask(rgb)

    # If mask is empty (bad image / non-leaf), fall back to whole image
    if mask.sum() < 50:
        valid = idx
    else:
        valid = idx[mask]

    ndvi_mean = float(np.mean(valid))

    # Stress pixels = low index among valid pixels
    # Tune thresholds later, but these work decently as a starting point.
    stress_pixels = valid < 0.10
    stress_percentage = float((np.sum(stress_pixels) / valid.size) * 100.0)

    # Health category based on mean (simple + interpretable)
    if ndvi_mean >= 0.25:
        category = "Healthy"
    elif ndvi_mean >= 0.12:
        category = "Moderate"
    else:
        category = "Stressed"

    return {
        "ndvi_mean": round(ndvi_mean, 4),
        "stress_percentage": round(stress_percentage, 2),
        "health_category": category
    }
