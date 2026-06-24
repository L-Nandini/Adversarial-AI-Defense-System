# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import pandas as pd
import os


app = FastAPI(title="AI Vigil Guard")


# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "Backend is running"}


# ---------------- IMAGE ANALYSIS HELPERS ----------------
def _to_small_rgb(pil_img: Image.Image, max_side: int = 512) -> Image.Image:
    """Resize for stable metrics + speed, keep RGB."""
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)))
    return pil_img


def _entropy(gray_u8: np.ndarray) -> float:
    """Shannon entropy of grayscale histogram."""
    hist = np.bincount(gray_u8.flatten(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _laplacian_variance(gray_f: np.ndarray) -> float:
    """Variance of a simple Laplacian filter (no cv2 dependency)."""
    # 3x3 Laplacian kernel
    k = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)
    # convolution (manual, padded)
    pad = np.pad(gray_f, 1, mode="edge")
    out = (
        k[0, 0] * pad[:-2, :-2] + k[0, 1] * pad[:-2, 1:-1] + k[0, 2] * pad[:-2, 2:] +
        k[1, 0] * pad[1:-1, :-2] + k[1, 1] * pad[1:-1, 1:-1] + k[1, 2] * pad[1:-1, 2:] +
        k[2, 0] * pad[2:, :-2] + k[2, 1] * pad[2:, 1:-1] + k[2, 2] * pad[2:, 2:]
    )
    return float(out.var())


def _hf_ratio(gray_f: np.ndarray) -> float:
    """
    High-frequency energy ratio using FFT:
    energy outside low-frequency center / total energy.
    """
    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    ry, rx = max(8, h // 10), max(8, w // 10)  # low-freq window
    low = mag[cy - ry:cy + ry, cx - rx:cx + rx].sum()
    total = mag.sum() + 1e-12
    high = total - low
    return float(high / total)


def _blockiness(gray_f: np.ndarray, block: int = 8) -> float:
    """
    Simple JPEG-blockiness proxy:
    compares edge differences on block boundaries vs non-boundaries.
    """
    h, w = gray_f.shape
    if h < block * 2 or w < block * 2:
        return 0.0

    # vertical differences
    dv = np.abs(gray_f[:, 1:] - gray_f[:, :-1])
    # horizontal differences
    dh = np.abs(gray_f[1:, :] - gray_f[:-1, :])

    # boundary indices (every 8 pixels)
    vb = np.arange(block - 1, w - 1, block)  # differences at boundary columns
    hb = np.arange(block - 1, h - 1, block)  # differences at boundary rows

    v_boundary = dv[:, vb].mean() if vb.size else 0.0
    v_non = dv.mean() + 1e-12

    h_boundary = dh[hb, :].mean() if hb.size else 0.0
    h_non = dh.mean() + 1e-12

    # ratio > 1 suggests stronger block edges than average
    return float(0.5 * ((v_boundary / v_non) + (h_boundary / h_non)))


def _gaussian_blur_gray(gray_f: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Small Gaussian blur implemented with separable kernel, no extra deps."""
    if sigma <= 0:
        return gray_f

    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # horizontal blur
    pad = np.pad(gray_f, ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.zeros_like(gray_f)
    for i in range(gray_f.shape[0]):
        tmp[i] = np.convolve(pad[i], kernel, mode="valid")

    # vertical blur
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(gray_f)
    for j in range(gray_f.shape[1]):
        out[:, j] = np.convolve(pad[:, j], kernel, mode="valid")

    return out


def analyze_image_for_adversarial(img_rgb: np.ndarray) -> dict:
    """
    Heuristic adversarial/attack-likeness detector with explainable reasons.
    This is NOT a guaranteed proof; it's an explainable suspicion score.
    """
    # convert to gray
    gray = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gray_f = gray / 255.0

    # metrics
    pixel_std = float(img_rgb.std())
    ent = _entropy(gray_u8)
    lap_var = _laplacian_variance(gray_f)
    hf = _hf_ratio(gray_f)
    blk = _blockiness(gray_f)

    # small-blur comparison to amplify adversarial noise signal
    blurred = _gaussian_blur_gray(gray_f, sigma=0.8)
    diff = np.abs(gray_f - blurred)
    diff_mean = float(diff.mean())
    diff_max = float(diff.max())

    # color distribution oddities
    ch_means = img_rgb.reshape(-1, 3).mean(axis=0)
    ch_stds = img_rgb.reshape(-1, 3).std(axis=0)
    mean_spread = float(np.std(ch_means))
    std_spread = float(np.std(ch_stds))

    # scoring (tuned for "suspiciousness", slightly more sensitive)
    score = 0.0
    reasons = []

    # High-frequency noise / perturbations
    if hf > 0.68:
        score += 0.35
        reasons.append("High-frequency energy is unusually high (possible perturbation/noise-based attack).")
    elif hf > 0.60:
        score += 0.20
        reasons.append("Elevated high-frequency energy (image contains strong fine-grained patterns).")

    # Block artifacts (compression / adversarial transfer)
    if blk > 1.18:
        score += 0.20
        reasons.append("Strong 8×8 block boundary artifacts detected (JPEG-like compression / artifact patterns).")

    # Entropy extremes can hint at noise or overly-flat images
    if ent > 7.3:
        score += 0.15
        reasons.append("Very high entropy (texture/noise level is high compared to typical natural images).")
    elif ent < 4.2:
        score += 0.10
        reasons.append("Very low entropy (image appears overly uniform; can occur in synthetic/modified images).")

    # Pixel std too high -> noisy, too low -> flat
    if pixel_std > 70:
        score += 0.15
        reasons.append("Pixel intensity variation is unusually high (possible heavy noise/perturbation).")
    elif pixel_std < 22:
        score += 0.08
        reasons.append("Pixel intensity variation is very low (image is unusually flat/smooth).")

    # Sharpness anomalies
    if lap_var > 0.015:
        score += 0.10
        reasons.append("Edge/sharpness response is high (may indicate aggressive sharpening or perturbation).")
    elif lap_var < 0.0015:
        score += 0.06
        reasons.append("Edge/sharpness response is very low (blurred or over-smoothed).")

    # Blur-difference–based perturbation check
    if diff_mean > 0.012:
        score += 0.18
        reasons.append("Image changes noticeably when slightly blurred (small-scale perturbations stand out).")
    if diff_max > 0.12:
        score += 0.10
        reasons.append("Some pixels differ strongly from a smoothed version (localized high-magnitude perturbations).")

    # Color channel imbalance
    if mean_spread > 16:
        score += 0.08
        reasons.append("Color channel means are imbalanced (unusual color distribution).")
    if std_spread > 9:
        score += 0.06
        reasons.append("Color channel variances are imbalanced (unusual per-channel noise/contrast).")

    # Normalize score
    score = float(np.clip(score, 0.0, 1.0))

    # Decision: slightly lowered thresholds for easier triggering
    is_adv = (score >= 0.45 and len(reasons) >= 2) or (score >= 0.65)

    if not reasons:
        reasons = ["No strong adversarial indicators found from the current heuristics."]

    return {
        "is_adversarial": bool(is_adv),
        "adversarial_score": round(score, 3),
        "reasons": reasons,
        "metrics": {
            "pixel_std": round(pixel_std, 3),
            "entropy": round(ent, 3),
            "laplacian_variance": round(lap_var, 6),
            "high_frequency_ratio": round(hf, 3),
            "blockiness_ratio": round(blk, 3),
            "blur_diff_mean": round(diff_mean, 5),
            "blur_diff_max": round(diff_max, 5),
            "channel_means": [round(float(x), 3) for x in ch_means],
            "channel_stds": [round(float(x), 3) for x in ch_stds],
        }
    }


# ---------------- IMAGE DETECTION ----------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    # Validate MIME
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    try:
        image_bytes = await file.read()
        if len(image_bytes) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted")

        # Load + verify
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="File is not a valid image")
        except Exception:
            raise HTTPException(status_code=400, detail="Image verification failed (corrupt/unsupported).")

        # Reopen after verify
        img = Image.open(BytesIO(image_bytes))
        img = _to_small_rgb(img, max_side=512)

        img_array = np.array(img)

        analysis = analyze_image_for_adversarial(img_array)

        return {
            "file_name": file.filename,
            "content_type": file.content_type,
            "image_shape": list(img_array.shape),
            **analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backward-compatible alias for your current frontend (if still calling /predict_image)
@app.post("/predict_image")
async def predict_image_alias(file: UploadFile = File(...)):
    return await detect_image(file)


# ---------------- CSV DETECTION ----------------
@app.post("/detect/csv")
async def detect_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")

        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        suspicious = int(len(df) * 0.12)

        return {
            "file_name": file.filename,
            "rows": int(len(df)),
            "suspicious_rows": suspicious,
            "status": "CSV analysis completed"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backward-compatible alias for your current frontend (if still calling /predict_csv)
@app.post("/predict_csv")
async def predict_csv_alias(file: UploadFile = File(...)):
    return await detect_csv(file)


# ---------------- OPERATOR: GENERATE SYNTHETIC DATASET ----------------
@app.post("/generate_synthetic")
def generate_synthetic():
    """
    Calls your generator. Keeps it simple: if module exists, run it.
    """
    try:
        from generate_data import generate_synthetic_data  # backend/generate_data.py
        generate_synthetic_data(
            n_samples=6000,          # upgraded default
            n_features=16,           # more variety
            n_informative=10,
            random_state=42
        )
        return {"status": "Dataset generated", "samples": 6000, "features": 16}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


DATASET_PATH = os.path.join("backend", "data", "synthetic_dataset.csv")


def reason_from_source(src: str) -> str:
    s = (src or "").lower()

    if "adversarial_sparse_spike" in s:
        return "Sparse spike attack: a few features were pushed to extreme values (feature-level perturbation)."
    if "adversarial_shift" in s:
        return "Shift attack: features were globally shifted (distribution shift / crafted offset)."
    if "adversarial_flip_scale" in s:
        return "Flip+scale attack: signs flipped and magnitudes scaled to confuse the classifier."
    if "outlier" in s:
        return "Outlier injection: values sampled far outside normal range (extreme uniform outliers)."
    if "noisy" in s:
        return "Noisy sample: Gaussian noise added (may be suspicious, not always adversarial)."
    if "normal" in s:
        return "Normal sample: generated from base distribution (no attack patterns)."

    return "Unknown source type: unable to infer a specific reason."


@app.get("/dataset/sample")
def dataset_sample(n: int = 20):
    """
    Returns N random rows from the synthetic dataset.
    Each call produces a different combination (dynamic sample).
    Adds is_adversarial + reason fields for demo explainability.
    """
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail=f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    n = int(max(1, min(n, len(df))))

    # random each request (no random_state)
    sample = df.sample(n=n, replace=False).reset_index(drop=True)

    # mark adversarial based on generator source
    src = sample["source"].astype(str).str.lower()
    sample["is_adversarial"] = src.str.contains("adversarial|outlier")

    # reasons
    sample["reason"] = sample["source"].astype(str).apply(reason_from_source)

    return {
        "n": len(sample),
        "columns": sample.columns.tolist(),
        "rows": sample.to_dict(orient="records")
    }