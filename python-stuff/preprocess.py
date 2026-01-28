from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2  
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

IMAGE_EXTS = {".jpg"}

def list_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def preprocess_pil(img: Image.Image, max_side: int = 1600) -> Image.Image:
    img = ImageOps.exif_transpose(img)  
    img = img.convert("L")              
    img = ImageOps.autocontrast(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img

def preprocess_cv2(img: Image.Image, max_side: int = 1600) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    use_threshold = False
    if use_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 11
        )

    return Image.fromarray(gray)


def save_processed(img: Image.Image, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    img.save(out_path.with_suffix(".png"), format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--max_side", type=int, default=1600)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    images = list_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return 1

    print(f"Found {len(images)} image(s). OpenCV available: {HAS_CV2}")

    for i, img_path in enumerate(images, 1):
        try:
            img = Image.open(img_path)
            if HAS_CV2:
                proc = preprocess_cv2(img, max_side=args.max_side)
            else:
                proc = preprocess_pil(img, max_side=args.max_side)
            rel = img_path.relative_to(input_dir)
            out_path = output_dir / rel
            save_processed(proc, out_path)

            print(f"[{i}/{len(images)}] {img_path.name} -> {out_path.with_suffix('.png')}")
        except Exception as e:
            print(f"[{i}/{len(images)}] FAILED {img_path}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())