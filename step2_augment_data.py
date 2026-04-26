"""
STEP 2 - Data Augmentation
===========================

"""

import os
import argparse
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


# ─── Augmentation functions ──────────────────────────────────────────────────

def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def rotate(img: np.ndarray, angle_range=(-20, 20)) -> np.ndarray:
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def adjust_brightness(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    factor = random.uniform(0.6, 1.6)
    enhancer = ImageEnhance.Brightness(pil)
    out = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def add_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def adjust_contrast(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    factor = random.uniform(0.7, 1.5)
    enhancer = ImageEnhance.Contrast(pil)
    out = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def zoom_in(img: np.ndarray, zoom_range=(0.8, 0.95)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = random.uniform(*zoom_range)
    new_h, new_w = int(h * scale), int(w * scale)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h))


AUGMENTATIONS = [flip_horizontal, rotate, adjust_brightness,
                 add_noise, blur, adjust_contrast, zoom_in]


# ─── Core logic ──────────────────────────────────────────────────────────────

def augment_person(person_dir: str, min_images: int = 80):
    """Augment images for a single person until reaching min_images."""
    images = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    current = len(images)
    person_name = os.path.basename(person_dir)

    if current == 0:
        print(f"[SKIP] '{person_name}' — no images found in {person_dir}")
        return

    if current >= min_images:
        print(f"[OK]   '{person_name}' already has {current} images (>= {min_images}). No augmentation needed.")
        return

    needed = min_images - current
    print(f"[AUG]  '{person_name}': {current} images → generating {needed} more …")

    aug_idx = 0
    while aug_idx < needed:
        src_file = random.choice(images)
        src_path = os.path.join(person_dir, src_file)
        img = cv2.imread(src_path)
        if img is None:
            continue

        # Apply 1-3 random augmentations in sequence
        num_ops = random.randint(1, 3)
        ops = random.sample(AUGMENTATIONS, min(num_ops, len(AUGMENTATIONS)))
        for op in ops:
            img = op(img)

        out_path = os.path.join(person_dir, f"{person_name}_aug_{aug_idx:04d}.jpg")
        cv2.imwrite(out_path, img)
        aug_idx += 1

    total = len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[✓]   '{person_name}': now has {total} images\n")


def augment_all(persons_dir: str = "persons", min_images: int = 80, only_person: str = None):
    if not os.path.isdir(persons_dir):
        print(f"[ERROR] Persons directory not found: {persons_dir}")
        return

    persons = [
        d for d in os.listdir(persons_dir)
        if os.path.isdir(os.path.join(persons_dir, d))
    ]

    if only_person:
        persons = [p for p in persons if p == only_person]
        if not persons:
            print(f"[ERROR] Person '{only_person}' not found in {persons_dir}")
            return

    if not persons:
        print(f"[ERROR] No person folders found in {persons_dir}")
        return

    print(f"[INFO] Found {len(persons)} person(s): {persons}")
    print(f"[INFO] Minimum images threshold: {min_images}\n")

    for person in sorted(persons):
        augment_person(os.path.join(persons_dir, person), min_images)

    print("Done! Now run:  python step3_train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment face images for team members")
    parser.add_argument("--person", default=None, help="Augment only this person (optional)")
    parser.add_argument("--min",    type=int, default=80, help="Minimum images per person (default: 80)")
    args = parser.parse_args()

    augment_all(persons_dir="persons", min_images=args.min, only_person=args.person)
