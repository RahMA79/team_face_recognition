"""
STEP 3 - Train the Face Recognition Model
===========================================
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from face_recognition_knn_classifier import train

TRAIN_DIR  = "persons"
MODEL_PATH = os.path.join("model", "face_recognition_model.clf")


def main():
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] Training directory not found: {TRAIN_DIR}")
        print("        Run step1_extract_frames.py first.")
        return

    persons = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    if not persons:
        print(f"[ERROR] No person folders found inside {TRAIN_DIR}/")
        return

    print(f"[INFO] Persons detected: {persons}")
    print("[INFO] Starting training …\n")

    train(
        train_dir=TRAIN_DIR,
        model_save_path=MODEL_PATH,
        n_neighbors=None,   # auto
    )

    print("\n[✓] Training complete!")
    print("    Next step: run the web app with  python web/app.py")


if __name__ == "__main__":
    main()
