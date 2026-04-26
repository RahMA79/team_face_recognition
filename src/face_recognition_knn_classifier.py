"""
KNN Face Recognition Classifier
=================================
Trains a K-Nearest Neighbors classifier on face encodings.
"""

import math
import os
import pickle
import numpy as np
from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from tqdm import tqdm
from PIL import Image, ImageEnhance


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Sharpen an image to improve face detection accuracy."""
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(pil_img)
    return np.array(enhancer.enhance(2.0))


def train(train_dir: str, model_save_path: str, n_neighbors: int = None) -> neighbors.KNeighborsClassifier:
    """
    Train a KNN classifier on face encodings from the training directory.

    Directory structure expected:
        persons/
            ahmed/
                img1.jpg
                img2.jpg
            sara/
                img1.jpg
                ...

    Args:
        train_dir        : Path to persons/ directory
        model_save_path  : Where to save the trained model (.clf file)
        n_neighbors      : Number of neighbors. If None, sqrt(n_samples) is used.

    Returns:
        Trained KNeighborsClassifier
    """
    encodings_list = []
    names_list = []

    # Count total images for the progress bar
    total_images = sum(
        len(list(image_files_in_folder(os.path.join(train_dir, d))))
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )

    print(f"[INFO] Found {total_images} training images across all persons.")

    with tqdm(total=total_images, desc="Encoding faces", unit="img") as pbar:
        for class_dir in sorted(os.listdir(train_dir)):
            person_path = os.path.join(train_dir, class_dir)
            if not os.path.isdir(person_path):
                continue

            for img_path in image_files_in_folder(person_path):
                try:
                    image = face_recognition.load_image_file(img_path)
                    image = enhance_image(image)

                    face_locations = face_recognition.face_locations(image, model="hog")

                    if len(face_locations) != 1:
                        # Skip images with 0 or multiple faces
                        pbar.update(1)
                        continue

                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    encodings_list.append(encoding)
                    names_list.append(class_dir)

                except Exception as e:
                    print(f"\n[WARN] Skipping {img_path}: {e}")

                pbar.update(1)

    if len(encodings_list) == 0:
        raise ValueError("No valid face encodings found. Check your persons/ directory.")

    print(f"\n[INFO] Successfully encoded {len(encodings_list)} images.")

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(encodings_list))))
        print(f"[INFO] Auto-selected n_neighbors = {n_neighbors}")

    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        weights="distance",
    )
    knn_clf.fit(np.array(encodings_list), np.array(names_list))

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(knn_clf, f)

    print(f"[✓] Model saved → {model_save_path}")
    return knn_clf


def load_model(model_path: str) -> neighbors.KNeighborsClassifier:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(image_array: np.ndarray, model_path: str, distance_threshold: float = 0.5):
    """
    Predict faces in an image.

    Args:
        image_array        : RGB numpy array
        model_path         : Path to the saved .clf model
        distance_threshold : Max KNN distance to be considered a match

    Returns:
        List of (name, (top, right, bottom, left)) tuples
    """
    knn_clf = load_model(model_path)

    image = enhance_image(image_array)
    face_locations = face_recognition.face_locations(image, model="hog")

    if len(face_locations) == 0:
        return []

    face_encodings = face_recognition.face_encodings(image, face_locations)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    results = []
    for pred, loc, match in zip(knn_clf.predict(face_encodings), face_locations, are_matches):
        name = pred if match else "Unknown"
        results.append((name, loc))

    return results
