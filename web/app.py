"""
Web App - Team Face Recognition
=================================
Run:
    cd team_face_recognition
    python web/app.py

Open: http://localhost:5000
"""

import os
import sys
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from face_recognition_knn_classifier import predict

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "face_recognition_model.clf")
DISTANCE_THRESHOLD = 0.5

# ─── Color palette for different people (BGR for OpenCV) ────────────────────
COLORS = [
    (52, 211, 153),   # green
    (99, 102, 241),   # indigo
    (251, 146, 60),   # orange
    (239, 68, 68),    # red
    (14, 165, 233),   # sky blue
    (168, 85, 247),   # purple
    (234, 179, 8),    # yellow
    (20, 184, 166),   # teal
]
_person_colors = {}


def get_color(name: str):
    if name not in _person_colors:
        _person_colors[name] = COLORS[len(_person_colors) % len(COLORS)]
    return _person_colors[name]


def decode_image(data_url: str) -> np.ndarray:
    """Convert a base64 data URL to an OpenCV BGR image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image(img: np.ndarray) -> str:
    """Convert an OpenCV BGR image back to a base64 data URL."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def draw_predictions(img_bgr: np.ndarray, predictions):
    """Draw bounding boxes and names on the image."""
    for name, (top, right, bottom, left) in predictions:
        color = get_color(name)

        # Bounding box
        cv2.rectangle(img_bgr, (left, top), (right, bottom), color, 2)

        # Label background
        label = name
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(img_bgr, (left, top - text_h - 12), (left + text_w + 8, top), color, -1)

        # Label text
        cv2.putText(
            img_bgr, label,
            (left + 4, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2,
        )
    return img_bgr


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = os.path.isfile(MODEL_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """Shared endpoint for both camera frames and uploaded images."""
    if not os.path.isfile(MODEL_PATH):
        return jsonify({"error": "Model not found. Run step3_train_model.py first."}), 400

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_bgr = decode_image(data["image"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        predictions = predict(img_rgb, MODEL_PATH, DISTANCE_THRESHOLD)

        # Draw results
        result_img = draw_predictions(img_bgr.copy(), predictions)
        result_b64 = encode_image(result_img)

        persons_found = [
            {"name": name, "location": {"top": loc[0], "right": loc[1], "bottom": loc[2], "left": loc[3]}}
            for name, loc in predictions
        ]

        return jsonify({
            "image": result_b64,
            "persons": persons_found,
            "count": len(predictions),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def status():
    model_exists = os.path.isfile(MODEL_PATH)
    return jsonify({"model_ready": model_exists, "model_path": MODEL_PATH})


if __name__ == "__main__":
    if not os.path.isfile(MODEL_PATH):
        print("[⚠]  Model not found. Run step3_train_model.py first.")
        print(f"     Expected at: {MODEL_PATH}\n")
    else:
        print(f"[✓]  Model loaded from {MODEL_PATH}")
    print("[🌐] Starting web server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
