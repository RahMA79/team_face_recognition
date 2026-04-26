# 👥 Team Face Recognition

A complete face recognition pipeline for your team — from video recording to a live web application.

---

## 🗺️ Pipeline Overview

```
Video → Extract Frames → Augmentation → Train Model → Web App (Camera / Upload)
```

---

## 📁 Project Structure

```
team_face_recognition/
│
├── step1_extract_frames.py     ← Extract frames from member videos
├── step2_augment_data.py       ← Auto-augment if images are few
├── step3_train_model.py        ← Train the KNN face recognition model
│
├── src/
│   └── face_recognition_knn_classifier.py   ← Core KNN model logic
│
├── web/
│   ├── app.py                  ← Flask web server
│   ├── templates/index.html    ← Web UI
│   └── static/                 ← CSS + JS
│
├── persons/                    ← Training images (auto-created)
│   ├── ahmed/
│   ├── sara/
│   └── ...
│
├── model/                      ← Saved model (auto-created after training)
│   └── face_recognition_model.clf
│
├── data/
│   └── videos/                 ← Put your team member videos here
│
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

> **Note:** `dlib` requires CMake and a C++ compiler.
> - Windows: Install [CMake](https://cmake.org/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
> - Ubuntu/Mac: `sudo apt install cmake build-essential` / `brew install cmake`

---

## 🚀 How to Use — Step by Step

### Step 1 — Each team member records a short video of themselves

Put the videos inside `data/videos/`.

Recommended: 30–60 second video, looking at the camera from different angles.

---

### Step 2 — Extract frames from each video

```bash
python step1_extract_frames.py --video data/videos/ahmed.mp4 --person ahmed
python step1_extract_frames.py --video data/videos/sara.mp4  --person sara
python step1_extract_frames.py --video data/videos/omar.mp4  --person omar
```

**Options:**

| Flag       | Default | Description                          |
|------------|---------|--------------------------------------|
| `--video`  | —       | Path to the video file               |
| `--person` | —       | Person's name (used as folder name)  |
| `--max`    | 200     | Max frames to extract                |
| `--skip`   | 5       | Save 1 frame every N frames          |

This creates `persons/ahmed/`, `persons/sara/` etc. with `.jpg` images.

---

### Step 3 — (Optional) Augment images if count is low

If any person has fewer than **80 images**, run:

```bash
python step2_augment_data.py
```

To augment only one person:

```bash
python step2_augment_data.py --person ahmed
```

To set a custom minimum:

```bash
python step2_augment_data.py --min 120
```

Augmentations applied: horizontal flip, rotation, brightness, contrast, noise, blur, zoom.

---

### Step 4 — Train the model

```bash
python step3_train_model.py
```

This saves the model to `model/face_recognition_model.clf`.

Training time depends on the number of images. Expect 1–5 minutes for a small team.

---

### Step 5 — Launch the Web App

```bash
python web/app.py
```

Open your browser at: **http://localhost:5000**

#### Web app features:
- 🎥 **Live Camera** — Opens your webcam and recognizes faces on demand
- 🖼️ **Upload Image** — Upload any photo and identify team members in it

---

## 🛠️ Tips

- **Better accuracy**: Record the video in good lighting, from different angles.
- **More data = better results**: Aim for at least 80 images per person.
- **Unknown faces**: Anyone not in the training set will be labeled "Unknown".
- **Distance threshold**: Adjust `DISTANCE_THRESHOLD` in `web/app.py` (default `0.5`).
  - Lower = stricter (fewer false positives)
  - Higher = more lenient (may misidentify)

---

## 📦 Tech Stack

| Component        | Library                  |
|------------------|--------------------------|
| Face detection   | `face_recognition` (dlib)|
| Classification   | `scikit-learn` KNN       |
| Image processing | `OpenCV`, `Pillow`       |
| Web framework    | `Flask`                  |
| Frontend         | Vanilla HTML/CSS/JS      |
