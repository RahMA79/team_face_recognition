"""
STEP 1 - Extract Frames from Video
===================================

"""

import cv2
import os
import argparse


def extract_frames(video_path: str, person_name: str, max_frames: int = 200, skip: int = 5):

    output_dir = os.path.join("persons", person_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {os.path.basename(video_path)}")
    print(f"[INFO] Total frames: {total_video_frames} | FPS: {fps:.1f}")
    print(f"[INFO] Saving every {skip}th frame (max {max_frames}) → {output_dir}")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break

        if frame_idx % skip == 0:
            filename = os.path.join(output_dir, f"{person_name}_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[✓] Saved {saved} frames for '{person_name}' in → {output_dir}\n")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video for a team member")
    parser.add_argument("--video",   required=True, help="Path to video file")
    parser.add_argument("--person",  required=True, help="Name of the person")
    parser.add_argument("--max",     type=int, default=200, help="Max frames to extract (default: 200)")
    parser.add_argument("--skip",    type=int, default=5,   help="Save every Nth frame (default: 5)")
    args = parser.parse_args()

    count = extract_frames(args.video, args.person, args.max, args.skip)
    if count is not None and count < 50:
        print(f"[⚠] Only {count} frames extracted. Consider running step2_augment_data.py")
