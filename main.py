"""
Face Recognition System
=======================
A simple menu-driven tool to:
  1. Register people (name + photo) into a JSON database
  2. Recognize faces from an MP4 video against the database
  3. List all registered people
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database.json")
SIMILARITY_THRESHOLD = 0.7

# ──────────────────────────────────────────────
# Initialize InsightFace (once, shared by all operations)
# Auto-detect GPU: use CUDA if CUDA runtime DLLs exist, otherwise CPU
# ──────────────────────────────────────────────
def is_cuda_ready():
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        return False
    import ctypes
    for dll in ["cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt64_11.dll"]:
        try:
            ctypes.cdll.LoadLibrary(dll)
            return True
        except Exception:
            continue
    return False

if is_cuda_ready():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("GPU (CUDA) detected and ready — using GPU acceleration!")
else:
    providers = ["CPUExecutionProvider"]
    print("Using CPU execution provider.")

print("Loading InsightFace model (this may take a moment)...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=providers
)
# 320x320 is ~4x faster than 640x640 with minimal accuracy loss for videos
face_app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(320, 320))
print("Model loaded successfully!\n")


# ──────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────
def load_database():
    """Load the face database from a JSON file."""
    if not os.path.exists(DATABASE_PATH):
        return {}
    with open(DATABASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    db = {}
    for name, raw in data.items():
        arr = np.array(raw)
        # Ensure 2D array: (num_embeddings, 512)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        db[name] = arr
    return db


def save_database(db):
    """Save the face database to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    data = {name: embs.tolist() for name, embs in db.items()}
    with open(DATABASE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────
def extract_embedding(image, high_accuracy=False):
    """
    Extract a face embedding from an image (numpy array).
    When high_accuracy is True (used during registration), attempts higher
    resolution and sensitive detection to capture profile/side views.
    Returns (embedding, status_string).
    """
    if not high_accuracy:
        faces = face_app.get(image)
        if len(faces) == 0:
            return None, "No face detected"
        primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return primary_face.normed_embedding, "OK"

    # High accuracy registration path
    ctx = 0 if "CUDAExecutionProvider" in providers else -1
    try:
        # First attempt: 640x640 at default threshold
        face_app.prepare(ctx_id=ctx, det_size=(640, 640), det_thresh=0.5)
        faces = face_app.get(image)

        # Fallback attempt for challenging side/profile angles
        if len(faces) == 0:
            face_app.prepare(ctx_id=ctx, det_size=(960, 960), det_thresh=0.08)
            faces = face_app.get(image)

        if len(faces) == 0:
            return None, "No face detected"

        # Choose the largest detected face
        primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return primary_face.normed_embedding, "OK"
    finally:
        # Always restore default fast video playback settings
        face_app.prepare(ctx_id=ctx, det_size=(320, 320), det_thresh=0.5)


def find_best_match(embedding, database):
    """
    Compare an embedding against all entries in the database.
    Each person can have multiple registered embeddings (front and sides).
    Returns (name, similarity) of the best match, or (None, best_sim).
    """
    best_name = None
    best_sim = 0.0

    for name, db_embs in database.items():
        # db_embs is (N, 512). np.dot gives cosine similarities for all angles of this person
        sims = np.dot(db_embs, embedding)
        person_best = float(np.max(sims))
        if person_best > best_sim:
            best_sim = person_best
            best_name = name

    if best_sim >= SIMILARITY_THRESHOLD:
        return best_name, best_sim
    return None, best_sim


# ──────────────────────────────────────────────
# Menu option 1: Register a person (multi-angle support)
# ──────────────────────────────────────────────
def register_person():
    """Register a new person into the database using multiple photos (front & sides)."""
    print("\n--- Register a New Person (Front & Side Views) ---")
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Error: Name cannot be empty.")
        return

    # Check if a folder named database/<name> or <name> already exists
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    base_dir = os.path.dirname(__file__)
    candidate_folder = os.path.join(base_dir, "database", name)
    if not os.path.isdir(candidate_folder):
        candidate_folder = os.path.join(base_dir, name)

    image_paths = []

    if os.path.isdir(candidate_folder):
        found_files = [
            os.path.join(candidate_folder, f)
            for f in sorted(os.listdir(candidate_folder))
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        if found_files:
            print(f"Found folder '{os.path.relpath(candidate_folder, base_dir)}' with {len(found_files)} image(s):")
            for f in found_files:
                print(f"  - {os.path.basename(f)}")
            use_folder = input("Use these images for registration? (Y/n): ").strip().lower()
            if use_folder in ("", "y", "yes"):
                image_paths = found_files

    if not image_paths:
        print("\nChoose how to provide the photos:")
        print("  1. Enter a folder path containing the photos (front and sides)")
        print("  2. Enter individual image paths (up to 4 images)")
        choice = input("Choice (1 or 2): ").strip()

        if choice == "1":
            folder_path = input("Enter folder path: ").strip().strip('"').strip("'")
            if not os.path.isdir(folder_path):
                print(f"Error: Folder not found: {folder_path}")
                return
            image_paths = [
                os.path.join(folder_path, f)
                for f in sorted(os.listdir(folder_path))
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            if not image_paths:
                print(f"Error: No image files found in {folder_path}")
                return
        else:
            angle_names = ["Front view", "Left profile", "Right profile", "Extra / Angle 4"]
            print("\nEnter paths for up to 4 images (press Enter without typing a path if you have fewer):")
            for i, label in enumerate(angle_names, 1):
                p = input(f"  Photo {i} [{label}]: ").strip().strip('"').strip("'")
                if p:
                    if os.path.exists(p):
                        image_paths.append(p)
                    else:
                        print(f"    Warning: File not found: '{p}', skipped.")

    if not image_paths:
        print("Error: No valid images provided.")
        return

    print(f"\nProcessing {len(image_paths)} image(s) for '{name}'...")
    embeddings = []

    for idx, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"  [{idx}/{len(image_paths)}] {filename}: Failed to read image file.")
            continue

        emb, status = extract_embedding(img, high_accuracy=True)
        if emb is not None:
            embeddings.append(emb)
            print(f"  [{idx}/{len(image_paths)}] {filename}: Face detected successfully -> Added ({status})")
        else:
            print(f"  [{idx}/{len(image_paths)}] {filename}: Warning - {status}, skipping.")

    if not embeddings:
        print(f"\nError: Could not extract face embeddings from any of the provided images.")
        return

    db = load_database()
    if name in db:
        confirm = input(f"\n'{name}' already exists in database. Overwrite? (y/n): ").strip().lower()
        if confirm != "y":
            print("Registration cancelled.")
            return

    db[name] = np.array(embeddings)
    save_database(db)
    print(f"\nSuccessfully registered '{name}' with {len(embeddings)} face angle(s) in database.json!")


# ──────────────────────────────────────────────
# Menu option 2: Recognize faces in a video
# ──────────────────────────────────────────────
def recognize_video():
    """Process an MP4 video and recognize faces against the database."""
    print("\n--- Recognize Faces in Video ---")

    db = load_database()
    if not db:
        print("Error: Database is empty. Register at least one person first.")
        return

    video_path = input("Enter the path to the MP4 video: ").strip()
    video_path = video_path.strip('"').strip("'")

    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps) if fps > 0 else 30  # ms between frames

    print(f"Playing video: {video_path}")
    print(f"  FPS    : {fps:.1f}")
    print(f"  Frames : {total_frames}")
    print("  Press 'q' to quit.\n")

    # Process every Nth frame for speed (detect every 3 frames, draw on all)
    DETECT_EVERY_N = 5
    frame_idx = 0
    last_results = []  # Cache of (bbox, name, similarity) from last detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N == 0:
            # Run face detection and recognition on this frame
            faces = face_app.get(frame)
            last_results = []

            for face in faces:
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                embedding = face.normed_embedding
                matched_name, similarity = find_best_match(embedding, db)

                if matched_name:
                    label = f"{matched_name} ({similarity:.2f})"
                    last_results.append((bbox, label, True))
                else:
                    label = f"UNKNOWN ({similarity:.2f})"
                    last_results.append((bbox, label, False))

                # Print to terminal
                timestamp = frame_idx / fps if fps > 0 else 0
                print(f"  [{timestamp:>7.2f}s] Frame {frame_idx:>5d}: {label}")

        # Draw bounding boxes and labels on the frame
        for bbox, label, is_match in last_results:
            x1, y1, x2, y2 = bbox
            color = (0, 200, 0) if is_match else (0, 0, 255)  # Green = match, Red = unknown

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 4, y1),
                color,
                -1,
            )
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Show the frame in a window
        cv2.imshow("Face Recognition", frame)
        frame_idx += 1

        # Press 'q' to quit early
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("\n  Playback stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Played {frame_idx} frames.")


# ──────────────────────────────────────────────
# Menu option 3: List registered people
# ──────────────────────────────────────────────
def list_registered():
    """List all people in the database."""
    print("\n--- Registered People ---")
    db = load_database()
    if not db:
        print("Database is empty. No one is registered yet.")
        return
    for i, (name, embs) in enumerate(db.items(), 1):
        num_angles = len(embs)
        print(f"  {i}. {name} ({num_angles} face angle{'s' if num_angles != 1 else ''})")
    print(f"\nTotal: {len(db)} person(s)")


# ──────────────────────────────────────────────
# Main menu
# ──────────────────────────────────────────────
def main():
    """Run the interactive menu."""
    while True:
        print("\n" + "=" * 40)
        print("   FACE RECOGNITION SYSTEM")
        print("=" * 40)
        print("  1. Register a person")
        print("  2. Recognize faces in a video")
        print("  3. List registered people")
        print("  4. Exit")
        print("=" * 40)

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            register_person()
        elif choice == "2":
            recognize_video()
        elif choice == "3":
            list_registered()
        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
