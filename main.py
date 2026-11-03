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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database.json")
SIMILARITY_THRESHOLD = 0.63

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
# Drawing & Annotation Helpers
# ──────────────────────────────────────────────
def draw_annotations(frame, detections):
    """
    Draw clean, high-visibility bounding boxes and labels on a frame.
    detections: list of tuples (bbox, label, is_match)
    """
    for bbox, label, is_match in detections:
        x1, y1, x2, y2 = bbox
        # Green for registered matches, Red for unknown faces
        box_color = (40, 200, 40) if is_match else (40, 40, 230)

        # Draw main bounding box with anti-aliasing
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

        # Calculate text badge size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        tw, th = text_size

        # Place label above the box if room allows, otherwise inside the top of the box
        if y1 - th - 8 >= 0:
            badge_y1 = y1 - th - 8
            badge_y2 = y1
            text_y = y1 - 4
        else:
            badge_y1 = y1
            badge_y2 = y1 + th + 8
            text_y = y1 + th + 4

        badge_x1 = x1
        badge_x2 = min(frame.shape[1] - 1, x1 + tw + 8)

        # Draw filled label background
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

        # Draw label text
        cv2.putText(
            frame,
            label,
            (badge_x1 + 4, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def preserve_audio_if_present(source_video, exported_video):
    """
    If the source video has an audio track and ffmpeg is installed,
    mux the original audio into the exported video file without re-encoding video.
    """
    try:
        # Check if source video contains an audio stream
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", source_video],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "audio" not in probe.stdout.lower():
            return exported_video

        # Temporary filename for muxing
        base, ext = os.path.splitext(exported_video)
        temp_muxed = f"{base}_muxed{ext}"

        mux_cmd = [
            "ffmpeg", "-y",
            "-i", exported_video,
            "-i", source_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            temp_muxed
        ]
        res = subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if res.returncode == 0 and os.path.exists(temp_muxed) and os.path.getsize(temp_muxed) > 0:
            os.replace(temp_muxed, exported_video)
    except Exception:
        # If muxing fails, keep the OpenCV-generated video file
        pass
    return exported_video


# ──────────────────────────────────────────────
# Menu option 2: Process video and export to file (Batch mode)
# ──────────────────────────────────────────────
def process_and_export_video():
    """
    Read an input video, run face recognition frame-by-frame,
    and export the fully annotated video into a new file.
    """
    print("\n" + "=" * 50)
    print("   PROCESS VIDEO & EXPORT TO NEW FILE")
    print("=" * 50)

    db = load_database()
    if not db:
        print("Error: Database is empty. Register at least one person first.")
        return

    # Check for available videos in the current folder to suggest
    base_dir = os.path.dirname(__file__)
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    nearby_videos = [
        f for f in sorted(os.listdir(base_dir))
        if os.path.splitext(f)[1].lower() in supported_exts and not f.endswith("_recognized.mp4")
    ]
    if nearby_videos:
        print("Videos found in project directory:")
        for idx, v in enumerate(nearby_videos, 1):
            print(f"  [{idx}] {v}")

    video_input = input("\nEnter video path or number (or press Enter for default): ").strip().strip('"').strip("'")
    if not video_input and nearby_videos:
        video_path = os.path.join(base_dir, nearby_videos[0])
    elif video_input.isdigit() and 1 <= int(video_input) <= len(nearby_videos):
        video_path = os.path.join(base_dir, nearby_videos[int(video_input) - 1])
    else:
        video_path = video_input

    if not os.path.exists(video_path):
        print(f"Error: File not found: '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: '{video_path}'")
        return

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"\nSource Video Information:")
    print(f"  File       : {os.path.basename(video_path)}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames} (~{duration_sec:.1f} seconds)")

    # Output path suggestion
    dir_name = os.path.dirname(os.path.abspath(video_path))
    file_stem = os.path.splitext(os.path.basename(video_path))[0]
    default_output = os.path.join(dir_name, f"{file_stem}_recognized.mp4")

    out_prompt = input(f"\nOutput path [Press Enter for '{os.path.basename(default_output)}']: ").strip().strip('"').strip("'")
    output_path = out_prompt if out_prompt else default_output

    # Performance setting: detection stride
    print("\nProcessing Speed Preset:")
    print("  1. Fast (detect every 3 frames — recommended, smooth & ~3x faster)")
    print("  2. Balanced (detect every 2 frames)")
    print("  3. Maximum precision (detect every single frame — slowest)")
    stride_choice = input("Choice (1-3, default=1): ").strip()
    if stride_choice == "2":
        detect_every_n = 2
    elif stride_choice == "3":
        detect_every_n = 1
    else:
        detect_every_n = 3

    # Live preview option
    preview_prompt = input("Show live preview window while exporting? (y/N) [N is faster]: ").strip().lower()
    show_preview = preview_prompt in ("y", "yes")

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Warning: mp4v codec failed. Trying XVID fallback...")
        output_path = os.path.splitext(output_path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Unable to initialize VideoWriter for '{output_path}'.")
            cap.release()
            return

    print(f"\nExporting to: {output_path}")
    print("Working... Press Ctrl+C in terminal (or 'q' in preview window) to cancel.\n")

    frame_idx = 0
    start_time = time.time()
    last_results = []
    detections_summary = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Face detection every N frames
            if frame_idx % detect_every_n == 0:
                faces = face_app.get(frame)
                last_results = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.normed_embedding
                    matched_name, similarity = find_best_match(embedding, db)

                    if matched_name:
                        label = f"{matched_name} ({similarity:.2f})"
                        last_results.append((bbox, label, True))
                        detections_summary[matched_name] = detections_summary.get(matched_name, 0) + 1
                    else:
                        label = f"UNKNOWN ({similarity:.2f})"
                        last_results.append((bbox, label, False))
                        detections_summary["UNKNOWN"] = detections_summary.get("UNKNOWN", 0) + 1

            # Draw annotations on the frame
            draw_annotations(frame, last_results)

            # Write to output video
            out.write(frame)
            frame_idx += 1

            # Terminal progress bar update
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / current_fps if current_fps > 0 and total_frames > frame_idx else 0
            eta_str = f"{int(remaining)}s" if remaining < 60 else f"{int(remaining // 60)}m{int(remaining % 60):02d}s"

            bar_len = 25
            progress_ratio = frame_idx / total_frames if total_frames > 0 else 1.0
            filled = int(bar_len * progress_ratio)
            if filled >= bar_len:
                bar = "=" * bar_len
            elif filled > 0:
                bar = "=" * (filled - 1) + ">" + " " * (bar_len - filled)
            else:
                bar = " " * bar_len

            status_note = ""
            if last_results:
                top_match = last_results[0][1]
                status_note = f"| {top_match}"

            sys.stdout.write(
                f"\r  [{bar}] {progress_ratio * 100:5.1f}% ({frame_idx}/{total_frames}) | "
                f"{current_fps:4.1f} FPS | ETA: {eta_str} {status_note:<24}"
            )
            sys.stdout.flush()

            # Optional live preview
            if show_preview:
                cv2.imshow("Export Preview (Press 'q' to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nExport stopped early by user.")
                    break

    except KeyboardInterrupt:
        print("\n\nExport interrupted by user.")
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    sys.stdout.write("\n")

    if frame_idx == 0:
        print("No frames were processed.")
        return

    # Preserve audio if present
    print("Checking audio streams...")
    final_output = preserve_audio_if_present(video_path, output_path)

    file_size_mb = os.path.getsize(final_output) / (1024 * 1024) if os.path.exists(final_output) else 0

    print("\n" + "=" * 50)
    print("   EXPORT COMPLETE!")
    print("=" * 50)
    print(f"  Output File     : {final_output}")
    print(f"  File Size       : {file_size_mb:.2f} MB")
    print(f"  Frames Written  : {frame_idx} / {total_frames}")
    print(f"  Time Elapsed    : {total_time:.1f} seconds")
    print(f"  Avg Process Speed: {avg_fps:.1f} FPS")

    if detections_summary:
        print("\n  Detections Summary:")
        for name, count in sorted(detections_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {name}: detected in {count} analyzed frames")
    else:
        print("\n  No faces detected in the video.")
    print("=" * 50)

    # Prompt to open exported video
    open_choice = input("\nWould you like to open the exported video now? (Y/n): ").strip().lower()
    if open_choice in ("", "y", "yes"):
        try:
            os.startfile(final_output)
        except Exception as e:
            print(f"Note: Could not automatically launch player: {e}")


# ──────────────────────────────────────────────
# Menu option 3: Live real-time video preview
# ──────────────────────────────────────────────
def recognize_video_live():
    """Live interactive preview of face recognition on video playback."""
    print("\n--- Live Video Recognition Preview ---")

    db = load_database()
    if not db:
        print("Error: Database is empty. Register at least one person first.")
        return

    video_path = input("Enter the path to the MP4 video: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps) if fps > 0 else 30

    print(f"Playing video: {video_path}")
    print(f"  FPS    : {fps:.1f}")
    print(f"  Frames : {total_frames}")
    print("  Press 'q' in the window to quit.\n")

    DETECT_EVERY_N = 3
    frame_idx = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N == 0:
            faces = face_app.get(frame)
            last_results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding
                matched_name, similarity = find_best_match(embedding, db)
                if matched_name:
                    label = f"{matched_name} ({similarity:.2f})"
                    last_results.append((bbox, label, True))
                else:
                    label = f"UNKNOWN ({similarity:.2f})"
                    last_results.append((bbox, label, False))

        draw_annotations(frame, last_results)
        cv2.imshow("Live Face Recognition (Press 'q' to quit)", frame)
        frame_idx += 1

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("\nPlayback stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished preview after {frame_idx} frames.")


# ──────────────────────────────────────────────
# Menu option 4: List registered people
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
        print("\n" + "=" * 44)
        print("         FACE RECOGNITION SYSTEM")
        print("=" * 44)
        print("  1. Register a person")
        print("  2. Process video and export to new file (Recommended)")
        print("  3. Live video playback preview")
        print("  4. List registered people")
        print("  5. Exit")
        print("=" * 44)

        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            register_person()
        elif choice == "2":
            process_and_export_video()
        elif choice == "3":
            recognize_video_live()
        elif choice == "4":
            list_registered()
        elif choice == "5":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()

