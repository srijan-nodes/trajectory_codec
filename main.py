from src.encoder import encode_video
from src.decoder import decode_video_stream
import cv2
import numpy as np


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def check_exact_match(original, reconstructed):
    if len(original) != len(reconstructed):
        print(f"Frame count mismatch ❌ ({len(original)} vs {len(reconstructed)})")
        return False

    for i in range(len(original)):
        if not np.array_equal(original[i], reconstructed[i]):
            print(f"Frame {i} mismatch ❌")
            return False

    print("All frames match perfectly ✅")
    return True


def write_video(frames, output_path, fps):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for f in frames:
        out.write(f)

    out.release()


if __name__ == "__main__":
    input_video = "countdown.mp4"
    encoded_file = "encoded.namaste"

    fps = get_video_fps(input_video)
    print("FPS:", fps)

    encode_video(input_video, encoded_file, QP=2)

    reconstructed_frames = list(decode_video_stream(encoded_file))
    original_frames = extract_frames(input_video)

    print("Original:", len(original_frames))
    print("Reconstructed:", len(reconstructed_frames))

    check_exact_match(original_frames, reconstructed_frames)

    write_video(reconstructed_frames, "reconstructed.mp4", fps)