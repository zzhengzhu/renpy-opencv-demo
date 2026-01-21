#!/usr/bin/env python3

import json
import os
import sys
import traceback

import cv2
import numpy as np

# Optional dependency: onnxruntime
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None


# OpenCV MobileFaceNet FER labels (7 classes)
FER_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def _write_json(out_path: str | None, payload: dict) -> None:
    """Write JSON to out_path if provided, else stdout."""
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    else:
        # One-line JSON so it's easy to parse.
        print(json.dumps(payload), flush=True)


def _log(out_path: str | None, msg: str) -> None:
    """Best-effort log to a sibling .log file (useful when launched via `open`)."""
    try:
        log_path = (out_path + ".log") if out_path else "/tmp/fer_helper.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _debug_frame_paths(out_path: str | None):
    """Return (raw_path, processed_path) for debug frame captures."""
    if out_path:
        return out_path + ".frame_raw.jpg", out_path + ".frame.jpg"
    # If no out_path was provided (e.g., running from Terminal), fall back to /tmp.
    return "/tmp/fer_helper.frame_raw.jpg", "/tmp/fer_helper.frame.jpg"


def _brighten_frame(frame_bgr):
    """Brighten a dark frame to help face detection.

    - Gamma correction lifts shadows.
    - CLAHE on luminance improves local contrast.
    """
    # Gamma < 1 brightens. 0.6 is a good starting point.
    gamma = 0.6
    inv = 1.0 / gamma
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** inv) * 255.0
    table = table.astype(np.uint8)
    out = cv2.LUT(frame_bgr, table)

    # CLAHE on L channel in LAB space
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _detect_primary_face_bgr(frame_bgr):
    """Return (x, y, w, h) for the largest detected face, or None.

    Notes:
    - Haar cascades are sensitive to lighting/pose. We try a couple of passes
      (histogram equalization + a flipped frame) to improve detection.
    """
    # Downscale very large frames to speed up detection and improve stability.
    h0, w0 = frame_bgr.shape[:2]
    scale = 1.0
    if max(h0, w0) > 900:
        scale = 900.0 / float(max(h0, w0))
        frame_bgr = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(f"Failed to load haarcascade: {cascade_path}")

    def _run(gray_img):
        return face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    # Pass 1: equalized grayscale
    faces = _run(gray_eq)

    # Pass 2: try mirrored image (sometimes helps with webcams/angles)
    flipped = False
    if faces is None or len(faces) == 0:
        flipped = True
        gray_eq_f = cv2.flip(gray_eq, 1)
        faces = _run(gray_eq_f)

    if faces is None or len(faces) == 0:
        return None

    # Pick largest face.
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = int(x), int(y), int(w), int(h)

    # If detection happened on flipped image, unflip x.
    if flipped:
        img_w = gray_eq.shape[1]
        x = img_w - (x + w)

    # If we scaled the frame down, scale rect back up to original coordinate space.
    if scale != 1.0:
        inv = 1.0 / scale
        x = int(x * inv)
        y = int(y * inv)
        w = int(w * inv)
        h = int(h * inv)

    return x, y, w, h


def _softmax(logits):
    import numpy as np

    logits = np.asarray(logits, dtype="float32")
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / (exp.sum() + 1e-9)


def _infer_emotion_opencv_dnn(face_bgr, model_path: str):
    """Run OpenCV DNN MobileFaceNet FER model and return (label, scores_dict)."""
    net = cv2.dnn.readNetFromONNX(model_path)

    # Model expects 112x112 RGB, normalized to [-1, 1]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(
        face_resized,
        scalefactor=1.0 / 255.0,
        size=(112, 112),
        mean=(0.5, 0.5, 0.5),
        swapRB=False,
        crop=False,
    )
    blob = (blob - 0.5) / 0.5  # normalize to [-1, 1]

    net.setInput(blob)
    y = net.forward().reshape(-1)

    probs = _softmax(y)
    scores = {FER_LABELS[i]: float(probs[i]) for i in range(len(FER_LABELS))}

    best_i = int(np.argmax(probs))
    best_label = FER_LABELS[best_i]
    return best_label, scores


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None

    _log(out_path, f"started argv={sys.argv!r}")

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        _write_json(out_path, {"error": "camera_open_failed"})
        return 1

    ok = False
    frame = None
    for _ in range(30):  # ~1 second at 30fps
        ok, frame = cap.read()
        if ok and frame is not None:
            pass
    cap.release()

    if not ok or frame is None:
        _write_json(out_path, {"error": "camera_read_failed"})
        return 1

    # Always capture debug frames for troubleshooting.
    raw_path, processed_path = _debug_frame_paths(out_path)
    try:
        #cv2.imwrite(raw_path, frame)
        pass
    except Exception:
        pass

    # Brighten the frame for more reliable face detection.
    frame_proc = _brighten_frame(frame)

    try:
        #cv2.imwrite(processed_path, frame_proc)
        pass
    except Exception:
        pass

    # Detect face
    face_rect = _detect_primary_face_bgr(frame_proc)
    if face_rect is None:
        _write_json(
            out_path,
            {
                "error": "no_face_detected",
                "hint": "Try better lighting, face the camera, and move closer.",
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "debug_frame_raw": raw_path,
                "debug_frame": processed_path,
            },
        )
        return 0

    x, y, w, h = face_rect
    face_bgr = frame_proc[y : y + h, x : x + w]

    try:
        #cv2.imwrite(processed_path.replace(".frame.jpg", ".face_crop.jpg"), face_bgr)
        pass
    except Exception:
        pass

    # Try OpenCV MobileFaceNet emotion model if present
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "facial_expression_recognition_mobilefacenet_2022july.onnx")

    try:
        if os.path.exists(model_path):
            emotion, scores = _infer_emotion_opencv_dnn(face_bgr, model_path)
            payload = {
                "emotion": emotion,
                "scores": scores,
                "model": "opencv_dnn:mobilefacenet_fer",
            }
        else:
            payload = {
                "error": "model_missing",
                "detail": "Place the OpenCV FER model at models/facial_expression_recognition_mobilefacenet_2022july.onnx",
            }

        _write_json(out_path, payload)
        return 0

    except Exception as e:
        _log(out_path, "exception: " + repr(e))
        _log(out_path, traceback.format_exc())
        _write_json(
            out_path,
            {
                "error": "inference_failed",
                "detail": str(e),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())