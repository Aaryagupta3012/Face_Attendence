"""Face detection (Haar cascade) + recognition (LBPH) using OpenCV only — no dlib needed."""
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "lbph_model.yml"
FACE_SIZE = (200, 200)

CASCADE_PATH = Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if _cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")


def bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) boxes, largest first."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boxes = [tuple(int(v) for v in f) for f in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, FACE_SIZE)
    return cv2.equalizeHist(face)


def save_face_sample(student_id: int, face_gray: np.ndarray) -> Path:
    folder = FACES_DIR / str(student_id)
    folder.mkdir(parents=True, exist_ok=True)
    idx = len(list(folder.glob("*.png")))
    path = folder / f"{idx:03d}.png"
    cv2.imwrite(str(path), face_gray)
    return path


def sample_count(student_id: int) -> int:
    folder = FACES_DIR / str(student_id)
    return len(list(folder.glob("*.png"))) if folder.exists() else 0


def train_model() -> int:
    """Train LBPH on all saved samples. Returns number of images used (0 = nothing to train)."""
    images, labels = [], []
    if FACES_DIR.exists():
        for folder in FACES_DIR.iterdir():
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            for img_path in folder.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(int(folder.name))
    if not images:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        return 0
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    recognizer.write(str(MODEL_PATH))
    return len(images)


def load_model():
    if not MODEL_PATH.exists():
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    return recognizer


def recognize(recognizer, face_gray: np.ndarray) -> tuple[int, float]:
    """Returns (student_id, distance). Lower distance = better match. Typical good matches < 60."""
    label, dist = recognizer.predict(face_gray)
    return int(label), float(dist)


def draw_boxes(bgr: np.ndarray, boxes, labels: list[str]) -> np.ndarray:
    out = bgr.copy()
    for (x, y, w, h), text in zip(boxes, labels):
        color = (0, 200, 0) if not text.startswith("Unknown") else (0, 0, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(out, (x, y + h), (x + w, y + h + 28), color, -1)
        cv2.putText(out, text, (x + 4, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
