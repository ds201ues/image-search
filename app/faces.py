"""Face detection (OpenCV YuNet, MIT) + recognition (AuraFace glintr100, Apache-2.0)."""
import threading
import cv2
import numpy as np
import onnxruntime as ort
import requests
from .config import MODEL_DIR

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
AURA_URL = "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"
# ArcFace 112x112 alignment template: image-left eye, image-right eye, nose, mouth left, mouth right
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
 [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
MIN_FACE = 36 # px, smaller faces are unreliable for recognition
DET_SIDE = 1280 # detect on images resized to this long side

_lock = threading.Lock()
_det = _rec = None


def _fetch(url, name):
 path = MODEL_DIR / name
 if not path.exists():
 MODEL_DIR.mkdir(parents=True, exist_ok=True)
 print(f"downloading {name} ...")
 r = requests.get(url, timeout=300, allow_redirects=True)
 r.raise_for_status()
 path.write_bytes(r.content)
 return path


def _load():
 global _det, _rec
 with _lock:
 if _det is None:
 _det = cv2.FaceDetectorYN.create(str(_fetch(YUNET_URL, "yunet.onnx")), "", (320, 320),
 score_threshold=0.75, nms_threshold=0.3, top_k=50)
 _rec = ort.InferenceSession(str(_fetch(AURA_URL, "auraface_glintr100.onnx")),
 providers=["CPUExecutionProvider"])
 return _det, _rec


def detect_and_embed(bgr: np.ndarray):
 """Return list of dicts: bbox (orig px), det_score, emb (512, L2-normalised)."""
 det, rec = _load()
 h, w = bgr.shape[:2]
 s = min(1.0, DET_SIDE / max(h, w))
 small = cv2.resize(bgr, (int(w * s), int(h * s))) if s < 1 else bgr
 with _lock: # cv2 detector object is not thread-safe
 det.setInputSize((small.shape[1], small.shape[0]))
 _, found = det.detect(small)
 out, crops = [], []
 for f in (found if found is not None else []):
 f = f / s
 x, y, fw, fh = f[:4]
 if min(fw, fh) < MIN_FACE:
 continue
 lm = f[4:14].reshape(5, 2).astype(np.float32)
 M, _ = cv2.estimateAffinePartial2D(lm, TEMPLATE, method=cv2.LMEDS)
 if M is None:
 continue
 crops.append(cv2.warpAffine(bgr, M, (112, 112), borderValue=0))
 out.append({"bbox": [int(x), int(y), int(fw), int(fh)], "det_score": float(f[14])})
 if crops:
 # model is exported with batch size 1, so embed one face at a time
 name = rec.get_inputs()[0].name
 embs = np.concatenate([rec.run(None, {name: cv2.dnn.blobFromImage(
 c, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)})[0] for c in crops])
 embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
 for o, e in zip(out, embs):
 o["emb"] = e.astype(np.float32)
 return out


def read_bgr(path) -> np.ndarray:
 return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
