"""SigLIP 2 image/text embeddings (one shared vector space). CPU by default."""
import threading
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from .config import SIGLIP_MODEL

_lock = threading.Lock()
_model = _proc = None
torch.set_num_threads(max(1, torch.get_num_threads()))


def _load():
 global _model, _proc
 with _lock:
 if _model is None:
 _model = AutoModel.from_pretrained(SIGLIP_MODEL).eval()
 _proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)
 return _model, _proc


def _norm(x: torch.Tensor) -> np.ndarray:
 x = x.float()
 return (x / x.norm(dim=-1, keepdim=True)).numpy()


def _features(out):
 # newer transformers return a ModelOutput, older return a tensor
 return out if isinstance(out, torch.Tensor) else out.pooler_output


@torch.inference_mode()
def embed_images(images: list[Image.Image]) -> np.ndarray:
 m, p = _load()
 inputs = p(images=[im.convert("RGB") for im in images], return_tensors="pt")
 return _norm(_features(m.get_image_features(**inputs)))


@torch.inference_mode()
def embed_text(texts: list[str]) -> np.ndarray:
 m, p = _load()
 # SigLIP 2 was trained on lowercased text padded to 64 tokens
 inputs = p(text=[t.lower() for t in texts], padding="max_length", max_length=64,
 truncation=True, return_tensors="pt")
 return _norm(_features(m.get_text_features(**inputs)))
