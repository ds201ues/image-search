import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://imgsearch:imgsearch@localhost:5433/imgsearch")
DATA_DIR = ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbs"
MODEL_DIR = ROOT / "models"

VLM_MODEL = os.getenv("VLM_MODEL", "gpt-6-luna")
VLM_PRICE_IN = float(os.getenv("VLM_PRICE_IN", "0.10"))
VLM_PRICE_OUT = float(os.getenv("VLM_PRICE_OUT", "0.50"))
VLM_MAX_SIDE = 768
# reasoning tokens bill as output; captioning needs little. Empty string = do not send the parameter.
VLM_REASONING = os.getenv("VLM_REASONING", "low")
VLM_DETAIL = os.getenv("VLM_DETAIL", "auto") # low | high | auto

SIGLIP_MODEL = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-256")
CLIP_DIM = 768
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.5"))

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TEXT_DIM = 384
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "256"))
QUERY_PARSE = os.getenv("QUERY_PARSE", "1") == "1" # LLM query understanding (cached); off = plain query
