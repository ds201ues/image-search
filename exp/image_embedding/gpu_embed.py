# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision", "transformers>=4.50", "pillow", "numpy", "sentencepiece", "protobuf", "open_clip_torch>=3.0"]
# ///
"""Embed thumbnails + query strings with candidate image-text models on a CUDA box. No Postgres.

 uv run gpu_embed.py --thumbs data/thumbs --queries queries.txt --out out --models fgclip2_base siglip2_so400m pe_core_l

Writes out/ids.json, out/<model>.npy (N x D, L2-normalised float32), out/<model>_queries.npy (Q x D),
out/timings.json (ms per image / per query, measured on the GPU with sync).
Rsync `out/` back and run `python -m exp.image_embedding.load_npy out` locally.
"""
import argparse, json, time, gc
from pathlib import Path
import numpy as np, torch
from PIL import Image

dev = "cuda" if torch.cuda.is_available() else "cpu"


def _n(x):
 x = x.float()
 return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()


class SigLIP2:
 def __init__(self, mid="google/siglip2-so400m-patch16-256"):
 from transformers import AutoModel, AutoProcessor
 self.m = AutoModel.from_pretrained(mid, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
 self.p = AutoProcessor.from_pretrained(mid)

 @torch.inference_mode()
 def images(self, ims):
 x = self.p(images=[i.convert("RGB") for i in ims], return_tensors="pt").to(dev)
 x["pixel_values"] = x["pixel_values"].to(self.m.dtype)
 o = self.m.get_image_features(**x)
 return _n(o if isinstance(o, torch.Tensor) else o.pooler_output)

 @torch.inference_mode()
 def text(self, ts):
 x = self.p(text=[t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
 o = self.m.get_text_features(**x)
 return _n(o if isinstance(o, torch.Tensor) else o.pooler_output)


class FGCLIP2:
 """qihoo360/fg-clip2-base. 256 patches for cost parity with siglip2-base-256. Short walk for typed queries."""
 def __init__(self, mid="qihoo360/fg-clip2-base"):
 from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
 self.m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).to(dev).eval()
 self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
 self.ip = AutoImageProcessor.from_pretrained(mid, trust_remote_code=True)
 e = self.m.text_model.embeddings; c = self.m.config.text_config # transformers 5.x leaves these on meta
 e.position_ids = torch.arange(c.longtext_len).unsqueeze(0).to(dev)
 e.mask1 = torch.zeros([c.longtext_len, 1], device=dev); e.mask1[:c.keep_len] = 1
 e.mask2 = torch.zeros([c.longtext_len, 1], device=dev); e.mask2[c.keep_len:] = 1

 @torch.inference_mode()
 def images(self, ims):
 x = self.ip(images=[i.convert("RGB") for i in ims], max_num_patches=256, return_tensors="pt").to(dev)
 return _n(self.m.get_image_features(**x))

 @torch.inference_mode()
 def text(self, ts):
 x = self.tok([t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
 return _n(self.m.get_text_features(input_ids=x["input_ids"], walk_type="short"))


class PECoreL:
 def __init__(self, mid="hf-hub:timm/PE-Core-L-14-336"):
 import open_clip
 self.m, _, self.pre = open_clip.create_model_and_transforms(mid)
 self.m = self.m.to(dev).eval(); self.tok = open_clip.get_tokenizer(mid)

 @torch.inference_mode()
 def images(self, ims):
 x = torch.stack([self.pre(i.convert("RGB")) for i in ims]).to(dev)
 return _n(self.m.encode_image(x, normalize=True))

 @torch.inference_mode()
 def text(self, ts):
 return _n(self.m.encode_text(self.tok(ts).to(dev), normalize=True))


MODELS = {"fgclip2_base": FGCLIP2, "siglip2_so400m": SigLIP2, "pe_core_l": PECoreL,
 "siglip2_base": lambda: SigLIP2("google/siglip2-base-patch16-256")}


def sync():
 if dev == "cuda":
 torch.cuda.synchronize()


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--thumbs", default="data/thumbs"); ap.add_argument("--queries", default="queries.txt")
 ap.add_argument("--out", default="out"); ap.add_argument("--models", nargs="+", default=["fgclip2_base", "siglip2_so400m"])
 ap.add_argument("--batch", type=int, default=64)
 ap.add_argument("--ids", default=None, help="json list of image ids to embed (default: every jpg in --thumbs)")
 a = ap.parse_args()
 out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
 files = sorted(Path(a.thumbs).glob("*.jpg"))
 if a.ids:
 want = set(json.load(open(a.ids))); files = [p for p in files if p.stem in want]
 missing = want - {p.stem for p in files}
 if missing:
 print(f"WARNING {len(missing)} ids have no thumbnail", flush=True)
 ids = [p.stem for p in files]
 json.dump(ids, open(out / "ids.json", "w"))
 queries = [l.strip() for l in open(a.queries) if l.strip()] if Path(a.queries).exists() else []
 timings = json.load(open(out / "timings.json")) if (out / "timings.json").exists() else {}
 print(f"{dev}: {len(files)} images, {len(queries)} queries, models {a.models}", flush=True)
 for key in a.models:
 if (out / f"{key}.npy").exists():
 print(f"{key}: exists, skipping"); continue
 t0 = time.perf_counter(); m = MODELS[key](); load_s = time.perf_counter() - t0
 m.images([Image.open(files[0])]); sync()
 vecs, t0 = [], time.perf_counter()
 for i in range(0, len(files), a.batch):
 vecs.append(m.images([Image.open(p) for p in files[i:i + a.batch]]))
 if (i // a.batch) % 10 == 0:
 print(f" {key} {i + a.batch}/{len(files)}", flush=True)
 sync(); img_ms = 1000 * (time.perf_counter() - t0) / len(files)
 V = np.concatenate(vecs).astype(np.float32); np.save(out / f"{key}.npy", V)
 q_ms = None
 if queries:
 m.text(queries[:1]); sync(); t0 = time.perf_counter()
 Q = np.concatenate([m.text(queries[i:i + 64]) for i in range(0, len(queries), 64)]).astype(np.float32)
 sync(); q_ms = 1000 * (time.perf_counter() - t0) / len(queries)
 np.save(out / f"{key}_queries.npy", Q)
 timings[key] = {"dim": int(V.shape[1]), "img_ms_gpu": round(img_ms, 2), "query_ms_gpu": round(q_ms, 2) if q_ms else None,
 "load_s": round(load_s, 1), "device": dev, "n": len(files)}
 json.dump(timings, open(out / "timings.json", "w"), indent=1)
 print(key, timings[key], flush=True)
 del m; gc.collect()
 if dev == "cuda":
 torch.cuda.empty_cache()


if __name__ == "__main__":
 main()
