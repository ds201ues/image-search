"""Candidate image-text models behind one interface: embed_images(list[PIL]) -> np, embed_text(list[str]) -> np."""
import numpy as np, torch
torch.set_num_threads(5)


def _n(x):
 x = x.float()
 return (x / x.norm(dim=-1, keepdim=True)).numpy()


class SigLIP2:
 def __init__(self, mid="google/siglip2-so400m-patch16-256"):
 from transformers import AutoModel, AutoProcessor
 self.m = AutoModel.from_pretrained(mid).eval(); self.p = AutoProcessor.from_pretrained(mid)
 self.dim = self.m.config.text_config.hidden_size if hasattr(self.m.config, "text_config") else 1152

 @torch.inference_mode()
 def embed_images(self, ims):
 out = self.m.get_image_features(**self.p(images=[i.convert("RGB") for i in ims], return_tensors="pt"))
 return _n(out if isinstance(out, torch.Tensor) else out.pooler_output)

 @torch.inference_mode()
 def embed_text(self, ts):
 out = self.m.get_text_features(**self.p(text=[t.lower() for t in ts], padding="max_length", max_length=64,
 truncation=True, return_tensors="pt"))
 return _n(out if isinstance(out, torch.Tensor) else out.pooler_output)


class FGCLIP2:
 def __init__(self, mid="qihoo360/fg-clip2-base"):
 from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
 self.m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).eval()
 self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
 self.ip = AutoImageProcessor.from_pretrained(mid, trust_remote_code=True)
 # transformers 5.x leaves these non-persistent buffers / plain tensors on the meta device: rebuild them
 e = self.m.text_model.embeddings; c = self.m.config.text_config
 e.position_ids = torch.arange(c.longtext_len).unsqueeze(0)
 e.mask1 = torch.zeros([c.longtext_len, 1]); e.mask1[:c.keep_len] = 1
 e.mask2 = torch.zeros([c.longtext_len, 1]); e.mask2[c.keep_len:] = 1
 self.dim = 768

 @torch.inference_mode()
 def embed_images(self, ims):
 vs = []
 for im in ims: # NaFlex processor: variable patch counts, so one image at a time
 x = self.ip(images=im.convert("RGB"), max_num_patches=256, return_tensors="pt")
 vs.append(self.m.get_image_features(**x))
 v = torch.cat(vs); self.dim = v.shape[-1]
 return _n(v)

 @torch.inference_mode()
 def embed_text(self, ts, long=False):
 """short walk (64 tokens) for typed queries; long walk (196) for sentence-length text."""
 ml, wt = (196, "long") if long else (64, "short")
 x = self.tok([t.lower() for t in ts], padding="max_length", max_length=ml, truncation=True, return_tensors="pt")
 return _n(self.m.get_text_features(input_ids=x["input_ids"], walk_type=wt))


MODELS = {"fgclip2_base": FGCLIP2, "siglip2_so400m": SigLIP2}
