"""Source metadata from the files themselves: IPTC captions, EXIF dates, XMP, PDF-wrapped photos.

The archive files carry photographer-written captions (IPTC 2:120) in ~69% of cases, plus headline,
keywords, city, byline, credit and a creation date. Dates are resolved from EXIF/IPTC/XMP/caption text,
never guessed by a model, and must fall in a sane window (camera clocks that were never set give 1980/2034/2099).
"""
import datetime as dt, io, re
from pathlib import Path
from PIL import Image, IptcImagePlugin, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".mpo"}
PDF_EXTS = {".pdf"}
MIN_YEAR, MAX_YEAR = 1990, dt.date.today().year
# IPTC IIM dataset numbers we care about
IPTC = {(2, 120): "caption", (2, 105): "headline", (2, 25): "keywords", (2, 90): "city", (2, 95): "state",
 (2, 101): "country", (2, 80): "byline", (2, 85): "byline_title", (2, 110): "credit", (2, 115): "source",
 (2, 55): "date_created", (2, 60): "time_created", (2, 5): "object_name", (2, 40): "instructions",
 (2, 20): "categories", (2, 116): "copyright"}


def _dec(b):
 if isinstance(b, list):
 return [_dec(x) for x in b]
 if isinstance(b, bytes):
 for enc in ("utf-8", "latin-1"):
 try:
 s = b.decode(enc); break
 except UnicodeDecodeError:
 continue
 else:
 s = str(b)
 s = s.replace("\x00", "").strip()
 return s


def read_iptc(im) -> dict:
 out = {}
 try:
 raw = IptcImagePlugin.getiptcinfo(im) or {}
 except Exception:
 return out
 for k, name in IPTC.items():
 if k in raw:
 v = _dec(raw[k])
 if name in ("keywords", "categories"):
 v = [x for x in (v if isinstance(v, list) else [v]) if x]
 elif isinstance(v, list):
 v = " ".join(x for x in v if x)
 if v:
 out[name] = v
 return out


def read_exif(im) -> dict:
 out = {}
 try:
 ex = im.getexif()
 except Exception:
 return out
 if 271 in ex or 272 in ex:
 out["camera"] = " ".join(str(ex.get(t, "")).replace("\x00", "").strip() for t in (271, 272)).strip()
 if 306 in ex:
 out["exif_datetime"] = str(ex[306]).replace("\x00", "").strip()
 try:
 ifd = ex.get_ifd(0x8769)
 if 36867 in ifd:
 out["exif_original"] = str(ifd[36867]).replace("\x00", "").strip()
 if 36868 in ifd:
 out["exif_digitized"] = str(ifd[36868]).replace("\x00", "").strip()
 except Exception:
 pass
 for k in ("exif_datetime", "exif_original", "exif_digitized"):
 if k in out and not re.search(r"\d{4}", out[k]):
 out.pop(k)
 return out


_XMP_FIELDS = {"photoshop:DateCreated": "xmp_date", "photoshop:City": "xmp_city", "photoshop:Headline": "xmp_headline",
 "photoshop:Credit": "xmp_credit", "dc:creator": "xmp_creator"}


def read_xmp(im) -> dict:
 xmp = im.info.get("xmp")
 if not xmp:
 return {}
 s = xmp.decode("utf-8", errors="ignore") if isinstance(xmp, bytes) else str(xmp)
 out = {}
 for tag, name in _XMP_FIELDS.items():
 m = re.search(rf"<{tag}>(.*?)</{tag}>", s, re.S) or re.search(rf'{tag}="([^"]*)"', s)
 if m:
 v = re.sub(r"<[^>]+>", " ", m.group(1)).strip()
 if v:
 out[name] = " ".join(v.split())
 m = re.search(r"<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>", s, re.S)
 if m and m.group(1).strip():
 out["xmp_description"] = " ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split())
 kws = re.findall(r"<dc:subject>(.*?)</dc:subject>", s, re.S)
 if kws:
 out["xmp_keywords"] = [k.strip() for k in re.findall(r"<rdf:li>(.*?)</rdf:li>", kws[0]) if k.strip()]
 return out


def _sane(d: dt.datetime | None):
 return d if d and MIN_YEAR <= d.year <= MAX_YEAR else None


def _parse_dt(s: str | None):
 if not s:
 return None
 s = s.strip()
 for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y%m%d", "%Y:%m:%d"):
 try:
 return dt.datetime.strptime(s[:len(dt.datetime.now().strftime(fmt))], fmt)
 except ValueError:
 continue
 m = re.match(r"(\d{4})[-:/](\d{1,2})[-:/](\d{1,2})", s)
 if m:
 try:
 return dt.datetime(int(m[1]), int(m[2]), int(m[3]))
 except ValueError:
 return None
 return None


_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"


def date_in_text(text: str | None):
 """Dates photographers type into captions: 16-04-2019, 16/04/19, 16.04.2019, 'April 16, 2019', '16 Apr 2019'."""
 if not text:
 return None
 for m in re.finditer(r"\b(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})\b", text):
 d, mo, y = int(m[1]), int(m[2]), int(m[3])
 y = y + 2000 if y < 100 else y
 for dd, mm in ((d, mo), (mo, d)):
 try:
 return _sane(dt.datetime(y, mm, dd))
 except ValueError:
 continue
 m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})[a-z]*\.?,?\s+(\d{{4}})\b", text, re.I) \
 or re.search(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text, re.I)
 if m:
 g = m.groups()
 day, mon, year = (g[0], g[1], g[2]) if g[0].isdigit() else (g[1], g[0], g[2])
 try:
 return _sane(dt.datetime.strptime(f"{int(day)} {mon[:3].title()} {year}", "%d %b %Y"))
 except ValueError:
 return None
 return None


def resolve_date(meta: dict):
 """(datetime or None, source). Order: camera original, IPTC created, XMP created, camera file time, caption text."""
 order = [("exif_original", meta.get("exif_original")), ("exif_digitized", meta.get("exif_digitized")),
 ("iptc", (meta.get("date_created") or "") + (" " + meta["time_created"][:6] if meta.get("time_created") else "")),
 ("xmp", meta.get("xmp_date")), ("exif_file", meta.get("exif_datetime"))]
 for src, s in order:
 d = _sane(_parse_dt(s)) if s and s.strip() else None
 if d:
 return d, src
 d = date_in_text(meta.get("caption")) or date_in_text(meta.get("headline"))
 if d:
 return d, "caption_text"
 return None, None


def pdf_photo(path: Path, out_dir: Path) -> Path | None:
 """Extract the largest embedded image of a one-page print-layout PDF as a JPEG. Returns the JPEG path."""
 import pymupdf as fitz
 out = out_dir / f"{path.stem}.jpg"
 if out.exists():
 return out
 doc = fitz.open(path)
 best, best_px = None, 0
 for page in doc:
 for info in page.get_images(full=True):
 xref = info[0]
 pix = fitz.Pixmap(doc, xref)
 if pix.width * pix.height > best_px:
 best, best_px = pix, pix.width * pix.height
 if best is None:
 return None
 if best.n - best.alpha >= 4 or best.colorspace is None or best.colorspace.n != 3:
 best = fitz.Pixmap(fitz.csRGB, best)
 out_dir.mkdir(parents=True, exist_ok=True)
 best.save(str(out), output="jpeg", jpg_quality=92)
 return out


def read_all(path: Path) -> dict:
 """Everything we can read from a file without a model. Opens lazily; does not decode pixels."""
 meta = {}
 with Image.open(path) as im:
 meta.update(read_exif(im)); meta.update(read_iptc(im)); meta.update(read_xmp(im))
 meta["width"], meta["height"], meta["mode"] = im.width, im.height, im.mode
 # XMP fallbacks when IPTC is missing
 for src, dst in (("xmp_description", "caption"), ("xmp_headline", "headline"), ("xmp_city", "city"),
 ("xmp_credit", "credit"), ("xmp_creator", "byline"), ("xmp_keywords", "keywords")):
 if src in meta and dst not in meta:
 meta[dst] = meta[src]
 if meta.get("caption") in ("no comments", "no", "-", "."):
 meta.pop("caption")
 d, src = resolve_date(meta)
 meta["date"], meta["date_source"] = d, src
 return meta


def open_scaled(path: Path, long_side: int) -> Image.Image:
 """Fast JPEG decode at reduced size (DCT scaling), then exact thumbnail. RGB."""
 im = Image.open(path)
 try:
 im.draft("RGB", (long_side, long_side))
 except Exception:
 pass
 im = im.convert("RGB")
 im.thumbnail((long_side, long_side), Image.LANCZOS)
 return im
