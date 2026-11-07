"""Ingestion layer: load documents -> clean -> chunk -> embed -> store in ChromaDB.

Run once, and again whenever a file in documents/ is added or edited:
    python ingest.py
Check how a file will be read before indexing it:
    python ingest.py --preview documents/bachelor_regulations.txt
"""
import re
import sys
import unicodedata
from pathlib import Path

import chromadb
import pymupdf as fitz  # PyMuPDF: reads PDF pages (for PDFs uploaded later)
from chromadb.utils import embedding_functions

# ================================================================ SETTINGS ===
# Change a value here, then re-run `python ingest.py` (the chunks are rebuilt).
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
DB_DIR = BASE_DIR / "chroma_db"       # ChromaDB saves the vectors here (on disk)
COLLECTION = "ppu_academic"

CHUNK_SIZE = 400      # max characters per chunk (800 -> 400 raised retrieval hits; see README)
OVERLAP_LINES = 1     # last line of a chunk is repeated at the start of the next one

# Multilingual model (Arabic + English). Chroma's default model is English-only.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Some Arabic PDFs store each line in reversed word order; set True if --preview shows that.
PDF_REVERSE_RTL = False


# ============================================================ ARABIC TEXT ===
_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰ]")  # harakat
_ARABIC = re.compile(r"[؀-ۿ]")
# Arabic-Indic digits (٠١٢ and ۰۱۲) -> 012, so "١٨ ساعة" matches "18 ساعة"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def clean_text(text: str) -> str:
    """Clean text before chunking.

    NFKC turns Arabic "presentation forms" from PDFs (e.g. ﻣﻌﺎﻳﲑ) into normal
    letters (معايير). We also unify digits and remove tatweel and extra spaces.
    """
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS)
    text = text.replace("ـ", "")  # tatweel
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_reversed_lines(text: str) -> str:
    """Reverse the word order of mostly-Arabic lines (for PDFs stored in visual order)."""
    out = []
    for line in text.split("\n"):
        letters = [c for c in line if c.isalpha()]
        arabic = sum(bool(_ARABIC.match(c)) for c in letters)
        out.append(" ".join(reversed(line.split())) if letters and arabic / len(letters) > 0.5 else line)
    return "\n".join(out)


def normalize_ar(text: str) -> str:
    """Aggressive normalization used ONLY for keyword (BM25) matching:
    "إجازة" / "اجازه" / "الإجازة" all match. Stored text stays original."""
    text = clean_text(text).lower()
    text = _TASHKEEL.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", normalize_ar(text))
    # strip the definite article so "الخوارزميات" matches "خوارزميات"
    return [t[2:] if t.startswith("ال") and len(t) > 4 else t for t in tokens]


# ---------------------------------------------------------------- loading ---
def parse_header(text: str) -> tuple[dict, str]:
    """Read the 'KEY: value' lines before '---' at the top of a .txt file.

    Example header:  SOURCE_TITLE / SOURCE_URL / DEGREE / VERIFICATION
    """
    meta = {}
    if "\n---" in text:
        head, body = text.split("\n---", 1)
        for line in head.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip().lower()] = value.strip()
        return meta, body
    return meta, text


def load_txt(path: Path) -> list[dict]:
    meta, body = parse_header(path.read_text(encoding="utf-8"))
    body = clean_text(body)
    if not body:
        return []
    return [{
        "text": body,
        "source": path.name,
        "title": meta.get("source_title", path.stem),
        "url": meta.get("source_url", ""),
        "degree": meta.get("degree", "عام"),
        "page": 0,  # 0 = not a PDF page
    }]


def load_pdf(path: Path) -> list[dict]:
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if PDF_REVERSE_RTL:
                text = fix_reversed_lines(text)
            if text:
                pages.append({"text": text, "source": path.name, "title": path.stem,
                              "url": "", "degree": "عام", "page": i})
    if not pages:
        print(f"  ! {path.name}: no text found (scanned PDF? needs OCR)")
    return pages


def load_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    if path.suffix.lower() in (".txt", ".md"):
        return load_txt(path)
    return []


def load_docs(folder: Path) -> list[dict]:
    records = []
    for path in sorted(folder.iterdir()):
        records += load_file(path)
    return records


# --------------------------------------------------------------- chunking ---
def chunk(text: str, size: int = CHUNK_SIZE, overlap_lines: int = OVERLAP_LINES) -> list[str]:
    """Line-aware chunking.

    In our documents every article ("المادة (7)") or course ("المساق 5050") is one
    line, so we pack WHOLE lines into a chunk until it reaches `size` characters:
    an article or a course row is never cut in half. Section titles like
    "[متطلبات التخصص الإجبارية ...]" are repeated at the top of every chunk of that
    section, so each chunk knows which part of the plan it comes from.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks, current, section = [], [], ""
    for line in lines:
        if line.startswith("[") and line.endswith("]"):  # a new section title
            if current:
                chunks.append("\n".join(current))
            section, current = line, [line]
            continue
        if current and len("\n".join(current)) + len(line) + 1 > size:
            chunks.append("\n".join(current))
            carry = current[-overlap_lines:] if overlap_lines else []
            carry = [c for c in carry if c != section]
            current = ([section] if section else []) + carry
        current.append(line)
    if current and current != [section]:
        chunks.append("\n".join(current))
    return chunks


def make_chunks(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        for i, c in enumerate(chunk(rec["text"])):
            out.append({
                "id": f"{rec['source']}-p{rec['page']}-{i}",
                "text": c,
                "metadata": {k: rec[k] for k in ("source", "title", "url", "degree", "page")},
            })
    return out


# ---------------------------------------------------------------- storing ---
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_collection(reset: bool = False):
    """ChromaDB PersistentClient saves vectors to disk, so we embed once, not every run."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},  # distances in [0, 2], easy to threshold
    )


def add_chunks(col, chunks: list[dict], batch: int = 64):
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.upsert(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
        )


def ingest_file(path: Path) -> int:
    """Add one new file to the existing index (used by the upload button)."""
    chunks = make_chunks(load_file(path))
    add_chunks(get_collection(), chunks)
    return len(chunks)


def build_index() -> int:
    records = load_docs(DOCS_DIR)
    chunks = make_chunks(records)
    col = get_collection(reset=True)
    add_chunks(col, chunks)
    print(f"Loaded {len(records)} documents/pages from {DOCS_DIR.name}/ -> stored {col.count()} chunks")
    return col.count()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) == 3 and sys.argv[1] == "--preview":
        recs = load_file(Path(sys.argv[2]))
        chs = make_chunks(recs)
        print(f"{len(recs)} record(s) -> {len(chs)} chunks. Metadata: {chs[0]['metadata'] if chs else '-'}\n")
        for c in chs[:2]:
            print(f"--- {c['id']} ({len(c['text'])} chars)\n{c['text']}\n")
    else:
        build_index()
