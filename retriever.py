"""Retrieval layer: find the best chunks in ChromaDB for a question.

Semantic search (vectors) finds chunks with the same MEANING even when the words
differ (dialect, rewording). Keyword search (BM25) finds exact terms such as course
codes (5050) and article numbers, which vectors often miss. Hybrid mode merges both
lists with Reciprocal Rank Fusion (RRF). Metadata filtering keeps Diploma / Master
rules apart from Bachelor rules.

Debug and tune from the terminal (prints the distances):
    python retriever.py "كم ساعة مادة تركيب البيانات؟"
"""
import sys
from functools import lru_cache

from rank_bm25 import BM25Okapi

from ingest import get_collection, normalize_ar, tokenize

# ================================================================ SETTINGS ===
# These take effect immediately (no need to re-run ingest.py).
TOP_K = 6            # chunks sent to the model per question (4 -> 6: multi-article questions)
# Cosine distance: 0 = identical, 2 = opposite. If even the closest chunk is
# further than this, the question is not about our documents -> refuse.
MAX_DISTANCE = 0.62  # measured: our questions 0.23-0.56, poem 0.64, World Cup 0.76
USE_HYBRID = True    # semantic + keyword search merged with RRF

# Words that tell us which degree the student is asking about (metadata filter).
DEGREE_WORDS = {"دبلوم": "دبلوم", "ماجستير": "ماجستير", "ماستر": "ماجستير",
                "diploma": "دبلوم", "master": "ماجستير"}

# Dialect / everyday words -> the formal words used in the regulations.
# The question is EXPANDED with these words before searching (it is not replaced).
# To support a new expression, just add one line here.
DIALECT_TERMS = {
    "غبت": "الغياب الحضور الحرمان",
    "اغيب": "الغياب الحضور الحرمان",
    "غياب": "الحضور الحرمان",
    "منذر": "التحذير الإنذار الأكاديمي",
    "انذار": "التحذير الأكاديمي",
    "معدلي": "المعدل التراكمي التحذير العبء الدراسي",
    "اسحب": "الانسحاب السحب والإضافة",
    "انسحب": "الانسحاب السحب",
    "اسجل": "العبء الدراسي الساعات المعتمدة الحد الأعلى",
    "رسبت": "الرسوب إعادة المساق",
    "اعيد": "إعادة المساق",
    "اتخرج": "التخرج",
    "اأجل": "التأجيل",
    "بقدر": "",
    "قديش": "كم",
    "شو": "ما",
    # English questions -> the Arabic words used in the documents
    "absent": "الغياب الحضور الحرمان",
    "absence": "الغياب الحضور الحرمان",
    "miss": "الغياب الحضور الحرمان",
    "attendance": "الحضور الغياب الحرمان",
    "withdraw": "الانسحاب السحب والإضافة",
    "warning": "التحذير الإنذار الأكاديمي",
    "probation": "التحذير الإنذار الأكاديمي",
    "register": "العبء الدراسي الساعات المعتمدة الحد الأعلى",
    "graduate": "التخرج",
    "repeat": "إعادة المساق",
    "postpone": "التأجيل",
    "passing": "النجاح علامة",
    "prerequisite": "المتطلب سابق",
}


# If the question is about an EXAM, "غبت" means missing an exam (not lectures),
# so these words replace the attendance expansion.
EXAM_WORDS = ("امتحان", "امتحانات", "فاينل", "نهائي", "ميدتيرم", "اختبار", "exam", "final", "midterm")
EXAM_TERMS = "الامتحان الفائت الامتحان النهائي التكميلي العذر ظرف قاهر قضية طلابية"
ATTENDANCE_WORDS = ("غبت", "اغيب", "غياب", "absent", "absence", "miss", "attendance")


def expand_question(question: str) -> str:
    """Append formal words for any dialect word found in the question."""
    q = normalize_ar(question)
    about_exam = any(normalize_ar(w) in q for w in EXAM_WORDS)
    extra = [EXAM_TERMS] if about_exam else []
    for word, formal in DIALECT_TERMS.items():
        if not formal or normalize_ar(word) not in q:
            continue
        if about_exam and word in ATTENDANCE_WORDS:
            continue  # exam absence, not lecture attendance
        extra.append(formal)
    return f"{question} {' '.join(extra)}" if extra else question


@lru_cache(maxsize=1)
def _collection():
    return get_collection()


@lru_cache(maxsize=1)
def _bm25_index():
    """Build the keyword index from everything stored in ChromaDB (once per run)."""
    data = _collection().get(include=["documents", "metadatas"])
    if not data["ids"]:
        return None, []
    bm25 = BM25Okapi([tokenize(d) for d in data["documents"]])
    return bm25, list(zip(data["ids"], data["documents"], data["metadatas"]))


def refresh():
    """Call after adding documents so the keyword index is rebuilt."""
    _bm25_index.cache_clear()


BACHELOR_WORDS = ("بكالوريوس", "بكالوريس", "bachelor")


def detect_degree(question: str) -> str | None:
    """Return the ONE degree the question is about, or None (no filter).
    If several degrees are named (e.g. "diploma vs bachelor"), we must not filter."""
    q = normalize_ar(question)
    found = {degree for word, degree in DEGREE_WORDS.items() if normalize_ar(word) in q}
    if any(normalize_ar(w) in q for w in BACHELOR_WORDS):
        found.add("بكالوريوس")
    return found.pop() if len(found) == 1 and "بكالوريوس" not in found else None


def semantic_search(question: str, k: int, degree: str | None = None) -> list[dict]:
    where = {"degree": {"$in": [degree, "عام"]}} if degree else None
    res = _collection().query(query_texts=[question], n_results=k, where=where)
    return [
        {"id": i, "text": d, "metadata": m, "distance": dist}
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                 res["metadatas"][0], res["distances"][0])
    ]


def keyword_search(question: str, k: int, degree: str | None = None) -> list[dict]:
    bm25, rows = _bm25_index()
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in order:
        if scores[i] <= 0 or len(hits) == k:
            break
        if degree and rows[i][2].get("degree") not in (degree, "عام"):
            continue
        hits.append({"id": rows[i][0], "text": rows[i][1], "metadata": rows[i][2], "bm25": float(scores[i])})
    return hits


def retrieve(question: str, k: int = TOP_K, max_dist: float = MAX_DISTANCE,
             hybrid: bool = USE_HYBRID) -> list[dict]:
    """Return up to k chunks, or [] when nothing is relevant enough (-> refusal)."""
    if _collection().count() == 0:
        return []
    degree = detect_degree(question)
    question = expand_question(question)  # dialect -> formal words
    semantic = semantic_search(question, k * 2, degree)

    # Relevance gate: if even the closest chunk is too far, the answer is not in
    # our documents. This is what makes the system refuse instead of inventing.
    if not semantic or semantic[0]["distance"] > max_dist:
        return []

    if not hybrid:
        return [h for h in semantic if h["distance"] <= max_dist][:k]

    # Reciprocal Rank Fusion: score = sum of 1 / (60 + rank) over both lists.
    fused: dict[str, dict] = {}
    for results in (semantic, keyword_search(question, k * 2, degree)):
        for rank, hit in enumerate(results):
            entry = fused.setdefault(hit["id"], {**hit, "rrf": 0.0})
            entry["rrf"] += 1 / (60 + rank)
            entry.update({key: v for key, v in hit.items() if key not in entry})
    ranked = sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)
    # drop far semantic hits (keyword-only hits are kept: they match an exact term)
    ranked = [h for h in ranked if h.get("distance", 0) <= max_dist]
    return ranked[:k]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or "كم ساعة مادة تركيب البيانات؟"
    print(f"Q: {q}   (degree filter: {detect_degree(q)})\n")
    print("--- semantic (distance: smaller = closer) ---")
    for h in semantic_search(q, TOP_K, detect_degree(q)):
        print(f"{h['distance']:.3f}  {h['metadata']['source']}  {h['text'][:80].replace(chr(10), ' ')}")
    print(f"\n--- final retrieve() (hybrid={USE_HYBRID}, max_dist={MAX_DISTANCE}) ---")
    hits = retrieve(q)
    if not hits:
        print("(nothing passed the gate -> the system will refuse)")
    for h in hits:
        print(f"{h['metadata']['source']}  {h['text'][:100].replace(chr(10), ' ')}")
