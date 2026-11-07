"""Evaluation: measure the system, change ONE thing, measure again.

    python evaluate.py --retrieval-only   # search only: free, no API calls
    python evaluate.py                    # search + answers + refusals (15 model calls)
    python evaluate.py --judge            # + Faithfulness scored by the model (costs more)

Every run is appended to eval_log.csv with the settings used -> before/after table.

tests.json: {"q": question, "has": [words a correct answer contains], "type": ...}
types: direct, dialect, table, multi, conflict -> must answer;
       not_in_docs, out_of_scope -> must refuse ("has" stays empty).
"""
import argparse
import csv
import json
import sys
from datetime import datetime

import ingest
import retriever
from generator import NOT_FOUND, OUT_OF_SCOPE, ask, build_context, llm
from ingest import BASE_DIR, normalize_ar
from retriever import retrieve

TESTS = json.loads((BASE_DIR / "tests.json").read_text(encoding="utf-8"))
LOG = BASE_DIR / "eval_log.csv"
REFUSE_TYPES = ("not_in_docs", "out_of_scope")


def contains(text: str, keywords: list[str]) -> bool:
    t = normalize_ar(text)
    return any(normalize_ar(k) in t for k in keywords)


def is_refusal(answer: str) -> bool:
    a = normalize_ar(answer)
    return a.startswith(normalize_ar(NOT_FOUND[:15])) or a.startswith(normalize_ar(OUT_OF_SCOPE[:15]))


def judge_faithfulness(question: str, answer: str, hits: list[dict]) -> bool:
    """Model-as-judge: is every fact in the answer supported by the retrieved chunks?"""
    verdict = llm("You are a strict evaluator. Reply with exactly one word: YES or NO.",
                  [{"role": "user", "content": f"{build_context(hits)}\n\nQuestion: {question}\n"
                    f"Answer: {answer}\n\nIs EVERY fact in the answer directly supported by the context?"}])
    return "YES" in verdict.upper()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--note", default="", help="what you changed, e.g. 'k 4->6'")
    args = ap.parse_args()

    answerable = [t for t in TESTS if t["type"] not in REFUSE_TYPES]
    refusals = [t for t in TESTS if t["type"] in REFUSE_TYPES]
    k = retriever.TOP_K

    # 1) Retrieval: Hit@k (right chunk found) and Precision@k (share of useful chunks)
    print(f"\n=== Retrieval (k={k}, hybrid={retriever.USE_HYBRID}) ===")
    hit_count, prec_sum = 0, 0.0
    for t in answerable:
        hits = retrieve(t["q"])
        useful = [h for h in hits if contains(h["text"], t["has"])]
        hit_count += bool(useful)
        prec_sum += len(useful) / k
        print(f"{'HIT ' if useful else 'MISS'}  [{t['type']}] {t['q']}")
    hit_rate, precision = hit_count / len(answerable), prec_sum / len(answerable)
    print(f"\nHit@{k} = {hit_rate:.0%}   Precision@{k} = {precision:.0%}")

    answer_acc = refusal_acc = faith = None
    passed = None
    if not args.retrieval_only:
        # 2) Answers + 3) refusals (+ optional faithfulness)
        print("\n=== Answers ===")
        correct = refused_ok = faithful = judged = 0
        for t in TESTS:
            r = ask(t["q"])
            refused = is_refusal(r["answer"])
            if t["type"] in REFUSE_TYPES:
                ok = refused
                refused_ok += ok
            else:
                ok = (not refused) and contains(r["answer"], t["has"])
                correct += ok
                if args.judge and not refused:
                    judged += 1
                    faithful += judge_faithfulness(t["q"], r["answer"], r["hits"])
            print(f"{'PASS' if ok else 'FAIL'}  [{t['type']}] {t['q']}\n      -> {r['answer'][:110]!r}")
        answer_acc = correct / len(answerable)
        refusal_acc = refused_ok / len(refusals)
        faith = faithful / judged if judged else None
        passed = correct + refused_ok
        print(f"\nPASSED {passed} of {len(TESTS)}   answers {correct}/{len(answerable)}   "
              f"refusals {refused_ok}/{len(refusals)}" + (f"   faithfulness {faith:.0%}" if faith is not None else ""))

    # log the run (before/after table)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "note", "chunk_size", "top_k", "max_dist", "hybrid", "model",
                        "hit@k", "precision@k", "answer_acc", "refusal_acc", "passed", "faithfulness"])
        fmt = lambda x: "" if x is None else f"{x:.2f}"
        import generator
        w.writerow([datetime.now().strftime("%H:%M"), args.note, ingest.CHUNK_SIZE, k, retriever.MAX_DISTANCE,
                    retriever.USE_HYBRID, generator.GEMINI_MODEL, fmt(hit_rate), fmt(precision),
                    fmt(answer_acc), fmt(refusal_acc), "" if passed is None else f"{passed}/{len(TESTS)}", fmt(faith)])
    print(f"Saved to {LOG.name}")


if __name__ == "__main__":
    main()
