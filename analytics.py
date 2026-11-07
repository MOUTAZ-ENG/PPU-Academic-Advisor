"""Usage analytics: save every question (web + Telegram) and summarize the most asked ones.

No personal data is stored: only the time, channel, question text and whether it was refused.
The log file logs/interactions.jsonl is excluded from GitHub (.gitignore).
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "logs" / "interactions.jsonl"


def log_interaction(channel: str, question: str, answered: bool, n_sources: int) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), "channel": channel,
              "question": question.strip(), "answered": answered, "sources": n_sources}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    return [json.loads(line) for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_markdown(top: int = 10) -> str:
    rows = load()
    if not rows:
        return "لا توجد أسئلة مسجلة بعد. جرّب المحادثة ثم اضغط تحديث."
    total = len(rows)
    answered = sum(r["answered"] for r in rows)
    channels = Counter(r["channel"] for r in rows)
    common = Counter(r["question"] for r in rows).most_common(top)
    lines = [
        f"| إجمالي الأسئلة | تمت الإجابة | رفض أو إحالة | الموقع | تيليجرام |",
        f"|---|---|---|---|---|",
        f"| **{total}** | {answered} ({answered / total:.0%}) | {total - answered} | "
        f"{channels.get('web', 0)} | {channels.get('telegram', 0)} |",
        "", "### الأسئلة الأكثر تكرارًا", "", "| # | السؤال | عدد المرات |", "|---|---|---|",
    ]
    lines += [f"| {i} | {q} | {n} |" for i, (q, n) in enumerate(common, 1)]
    lines += ["", "### آخر الأسئلة", ""]
    lines += [f"- `{r['time'][11:16]}` {'✅' if r['answered'] else '⛔'} {r['question']}" for r in rows[-5:][::-1]]
    return "\n".join(lines)
