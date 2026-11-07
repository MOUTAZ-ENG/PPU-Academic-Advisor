"""Generation layer: build a grounded prompt, call the LLM, return answer + sources.

`ask()` is the only function the interfaces (web app, Telegram bot) need.
    python generator.py      -> chat in the terminal
"""
import os
import sys
import time

from dotenv import load_dotenv

from retriever import retrieve

load_dotenv()  # reads the API keys from .env (.env is never committed)

# ================================================================ SETTINGS ===
LLM_PROVIDER = "gemini"          # "gemini" (free tier) or "claude"
GEMINI_MODEL = "gemini-3.6-flash"                # best answers (smaller free quota)
GEMINI_FALLBACK = "gemini-3.5-flash-lite"        # used automatically if the main model hits its limit
CLAUDE_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4000                # 3.6-flash "thinks" first: 900 cut answers in half
HISTORY_TURNS = 3                # previous Q/A pairs kept for follow-up questions

# Answer when the question is unrelated to the university (no model call is made).
OUT_OF_SCOPE = ("عذرًا، أنا المرشد الأكاديمي لجامعة بوليتكنك فلسطين، وأجيب فقط عن الأسئلة "
                "الأكاديمية من مستندات الجامعة: تعليمات الدراسة، والخطط الدراسية، وتعليمات الامتحانات.")
# Start of the answer when the topic is academic but the documents do not contain it.
NOT_FOUND = "لا أجد هذه المعلومة في المستندات المتاحة لي."
# English versions (the assistant answers in the language of the question).
OUT_OF_SCOPE_EN = ("Sorry, I am the Smart Academic Advisor of Palestine Polytechnic University. I only answer "
                   "academic questions from the university documents: study regulations, study plans and exam instructions.")
NOT_FOUND_EN = "I can't find this information in the documents available to me."
REFUSAL_PREFIXES = (NOT_FOUND, OUT_OF_SCOPE, NOT_FOUND_EN, OUT_OF_SCOPE_EN)


def is_english(text: str) -> bool:
    """True when most letters in the question are Latin (A-Z)."""
    letters = [c for c in text if c.isalpha()]
    latin = sum(c.isascii() for c in letters)
    return bool(letters) and latin / len(letters) > 0.5


def is_refusal(answer: str) -> bool:
    return any(answer.strip().startswith(p[:15]) for p in REFUSAL_PREFIXES)

SYSTEM_PROMPT = f"""أنت "المرشد الأكاديمي الذكي" لطلبة جامعة بوليتكنك فلسطين. تجيب فقط اعتمادًا على المقاطع داخل <context>.

القواعد:
1. أجب فقط من المقاطع داخل <context>. لا تستخدم معرفتك العامة ولا تخمّن أي رقم أو قاعدة.
2. إذا لم تكفِ المقاطع للإجابة، ابدأ ردك حرفيًا بـ: "{NOT_FOUND}" (أو بالإنجليزية: "{NOT_FOUND_EN}") ثم أحِل الطالب إلى الجهة المختصة: عمادة القبول والتسجيل (للتعليمات والتسجيل والتقويم)، أو الدائرة والمشرف الأكاديمي (للخطط والمساقات)، أو جدول الامتحانات الرسمي (لمواعيد الامتحانات).
3. إذا كان السؤال لا علاقة له بالدراسة في الجامعة، أجب بهذا النص فقط: "{OUT_OF_SCOPE}" (أو بالإنجليزية: "{OUT_OF_SCOPE_EN}")
4. اللغة: أجب بلغة السؤال. إذا كان السؤال بالإنجليزية فأجب بالإنجليزية بالكامل، مع ترجمة المعلومة من المقاطع العربية بدقة والحفاظ على الأرقام ورموز المساقات وأرقام المواد. وإذا كان بالعربية أو باللهجة المحكية فافهم القصد وأجب بعربية فصحى مبسّطة وواضحة.
5. في نهاية كل إجابة اكتب سطرًا يبدأ بـ "المصدر:" (أو "Source:" للإجابة الإنجليزية) يذكر اسم المستند كما في title، ورقم المادة أو رقم المساق أو الصفحة إن وُجدت في المقطع. مثال: "المصدر: تعليمات الدراسة لدرجة البكالوريوس، المادة (7)".
6. القواعد تختلف بين الدبلوم والبكالوريوس والماجستير: لا تخلط بينها. إذا لم يذكر الطالب درجته واختلفت القاعدة بين الدرجات، اذكر الحكم لكل درجة باختصار، أو افترض البكالوريوس وصرّح بذلك.
7. الخطط الدراسية تختلف حسب سنة الخطة (مثل 2020 و2022). لا تطبّق خطة على طالب من خطة أخرى، وإذا تعارضت نسختان اذكر الاثنتين وأحِل إلى الدائرة.
8. إذا احتاج السؤال الجمع بين أكثر من مقطع (مثل المعدل + العبء الدراسي)، اجمع بينها واذكر كل المصادر.
9. استخدم خطوات مرقمة أو جدولًا قصيرًا عندما يسهّل ذلك الفهم.
10. إذا اعتمدت على "تعليمات هامة للامتحانات" فنبّه أنها إعلان كلية وليست تعليمات رسمية موثّقة لكل الجامعة.
11. لا تحسب للطالب ساعاته المتبقية للتخرج دون سجله الدراسي؛ اشرح القاعدة وأحِله إلى مشرفه الأكاديمي."""

_client = None
_gemini_clients: dict[str, object] = {}
_exhausted: dict[tuple[str, str], float] = {}  # (key, model) -> time it hit its quota
RETRY_AFTER = 600  # seconds before trying an exhausted key/model again


def _gemini_keys() -> list[str]:
    """GEMINI_API_KEY plus optional backup keys GEMINI_API_KEY_2, _3 (other Google accounts)."""
    names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
    return [k for k in (os.getenv(n, "").strip() for n in names) if k]


def llm(system: str, messages: list[dict]) -> str:
    """The ONLY function that talks to the model. Switch provider with LLM_PROVIDER."""
    global _client
    if LLM_PROVIDER == "gemini":
        # Gemini accepts the OpenAI request format, so we use the openai library.
        from openai import APIStatusError, OpenAI
        # Free tier: every key and every model has its own quota. Try the best model on
        # each key first, then the fallback model. A key/model that hit its limit is
        # skipped for 10 minutes, so later questions do not waste time on it.
        for model in (GEMINI_MODEL, GEMINI_FALLBACK):
            for key in _gemini_keys():
                if time.time() - _exhausted.get((key, model), 0) < RETRY_AFTER:
                    continue
                client = _gemini_clients.setdefault(key, OpenAI(
                    api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"))
                try:
                    reply = client.chat.completions.create(
                        model=model,
                        max_tokens=MAX_TOKENS,
                        messages=[{"role": "system", "content": system}] + messages,
                    )
                    return reply.choices[0].message.content or ""
                except APIStatusError as e:
                    if e.status_code != 429 and e.status_code < 500:
                        raise
                    print(f"  ! {model} on key ...{key[-4:]} unavailable ({e.status_code}), trying next")
                    _exhausted[(key, model)] = time.time()
        raise RuntimeError("All Gemini keys/models are out of quota right now")

    import anthropic
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env
    reply = _client.messages.create(model=CLAUDE_MODEL, max_tokens=MAX_TOKENS,
                                    system=system, messages=messages)
    return reply.content[0].text


def build_context(hits: list[dict]) -> str:
    """Number the chunks and attach each chunk's document title and degree."""
    blocks = []
    for n, h in enumerate(hits, start=1):
        m = h["metadata"]
        page = f' page="{m["page"]}"' if m.get("page") else ""
        blocks.append(f'<chunk id="{n}" title="{m["title"]}" degree="{m["degree"]}"{page}>\n{h["text"]}\n</chunk>')
    return "<context>\n" + "\n\n".join(blocks) + "\n</context>"


def format_sources(hits: list[dict]) -> list[str]:
    """One line per document used (no duplicates): title + official link."""
    seen, out = set(), []
    for h in hits:
        m = h["metadata"]
        if m["source"] in seen:
            continue
        seen.add(m["source"])
        link = f" - {m['url']}" if m.get("url", "").startswith("http") else ""
        out.append(f"{m['title']}{link}")
    return out


def ask(question: str, history: list[tuple[str, str]] | None = None) -> dict:
    """Answer a question. `history` = previous (question, answer) pairs for follow-ups.

    Returns {"answer": str, "sources": [str], "hits": [chunks]}.
    """
    history = (history or [])[-HISTORY_TURNS:]

    # Follow-up questions ("وإذا كان معدلي أعلى؟") have no topic of their own, so we
    # search with the previous question added to give the retriever the context.
    search_query = f"{history[-1][0]} {question}" if history else question
    hits = retrieve(search_query)

    if not hits:  # relevance gate: nothing close enough -> polite out-of-scope answer, no model call
        return {"answer": OUT_OF_SCOPE_EN if is_english(question) else OUT_OF_SCOPE, "sources": [], "hits": []}

    messages = []
    for q, a in history:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": f"{build_context(hits)}\n\nسؤال الطالب: {question}"})

    answer = llm(SYSTEM_PROMPT, messages).strip()
    return {"answer": answer, "sources": [] if is_refusal(answer) else format_sources(hits), "hits": hits}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    hist = []
    while True:
        q = input("\nسؤالك (أو quit): ").strip()
        if q == "quit":
            break
        result = ask(q, hist)
        print("\n" + result["answer"])
        for s in result["sources"]:
            print("  📄 " + s)
        hist.append((q, result["answer"]))
