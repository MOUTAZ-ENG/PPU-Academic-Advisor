"""Interface layer: Arabic (RTL) web app for the PPU Smart Academic Advisor.

    python app.py   ->  http://127.0.0.1:7860
Tabs: chat (with sources), usage dashboard, add documents, about the system.
Dark theme with animated header, glass panels and message animations.
"""
import shutil
from pathlib import Path

import gradio as gr

import retriever
from analytics import log_interaction, summary_markdown
from generator import ask
from ingest import DOCS_DIR, ingest_file

TITLE = "المرشد الأكاديمي الذكي"
TELEGRAM_URL = "https://t.me/Guide_Assistant12_bot"

SUGGESTED = [
    "هل بقدر أسحب مادة بعد الأسبوع الرابع؟",
    "كم ساعة معتمدة لمادة تركيب البيانات وشو متطلبها؟",
    "معدلي 60% وأنا سنة ثالثة، كم ساعة أقدر أسجل؟",
    "اذا غبت عن امتحان لظرف ما شو ممكن اعمل؟",
    "ما الفرق بين خطة علم الحاسوب 2020 و2022؟",
    "متى امتحان الخوارزميات؟",
]

HEADER = f"""
<div class="hero">
  <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
  <div class="hero-inner">
    <div class="hero-badge"><span class="live-dot"></span> جامعة بوليتكنك فلسطين · PPU</div>
    <h1 class="glow">🎓 {TITLE}</h1>
    <p class="typing">اسأل بالفصحى أو باللهجة… والإجابة من مستندات الجامعة الرسمية مع المصدر</p>
    <div class="chips">
      <span class="chip">📄 مصادر رسمية</span>
      <span class="chip">🗣️ يفهم اللهجة</span>
      <span class="chip">🔢 رقم المادة والمساق</span>
      <span class="chip">🛑 "لا أعرف" بدل التخمين</span>
    </div>
    <div class="stats">
      <div class="stat"><b>8</b><span>مستندات رسمية</span></div>
      <div class="stat"><b>100+</b><span>مساق في الخطة</span></div>
      <div class="stat"><b>3</b><span>درجات علمية</span></div>
      <div class="stat"><b>24/7</b><span>متاح دائمًا</span></div>
    </div>
    <a class="tg-btn" href="{TELEGRAM_URL}" target="_blank">✈️ جرّبه على تيليجرام</a>
  </div>
</div>
"""

ABOUT = """
### ⚙️ كيف يعمل؟
1. **الاستيعاب:** مستندات الجامعة تُنظَّف وتُقطَّع (كل مادة أو مساق في مقطع) وتُخزَّن في قاعدة متجهات **ChromaDB**.
2. **الاسترجاع:** نوسّع سؤالك من اللهجة إلى الكلمات الرسمية، ثم نبحث **بالمعنى وبالكلمات معًا** (بحث هجين) ونأخذ أقرب 6 مقاطع.
3. **التوليد:** موديل **Gemini** يكتب الإجابة **من هذه المقاطع فقط** ويذكر المصدر، أو يعتذر ويحيلك للجهة المختصة.

### 📚 المستندات المتوفرة
| المستند | الدرجة |
|---|---|
| تعليمات الدراسة لدرجة البكالوريوس | بكالوريوس |
| خطة علم الحاسوب 2022 (جدول كامل) و2020 (ملخص) | بكالوريوس |
| تعليمات الدراسة لدرجة الدبلوم | دبلوم |
| تعليمات الماجستير وخطة ماجستير المعلوماتية | ماجستير |
| تعليمات الامتحانات (إعلان كلية) | عام |

### ⚠️ حدود النظام
- يعرف فقط ما في المستندات أعلاه، ولا يحسب ساعاتك المتبقية دون سجلك الدراسي.
- الإجابات ليست استشارة رسمية: تحقّق دائمًا من المصدر، وراجع مشرفك الأكاديمي أو عمادة القبول والتسجيل.
"""

# Force dark mode for everyone (Gradio reads ?__theme=dark).
FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap');

:root { --c1: #22d3ee; --c2: #6366f1; --c3: #a855f7; --bg: #060912; --panel: rgba(17, 24, 39, .72);
        --line: rgba(148, 163, 184, .18); --text: #eef2ff; --muted: #a5b4cc; }

body, .gradio-container { background: radial-gradient(1200px 600px at 85% -10%, rgba(99,102,241,.18), transparent 60%),
                                     radial-gradient(900px 500px at 0% 110%, rgba(34,211,238,.12), transparent 60%),
                                     var(--bg) !important; }
.gradio-container { direction: rtl; max-width: 1150px !important; margin: auto; color: var(--text); }
.gradio-container, .gradio-container * { font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif !important; }

/* ---------------- hero ---------------- */
.hero { position: relative; overflow: hidden; border-radius: 26px; margin: 6px 0 16px; padding: 38px 24px 30px;
        text-align: center; background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(99,102,241,.14), rgba(168,85,247,.12));
        border: 1px solid var(--line); box-shadow: 0 20px 60px rgba(0,0,0,.45), inset 0 0 60px rgba(99,102,241,.08);
        animation: fadeDown .9s ease both; }
.hero::before { content: ""; position: absolute; inset: -2px; border-radius: 26px; padding: 2px;
                background: linear-gradient(90deg, var(--c1), var(--c2), var(--c3), var(--c1)); background-size: 300% 100%;
                -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
                -webkit-mask-composite: xor; mask-composite: exclude; animation: borderFlow 6s linear infinite; }
.hero-inner { position: relative; z-index: 2; }
.orb { position: absolute; border-radius: 50%; filter: blur(40px); opacity: .55; z-index: 1; }
.orb1 { width: 220px; height: 220px; background: var(--c2); top: -70px; right: -40px; animation: float 9s ease-in-out infinite; }
.orb2 { width: 180px; height: 180px; background: var(--c1); bottom: -70px; left: 8%; animation: float 11s ease-in-out infinite reverse; }
.orb3 { width: 140px; height: 140px; background: var(--c3); top: 30%; left: -50px; animation: float 13s ease-in-out infinite; }

.hero-badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 999px;
              background: rgba(255,255,255,.06); border: 1px solid var(--line); color: var(--muted); font-size: .95em; }
.live-dot { width: 9px; height: 9px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 0 0 rgba(34,197,94,.7);
            animation: pulse 1.8s infinite; }
.hero h1 { font-size: 2.6em !important; font-weight: 900 !important; margin: 14px 0 8px !important;
           background: linear-gradient(90deg, #fff, var(--c1), #c7d2fe, #fff); background-size: 300% 100%;
           -webkit-background-clip: text; background-clip: text; color: transparent !important;
           animation: shine 5s linear infinite; }
.typing { display: inline-block; color: #dbe4ff !important; font-size: 1.15em; line-height: 1.9; margin: 0 auto 16px;
          overflow: hidden; white-space: nowrap; border-left: 3px solid var(--c1);
          animation: typing 3.2s steps(60, end) .4s both, caret .8s step-end infinite; max-width: 100%; }
@media (max-width: 760px) { .typing { white-space: normal; border: none; animation: fadeUp .8s ease both; } }

.chips { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 4px 0 18px; }
.chip { padding: 8px 14px; border-radius: 999px; background: rgba(255,255,255,.05); border: 1px solid var(--line);
        color: var(--text); font-size: .98em; transition: transform .25s, border-color .25s, box-shadow .25s;
        animation: fadeUp .6s ease both; cursor: default; }
.chip:nth-child(2){animation-delay:.12s} .chip:nth-child(3){animation-delay:.24s} .chip:nth-child(4){animation-delay:.36s}
.chip:hover { transform: translateY(-3px); border-color: var(--c1); box-shadow: 0 8px 22px rgba(34,211,238,.18); }

.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; max-width: 760px; margin: 0 auto 20px; }
@media (max-width: 760px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.stat { padding: 12px 8px; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid var(--line);
        transition: transform .25s, background .25s; animation: fadeUp .7s ease both; }
.stat:hover { transform: translateY(-4px); background: rgba(99,102,241,.12); }
.stat b { display: block; font-size: 1.6em; font-weight: 900; color: var(--c1); }
.stat span { color: var(--muted); font-size: .9em; }

.tg-btn { display: inline-block; padding: 12px 26px; border-radius: 14px; font-weight: 800; text-decoration: none;
          color: #04121a !important; background: linear-gradient(90deg, var(--c1), #7dd3fc);
          box-shadow: 0 10px 26px rgba(34,211,238,.35); transition: transform .2s, box-shadow .2s; }
.tg-btn:hover { transform: translateY(-3px) scale(1.04); box-shadow: 0 14px 34px rgba(34,211,238,.55); }

/* ---------------- panels, tabs, chat ---------------- */
.block, .form, .tabitem { background: var(--panel) !important; border-color: var(--line) !important;
                          backdrop-filter: blur(10px); border-radius: 18px !important; }
button[role="tab"] { font-size: 1.05em !important; font-weight: 700 !important; color: var(--muted) !important;
                     transition: color .2s; }
button[role="tab"][aria-selected="true"] { color: var(--c1) !important; }

#chat { border: 1px solid var(--line) !important; box-shadow: 0 16px 40px rgba(0,0,0,.35); }
#chat .message, .message { direction: rtl; text-align: right; font-size: 1.07em !important; line-height: 2 !important;
                           color: var(--text) !important; animation: msgIn .45s cubic-bezier(.2,.8,.2,1) both; }
#chat .user, .message.user { background: linear-gradient(135deg, rgba(99,102,241,.35), rgba(34,211,238,.22)) !important;
                              border: 1px solid rgba(99,102,241,.45) !important; }
#chat .bot, .message.bot { background: rgba(255,255,255,.04) !important; border: 1px solid var(--line) !important; }
#chat a { color: var(--c1) !important; }
#chat strong { color: #fff; }

#ask-box textarea { direction: rtl; text-align: right; font-size: 1.1em !important; color: var(--text) !important;
                    background: rgba(255,255,255,.04) !important; border: 1px solid var(--line) !important;
                    border-radius: 16px !important; transition: border-color .2s, box-shadow .2s; }
#ask-box textarea:focus { border-color: var(--c1) !important; box-shadow: 0 0 0 4px rgba(34,211,238,.15) !important; }
#send-btn { font-weight: 800 !important; font-size: 1.05em !important; color: #fff !important; border: none !important;
            border-radius: 16px !important; background: linear-gradient(90deg, var(--c2), var(--c1)) !important;
            box-shadow: 0 10px 24px rgba(99,102,241,.35); transition: transform .15s, filter .2s; animation: glowPulse 2.6s infinite; }
#send-btn:hover { transform: translateY(-2px); filter: brightness(1.15); }
#clear-btn { border-radius: 14px !important; transition: transform .15s; }
#clear-btn:hover { transform: translateY(-2px); }

/* suggested questions */
#suggested .label-wrap span, #suggested > div > span { font-weight: 800 !important; color: var(--c1) !important; font-size: 1.05em; }
#suggested button { border-radius: 999px !important; border: 1px solid var(--line) !important; color: var(--text) !important;
                    background: rgba(255,255,255,.04) !important; transition: transform .2s, border-color .2s, background .2s; }
#suggested button:hover { transform: translateY(-3px); border-color: var(--c1) !important; background: rgba(34,211,238,.10) !important; }

.section-note { padding: 12px 16px; border-radius: 14px; margin-bottom: 10px; color: var(--text);
                background: linear-gradient(90deg, rgba(34,211,238,.10), transparent); border-right: 4px solid var(--c1); }
.prose, .prose * { color: var(--text) !important; }
.prose table { border-color: var(--line) !important; }
.disclaimer { text-align: center; color: var(--muted); font-size: .95em; margin: 14px 0 6px; }

::-webkit-scrollbar { width: 10px; } ::-webkit-scrollbar-thumb { background: rgba(99,102,241,.45); border-radius: 10px; }

@keyframes borderFlow { to { background-position: 300% 0; } }
@keyframes shine { to { background-position: -300% 0; } }
@keyframes float { 0%,100% { transform: translate(0,0) } 50% { transform: translate(20px,-18px) } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.7) } 70% { box-shadow: 0 0 0 10px rgba(34,197,94,0) } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0) } }
@keyframes typing { from { max-width: 0 } to { max-width: 100% } }
@keyframes caret { 50% { border-color: transparent } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px) } to { opacity: 1; transform: none } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-16px) } to { opacity: 1; transform: none } }
@keyframes msgIn { from { opacity: 0; transform: translateY(14px) scale(.98) } to { opacity: 1; transform: none } }
@keyframes glowPulse { 0%,100% { box-shadow: 0 10px 24px rgba(99,102,241,.30) } 50% { box-shadow: 0 10px 32px rgba(34,211,238,.50) } }
"""

THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="indigo", neutral_hue="slate").set(
    body_background_fill_dark="#060912",
    block_background_fill_dark="rgba(17, 24, 39, .72)",
    body_text_color_dark="#eef2ff",
    block_label_text_color_dark="#a5b4cc",
)


def _text(content) -> str:
    """Gradio 6 may store message content as a list of parts ([{"type": "text", "text": ...}])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def to_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """Gradio keeps messages as [{role, content}, ...]; ask() wants (question, answer) pairs."""
    pairs, q = [], None
    for m in messages:
        if m["role"] == "user":
            q = _text(m["content"])
        elif m["role"] == "assistant" and q is not None:
            pairs.append((q, _text(m["content"]).split("\n\n---\n📄")[0]))
            q = None
    return pairs


def respond(question: str, messages: list[dict]):
    question = (question or "").strip()
    if not question:
        return "", messages
    try:
        result = ask(question, to_pairs(messages))
    except Exception as e:  # e.g. no internet or no model quota
        print("Error:", e)
        result = {"answer": "⚠️ تعذّر الاتصال بالموديل حاليًا. حاول مرة أخرى بعد قليل.", "sources": [], "hits": []}
    answer = result["answer"]
    if result["sources"]:
        answer += "\n\n---\n📄 **المستندات المستخدمة:**\n" + "\n".join(f"- {s}" for s in result["sources"])
    log_interaction("web", question, answered=bool(result["sources"]), n_sources=len(result["sources"]))
    messages = messages + [{"role": "user", "content": question},
                           {"role": "assistant", "content": answer}]
    return "", messages


def upload(files) -> str:
    if not files:
        return ""
    lines = []
    for f in files:
        src = Path(f if isinstance(f, str) else f.name)
        dest = DOCS_DIR / src.name
        shutil.copy(src, dest)
        n = ingest_file(dest)
        lines.append(f"✅ **{src.name}**: أُضيف {n} مقطع إلى قاعدة المعرفة")
    retriever.refresh()  # rebuild the keyword index with the new chunks
    return "\n\n".join(lines)


with gr.Blocks(title=TITLE) as demo:
    gr.HTML(HEADER)
    with gr.Tab("💬 المحادثة"):
        chatbot = gr.Chatbot(elem_id="chat", height=520, rtl=True, label="المحادثة")
        with gr.Row():
            box = gr.Textbox(elem_id="ask-box", placeholder="✍️ اكتب سؤالك هنا… مثال: كم ساعة بقدر أسجل إذا كنت منذر؟",
                             scale=6, show_label=False, rtl=True, autofocus=True)
            send = gr.Button("إرسال ✨", elem_id="send-btn", variant="primary", scale=1)
        with gr.Column(elem_id="suggested"):
            gr.Examples(SUGGESTED, inputs=box, label="💡 أسئلة مقترحة (اضغط على أي سؤال لتجربته)")
        clear = gr.Button("🗑️ محادثة جديدة", elem_id="clear-btn")
        send.click(respond, [box, chatbot], [box, chatbot])
        box.submit(respond, [box, chatbot], [box, chatbot])
        clear.click(lambda: [], None, chatbot)

    with gr.Tab("📊 لوحة الاستخدام"):
        gr.HTML('<div class="section-note">📈 إحصائيات من الموقع وبوت تيليجرام معًا: كم سؤالًا وصل، '
                'وما الأسئلة الأكثر تكرارًا، وكم مرة رفض النظام الإجابة.</div>')
        stats = gr.Markdown(rtl=True)
        refresh_btn = gr.Button("🔄 تحديث")
        refresh_btn.click(summary_markdown, None, stats)
        demo.load(summary_markdown, None, stats)

    with gr.Tab("📂 إضافة مستندات"):
        gr.HTML('<div class="section-note">📤 ارفع ملف PDF أو TXT جديدًا من مستندات الجامعة، '
                'فيصبح جزءًا من معرفة المرشد فورًا.</div>')
        files = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], label="ارفع ملفات PDF أو TXT")
        status = gr.Markdown(rtl=True)
        files.upload(upload, files, status)

    with gr.Tab("ℹ️ عن النظام"):
        gr.Markdown(ABOUT, rtl=True)

    gr.HTML('<div class="disclaimer">⚠️ الإجابات مبنية على مستندات الجامعة المتاحة فقط وليست استشارة رسمية. '
            'تحقّق دائمًا من المصدر وراجع مشرفك الأكاديمي.</div>')


if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, js=FORCE_DARK_JS, inbrowser=True)
