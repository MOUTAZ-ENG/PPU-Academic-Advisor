"""Interface layer: Arabic (RTL) web app for the PPU Smart Academic Advisor.

    python app.py   ->  http://127.0.0.1:7860
Tabs: chat (with sources), add documents, usage dashboard, about the system.
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

EXAMPLES = [
    "هل بقدر أسحب مادة بعد الأسبوع الرابع؟",
    "كم ساعة معتمدة لمادة تركيب البيانات وشو متطلبها؟",
    "معدلي 60% وأنا سنة ثالثة، كم ساعة أقدر أسجل؟",
    "شو بصير إذا غبت أكثر من المسموح؟",
    "ما الفرق بين خطة علم الحاسوب 2020 و2022؟",
    "متى امتحان الخوارزميات؟",
]

HEADER = f"""
<div class="hero">
  <div class="hero-badge">جامعة بوليتكنك فلسطين · PPU</div>
  <h1>🎓 {TITLE}</h1>
  <p>اسأل بالفصحى أو باللهجة عن تعليمات الدراسة والخطط الدراسية والامتحانات،
     وستصلك الإجابة من مستندات الجامعة الرسمية مع ذكر المصدر.</p>
  <div class="chips">
    <span class="chip">📄 إجابات من مصادر رسمية</span>
    <span class="chip">🗣️ يفهم اللهجة المحكية</span>
    <span class="chip">🔢 يذكر رقم المادة والمساق</span>
    <span class="chip">🛑 يقول "لا أعرف" بدل التخمين</span>
  </div>
  <a class="tg-btn" href="{TELEGRAM_URL}" target="_blank">✈️ جرّبه على تيليجرام</a>
</div>
"""

ABOUT = """
### كيف يعمل؟
1. **الاستيعاب:** مستندات الجامعة تُنظَّف وتُقطَّع (كل مادة أو مساق في مقطع) وتُخزَّن في قاعدة متجهات **ChromaDB**.
2. **الاسترجاع:** نوسّع سؤالك من اللهجة إلى الكلمات الرسمية، ثم نبحث **بالمعنى وبالكلمات معًا** (بحث هجين) ونأخذ أقرب 6 مقاطع.
3. **التوليد:** موديل **Gemini** يكتب الإجابة **من هذه المقاطع فقط** ويذكر المصدر، أو يعتذر ويحيلك للجهة المختصة.

### المستندات المتوفرة
| المستند | الدرجة |
|---|---|
| تعليمات الدراسة لدرجة البكالوريوس | بكالوريوس |
| خطة علم الحاسوب 2022 (جدول كامل) و2020 (ملخص) | بكالوريوس |
| تعليمات الدراسة لدرجة الدبلوم | دبلوم |
| تعليمات الماجستير وخطة ماجستير المعلوماتية | ماجستير |
| تعليمات الامتحانات (إعلان كلية) | عام |

### حدود النظام
- يعرف فقط ما في المستندات أعلاه، ولا يحسب ساعاتك المتبقية دون سجلك الدراسي.
- الإجابات ليست استشارة رسمية: تحقّق دائمًا من المصدر، وراجع مشرفك الأكاديمي أو عمادة القبول والتسجيل.
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

:root { --ppu-1: #0b3d91; --ppu-2: #1565c0; --ppu-3: #00a19a; --soft: #f4f7fb; }

.gradio-container { direction: rtl; font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif !important;
                    max-width: 1100px !important; margin: auto; }
.gradio-container * { font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif; }

/* ---------- hero header ---------- */
.hero { position: relative; overflow: hidden; text-align: center; color: #fff; padding: 34px 24px 28px;
        border-radius: 22px; margin-bottom: 14px;
        background: linear-gradient(120deg, var(--ppu-1), var(--ppu-2), var(--ppu-3), var(--ppu-1));
        background-size: 300% 300%; animation: gradientMove 12s ease infinite, fadeDown .8s ease both;
        box-shadow: 0 12px 30px rgba(11, 61, 145, .25); }
.hero h1 { color: #fff !important; font-size: 2.2em; font-weight: 800; margin: 10px 0 6px; }
.hero p  { color: #eaf2ff !important; font-size: 1.08em; max-width: 720px; margin: 0 auto 14px; line-height: 1.8; }
.hero-badge { display: inline-block; background: rgba(255,255,255,.18); padding: 4px 14px; border-radius: 999px;
              font-size: .9em; letter-spacing: .5px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 16px; }
.chip { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35); padding: 6px 12px;
        border-radius: 999px; font-size: .92em; animation: fadeUp .6s ease both; }
.chip:nth-child(2) { animation-delay: .1s } .chip:nth-child(3) { animation-delay: .2s } .chip:nth-child(4) { animation-delay: .3s }
.tg-btn { display: inline-block; background: #fff; color: var(--ppu-1) !important; font-weight: 700;
          padding: 10px 22px; border-radius: 12px; text-decoration: none; transition: transform .2s, box-shadow .2s; }
.tg-btn:hover { transform: translateY(-2px) scale(1.03); box-shadow: 0 8px 18px rgba(0,0,0,.2); }

/* ---------- chat ---------- */
#chat { border-radius: 18px !important; box-shadow: 0 6px 22px rgba(0,0,0,.06); }
#chat .message, .message { direction: rtl; text-align: right; line-height: 1.9; font-size: 1.02em;
                           animation: fadeUp .35s ease both; }
#ask-box textarea { direction: rtl; text-align: right; font-size: 1.05em; border-radius: 14px !important; }
#send-btn { background: linear-gradient(90deg, var(--ppu-2), var(--ppu-3)) !important; color: #fff !important;
            border: none !important; font-weight: 700; border-radius: 14px !important; transition: transform .15s; }
#send-btn:hover { transform: translateY(-1px); filter: brightness(1.08); }
#clear-btn { border-radius: 14px !important; }

.disclaimer { text-align: center; color: #6b7280; font-size: .9em; margin-top: 10px; }
.section-note { background: var(--soft); border-right: 4px solid var(--ppu-3); padding: 10px 14px;
                border-radius: 10px; margin-bottom: 10px; }

@keyframes gradientMove { 0% {background-position: 0% 50%} 50% {background-position: 100% 50%} 100% {background-position: 0% 50%} }
@keyframes fadeUp   { from { opacity: 0; transform: translateY(10px) } to { opacity: 1; transform: none } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-14px) } to { opacity: 1; transform: none } }
"""


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
        chatbot = gr.Chatbot(elem_id="chat", height=500, rtl=True, label="المحادثة")
        with gr.Row():
            box = gr.Textbox(elem_id="ask-box", placeholder="اكتب سؤالك هنا... مثال: كم ساعة بقدر أسجل إذا كنت منذر؟",
                             scale=6, show_label=False, rtl=True, autofocus=True)
            send = gr.Button("إرسال ✨", elem_id="send-btn", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=box, label="💡 جرّب أحد هذه الأسئلة")
        clear = gr.Button("🗑️ محادثة جديدة", elem_id="clear-btn")
        send.click(respond, [box, chatbot], [box, chatbot])
        box.submit(respond, [box, chatbot], [box, chatbot])
        clear.click(lambda: [], None, chatbot)

    with gr.Tab("📊 لوحة الاستخدام"):
        gr.HTML('<div class="section-note">إحصائيات من الموقع وبوت تيليجرام معًا: كم سؤالًا وصل، '
                'وما الأسئلة الأكثر تكرارًا، وكم مرة رفض النظام الإجابة.</div>')
        stats = gr.Markdown()
        refresh_btn = gr.Button("🔄 تحديث")
        refresh_btn.click(summary_markdown, None, stats)
        demo.load(summary_markdown, None, stats)

    with gr.Tab("📂 إضافة مستندات"):
        gr.HTML('<div class="section-note">ارفع ملف PDF أو TXT جديدًا من مستندات الجامعة، '
                'فيصبح جزءًا من معرفة المرشد فورًا.</div>')
        files = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], label="ارفع ملفات PDF أو TXT")
        status = gr.Markdown()
        files.upload(upload, files, status)

    with gr.Tab("ℹ️ عن النظام"):
        gr.Markdown(ABOUT, rtl=True)

    gr.HTML('<div class="disclaimer">⚠️ الإجابات مبنية على مستندات الجامعة المتاحة فقط وليست استشارة رسمية. '
            'تحقّق دائمًا من المصدر وراجع مشرفك الأكاديمي.</div>')


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"), inbrowser=True)
