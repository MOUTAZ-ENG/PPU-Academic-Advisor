"""Interface layer: Arabic (RTL) web chat with sources and document upload.

    python app.py   ->  http://127.0.0.1:7860
"""
import shutil
from pathlib import Path

import gradio as gr

import retriever
from generator import ask
from ingest import DOCS_DIR, ingest_file

TITLE = "🎓 المرشد الأكاديمي الذكي - جامعة بوليتكنك فلسطين"
SUBTITLE = ("اسأل بالفصحى أو باللهجة عن تعليمات الدراسة والخطط الدراسية والامتحانات، "
            "وسأجيبك من مستندات الجامعة الرسمية مع ذكر المصدر.")
EXAMPLES = [
    "هل أقدر أسحب مادة بعد الأسبوع الرابع؟",
    "كم ساعة معتمدة لمادة تركيب البيانات وشو متطلبها؟",
    "معدلي 60% وأنا سنة ثالثة، كم ساعة أقدر أسجل؟",
    "متى امتحان الخوارزميات؟",
]

CSS = """
.gradio-container { direction: rtl; font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif; }
.message, .prose, textarea, input { direction: rtl; text-align: right; }
#sources { font-size: 0.9em; }
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
            pairs.append((q, _text(m["content"]).split("\n\n**المصادر:**")[0]))
            q = None
    return pairs


def respond(question: str, messages: list[dict]):
    question = (question or "").strip()
    if not question:
        return "", messages
    result = ask(question, to_pairs(messages))
    answer = result["answer"]
    if result["sources"]:
        answer += "\n\n**المصادر:**\n" + "\n".join(f"- {s}" for s in result["sources"])
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
        lines.append(f"✅ {src.name}: أضيف {n} مقطع")
    retriever.refresh()  # rebuild the keyword index with the new chunks
    return "\n".join(lines)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{SUBTITLE}")
    with gr.Tab("المحادثة"):
        chatbot = gr.Chatbot(height=480, rtl=True, label="المحادثة")
        with gr.Row():
            box = gr.Textbox(placeholder="اكتب سؤالك هنا...", scale=5, show_label=False, rtl=True)
            send = gr.Button("إرسال", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=box)
        clear = gr.Button("محادثة جديدة")
        send.click(respond, [box, chatbot], [box, chatbot])
        box.submit(respond, [box, chatbot], [box, chatbot])
        clear.click(lambda: [], None, chatbot)
    with gr.Tab("إضافة مستندات"):
        files = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], label="ارفع ملفات PDF أو TXT")
        status = gr.Markdown()
        files.upload(upload, files, status)
    gr.Markdown("⚠️ الإجابات مبنية على المستندات المتاحة فقط وقد تحتوي أخطاء. تحقّق دائمًا من المصدر.")


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(), inbrowser=True)  # opens the page automatically
