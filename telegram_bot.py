"""Telegram interface: the same RAG system, reachable from a phone.

Setup: create a bot with @BotFather in Telegram, then put the token in .env:
    TELEGRAM_BOT_TOKEN=...
Run:
    python telegram_bot.py
"""
import asyncio
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from generator import HISTORY_TURNS, ask

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

WELCOME = ("أهلًا! 👋 أنا المرشد الأكاديمي الذكي لجامعة بوليتكنك فلسطين.\n"
           "أجيب عن أسئلتك حول تعليمات الدراسة والخطط الدراسية والامتحانات من مستندات الجامعة مع ذكر المصدر.\n"
           "اكتب سؤالك بالفصحى أو باللهجة.\n"
           "/new لبدء محادثة جديدة")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["history"] = []
    await update.message.reply_text(WELCOME)


async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["history"] = []
    await update.message.reply_text("بدأنا محادثة جديدة ✅")


async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    history = context.user_data.setdefault("history", [])  # each user has their own history
    await update.message.chat.send_action("typing")
    try:
        # ask() is slow (search + model), so run it in a thread to keep the bot responsive
        result = await asyncio.to_thread(ask, question, history)
    except Exception as e:  # e.g. missing API key or no internet
        print("Error:", e)
        await update.message.reply_text("⚠️ تعذّر الاتصال بالموديل حاليًا. حاول مرة أخرى بعد قليل.")
        return
    reply = result["answer"]
    if result["sources"]:
        reply += "\n\n📄 المصادر:\n" + "\n".join(result["sources"])
    history.append((question, result["answer"]))
    del history[:-HISTORY_TURNS]  # keep only the last turns
    await update.message.reply_text(reply)


def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing in .env")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_chat))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))
    print("Bot is running... (Ctrl+C to stop)")
    app.run_polling()


if __name__ == "__main__":
    main()
