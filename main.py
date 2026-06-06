from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app import EmotionChatBot
import io, os

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[EmoChat] reportlab not installed.")

app = FastAPI(title="EmoChat API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

bot = EmotionChatBot()

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return bot.process(req.message.strip())

@app.get("/history")
def get_history():
    return {"history": bot.get_history()}

@app.post("/clear")
def clear_history():
    bot.clear_history()
    return {"status": "cleared"}

@app.get("/export/txt")
def export_txt():
    return PlainTextResponse(
        content=bot.export_txt(),
        headers={"Content-Disposition": "attachment; filename=emochat_history.txt"}
    )

@app.get("/export/pdf")
def export_pdf():
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Install reportlab: pip install reportlab")
    history = bot.get_history()
    if not history:
        raise HTTPException(status_code=400, detail="No chat history to export.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=18,
        textColor=colors.HexColor("#A78BFA"), spaceAfter=6)
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#888888"), spaceAfter=12)
    user_s  = ParagraphStyle("U", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#E2E8F0"), backColor=colors.HexColor("#1E293B"),
        borderPadding=(6,10,6,10), spaceBefore=8, spaceAfter=4)
    bot_s   = ParagraphStyle("B", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#C4B5FD"), backColor=colors.HexColor("#0F172A"),
        borderPadding=(6,10,6,10), spaceBefore=2, spaceAfter=8)

    story = [Paragraph("EmoChat — Chat Export", title_s),
             Paragraph("Your emotional support conversation", sub_s)]
    for entry in history:
        if entry["role"] == "user":
            story.append(Paragraph(f"<b>You:</b> {entry['text']}", user_s))
        else:
            label = f"{entry['emoji']} {entry['emotion'].capitalize()} · {entry['confidence']}%"
            story.append(Paragraph(
                f"<b>EmoChat</b> <font size='9' color='#888'>[{label}]</font><br/>{entry['response']}",
                bot_s))
    doc.build(story)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=emochat_history.pdf"})


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is required by Hugging Face Spaces
    PORT = int(os.environ.get("PORT", 7860))
    print(f"[EmoChat] Starting server on http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
