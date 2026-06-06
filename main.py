from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app import EmotionChatBot
import io, os, sys

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[EmoChat] reportlab not installed. Run: pip install reportlab")

app = FastAPI(title="EmoChat API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Bot loaded once here — NOT inside __main__ block
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
    import socket

    PORT = 8000

    # Auto-kill any process already using the port
    def free_port(port):
        import subprocess
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True, capture_output=True, text=True
        )
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if parts and parts[-1].isdigit():
                pid = parts[-1]
                subprocess.run(f'taskkill /PID {pid} /F', shell=True,
                               capture_output=True)
                print(f"[EmoChat] Freed port {port} (killed PID {pid})")
                break

    # Check if port is in use, free it if so
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            print(f"[EmoChat] Port {PORT} busy — freeing it...")
            free_port(PORT)
            import time
            time.sleep(1)

    print(f"[EmoChat] Starting server on http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, reload=False)
