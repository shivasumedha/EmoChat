import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import random
import re
import warnings

warnings.filterwarnings("ignore")


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


class EmotionChatBot:
    def __init__(self):
        self.model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[EmoChat] Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[EmoChat] Model ready.")

        self.emotion_map = {
            "joy":        "joy",
            "sadness":    "sadness",
            "anger":      "anger",
            "fear":       "fear",
            "love":       "love",
            "surprise":   "surprise",
            "annoyance":  "annoyance",
            "disgust":    "disgust",
            "gratitude":  "gratitude",
        }

        self.emotion_emoji = {
            "joy":        "😄",
            "sadness":    "😢",
            "anger":      "😠",
            "fear":       "😨",
            "love":       "❤️",
            "surprise":   "😲",
            "annoyance":  "😤",
            "disgust":    "🤢",
            "gratitude":  "🙏",
            "neutral":    "😐",
        }

        self.emotion_color = {
            "joy":        "#FFD700",
            "sadness":    "#6495ED",
            "anger":      "#FF4500",
            "fear":       "#9370DB",
            "love":       "#FF69B4",
            "surprise":   "#FFA500",
            "annoyance":  "#CD853F",
            "disgust":    "#6B8E23",
            "gratitude":  "#40E0D0",
            "neutral":    "#A9A9A9",
        }

        raw = load_json("solmate_response.json")

        # Clean responses: deduplicate and strip legacy "(N)" suffixes
        self.responses = {}
        for emotion, lines in raw.items():
            cleaned = list({re.sub(r'\s*\(\d+\)\s*$', '', l).strip() for l in lines})
            self.responses[emotion] = cleaned

        self.history = []

    def process(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Top-2 emotions for richer context
        top2 = probs[0].topk(2)
        confidence  = round(top2.values[0].item() * 100, 1)
        raw_label   = self.model.config.id2label[top2.indices[0].item()]
        emotion     = self.emotion_map.get(raw_label, "neutral")

        if emotion not in self.responses:
            emotion = "neutral"

        response = random.choice(self.responses[emotion])

        result = {
            "emotion":    emotion,
            "emoji":      self.emotion_emoji.get(emotion, "😐"),
            "color":      self.emotion_color.get(emotion, "#A9A9A9"),
            "confidence": confidence,
            "response":   response,
        }

        self.history.append({"role": "user", "text": text})
        self.history.append({"role": "bot",  **result})
        return result

    def get_history(self) -> list:
        return self.history

    def clear_history(self):
        self.history = []

    def export_txt(self) -> str:
        lines = ["EmoChat — Chat Export\n" + "=" * 40]
        for entry in self.history:
            if entry["role"] == "user":
                lines.append(f"\nYou: {entry['text']}")
            else:
                lines.append(
                    f"EmoChat [{entry['emoji']} {entry['emotion'].capitalize()} {entry['confidence']}%]: {entry['response']}"
                )
        return "\n".join(lines)
