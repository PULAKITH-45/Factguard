from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from model import get_prediction

app = FastAPI()

# 🔹 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 API KEY
GNEWS_API_KEY = "2d2bcb35ddf0400ae490667e858bae7b"

# 🔹 Schema
class NewsInput(BaseModel):
    text: str

# 🔹 GNews function
def check_news(query):
    url = "https://gnews.io/api/v4/search"

    params = {
        "q": query,
        "lang": "en",
        "max": 3,
        "apikey": GNEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        articles = data.get("articles", [])

        results = [
            {
                "title": a["title"],
                "url": a["url"]
            }
            for a in articles
        ]

        return len(results) > 0, results

    except Exception:
        return False, []

# 🔹 Root
@app.get("/")
def home():
    return {"message": "FactGuard Running"}

# 🔹 Predict (FINAL LOGIC)
@app.post("/predict")
def predict(news: NewsInput):

    text = news.text.lower()

    # =========================
    # 1. RULE-BASED FILTER
    # =========================
    fake_keywords = ["alien", "dragon", "time travel", "invisible"]

    if any(word in text for word in fake_keywords):
        return {
            "input_text": news.text,
            "prediction": "NOT VERIFIED",
            "confidence": 95,
            "source": None
        }

    # =========================
    # 2. NEWS API CHECK (PRIORITY)
    # =========================
    query = " ".join(text.split()[:5])
    found, articles = check_news(query)

    if found:
        return {
            "input_text": news.text,
            "prediction": "REAL",
            "confidence": 90,
            "source": articles
        }

    # =========================
    # 3. ML MODEL (ONLY SUPPORT)
    # =========================
    try:
        result = get_prediction(text)

        # 🔥 IMPORTANT: NEVER TRUST ML ALONE
        if result["prediction"] == "REAL":
            return {
                "input_text": news.text,
                "prediction": "NOT VERIFIED",
                "confidence": result["confidence"],
                "source": None
            }

        return {
            "input_text": news.text,
            "prediction": "NOT VERIFIED",
            "confidence": result["confidence"],
            "source": None
        }

    except Exception:
        return {
            "input_text": news.text,
            "prediction": "NOT VERIFIED",
            "confidence": 50,
            "source": None
        }