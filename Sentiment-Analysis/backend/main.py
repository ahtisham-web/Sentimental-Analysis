"""
FastAPI backend for the Sentiment Analysis app.
Loads the trained TF-IDF vectorizer + ML model and exposes a /predict endpoint.
"""
import pickle
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from utils import preprocess_text
import youtube_service

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # loads YOUTUBE_API_KEY if present

MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="Sentiment Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

try:
    with open(MODEL_INFO_PATH, "r") as f:
        MODEL_INFO = json.load(f)
except FileNotFoundError:
    MODEL_INFO = {"model_name": model.__class__.__name__, "accuracy": None}

EMOJI_MAP = {"positive": "😊", "neutral": "😐", "negative": "😞"}


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str
    label: str
    emoji: str
    confidence: float
    probabilities: dict


def predict_sentiment(text: str) -> dict:
    """Shared prediction logic, reused by /api/predict and /api/youtube/analyze."""
    text = text.strip()
    if not text:
        return {
            "prediction": "Neutral", "label": "neutral", "emoji": "😐",
            "confidence": 0.0, "probabilities": {"positive": 0, "neutral": 0, "negative": 0},
        }

    cleaned = preprocess_text(text, method="lemmatize")
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = model.classes_
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        confidence = float(max(proba))
    else:
        prob_dict = {label: 1.0}
        confidence = 1.0

    return {
        "prediction": label.capitalize(),
        "label": label,
        "emoji": EMOJI_MAP.get(label, ""),
        "confidence": confidence,
        "probabilities": prob_dict,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    return MODEL_INFO


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    return predict_sentiment(payload.text)


class YouTubeRequest(BaseModel):
    video_url: str
    max_comments: int = 50


class CommentResult(BaseModel):
    author: str
    text: str
    like_count: int
    label: str
    emoji: str
    confidence: float


class YouTubeResponse(BaseModel):
    video_id: str
    total_comments_analyzed: int
    sentiment_summary: dict
    comments: List[CommentResult]


@app.post("/api/youtube/analyze", response_model=YouTubeResponse)
def analyze_youtube_comments(payload: YouTubeRequest):
    try:
        video_id = youtube_service.extract_video_id(payload.video_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        comments = youtube_service.fetch_comments(video_id, max_comments=payload.max_comments)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this video.")

    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for c in comments:
        pred = predict_sentiment(c["text"])
        counts[pred["label"]] += 1
        results.append(CommentResult(
            author=c["author"],
            text=c["text"],
            like_count=c["like_count"],
            label=pred["label"],
            emoji=pred["emoji"],
            confidence=pred["confidence"],
        ))

    total = len(results)
    summary = {k: round(v / total * 100, 1) for k, v in counts.items()}

    return YouTubeResponse(
        video_id=video_id,
        total_comments_analyzed=total,
        sentiment_summary=summary,
        comments=results,
    )


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
