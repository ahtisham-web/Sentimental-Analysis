# Sentiment Analysis — Full Project (13-Phase Spec)

An end-to-end sentiment analysis app: dataset generation → EDA → cleaning →
preprocessing (stemming *and* lemmatization) → feature engineering
(Bag-of-Words *and* TF-IDF) → model training (4 algorithms) → evaluation →
FastAPI backend → HTML/CSS/JS frontend → testing.

## Project structure

```
Sentiment-Analysis/
├── backend/
│   ├── main.py            # FastAPI app (/api/predict, /api/youtube/analyze, /api/health)
│   ├── utils.py           # shared preprocessing (used by backend AND training)
│   ├── youtube_service.py # YouTube Data API comment fetching
│   ├── model.pkl          # trained classifier (generated)
│   ├── vectorizer.pkl     # fitted TF-IDF vectorizer (generated)
│   ├── .env.example       # template for YOUTUBE_API_KEY
│   └── requirements.txt
├── frontend/
│   ├── index.html         # now has Text / YouTube tabs
│   ├── style.css
│   └── script.js
├── dataset/
│   ├── build_dataset.py
│   └── sentiment.csv
├── training/
│   ├── train_model.py           # full pipeline, phases 2, 4, 5, 6, 7, 8
│   ├── train_model.ipynb        # same pipeline, runnable in Google Colab
│   └── label_distribution.png   # EDA chart (generated)
├── test_cases.py           # Phase 12 formal test cases
├── .gitignore
└── README.md
```

## Phase-by-phase status — what's done, what's manual

| Phase | Status | Notes |
|---|---|---|
| 1. Environment Setup | ⚠️ Partial | Libraries + Colab notebook done. **GitHub repo creation is a manual step** — see below, I can't create your GitHub account/repo for you. |
| 2. Dataset Exploration | ✅ Done | `train_model.py` prints head/info/missing values/class counts and saves `label_distribution.png`. |
| 3. Data Cleaning | ✅ Done | Lowercase, punctuation/URL/number/special-char removal, whitespace collapse — `utils.py::clean_text`. |
| 4. Text Preprocessing | ✅ Done | Tokenization, stopword removal (negations kept on purpose), **both** stemming and lemmatization (compared side-by-side in training output; lemmatization used in the production model). |
| 5. Feature Engineering | ✅ Done | **Both** Bag-of-Words and TF-IDF trained and compared; TF-IDF used in production (see training output for why). |
| 6. Model Training | ✅ Done | Logistic Regression, Naive Bayes, LinearSVC, **and Random Forest** all trained and compared; best one auto-selected. |
| 7. Model Evaluation | ✅ Done | Accuracy, precision, recall, F1, confusion matrix printed by `train_model.py`. |
| 8. Save the Model | ✅ Done | `pickle` used (not joblib — functionally equivalent for this use case; I can switch if your grader specifically requires joblib). |
| 9. Backend Development | ✅ Done | FastAPI, POST `/api/predict`, JSON in/out. |
| 10. Frontend Development | ✅ Done | Text input, predict button, **real loading spinner**, prediction card, confidence score. |
| 11. Integration | ✅ Done | `fetch()` in `script.js` → FastAPI → JSON → rendered in UI. |
| 12. Testing | ✅ Done | `test_cases.py` runs all 5 required cases directly against the model. **Honest result: 4/5 passed** — the long-paragraph case predicted negative instead of positive. This is a real finding, not hidden: mixed-sentiment paragraphs (starts skeptical, ends positive) are genuinely hard for a TF-IDF+linear model with no sequence understanding. Noted under Limitations below. |
| 13. Deployment | ❌ Not done | **Requires your accounts/credentials** — I've included ready-to-use deployment configs and step-by-step instructions below, but actually deploying needs you to create accounts on a hosting platform and a GitHub repo. |

## What I could not do for you (and why)

- **Create a GitHub repository** — this needs your GitHub account/login.
- **Deploy the app live** — this needs a hosting account (Render, Railway, etc.) that only you can create and authorize.

Both are quick to do yourself — instructions below.

## Running locally

```bash
cd backend
pip install -r requirements.txt
```

Train (from the project root):
```bash
python dataset/build_dataset.py
python training/train_model.py
```

Run tests:
```bash
python test_cases.py
```

Start the server:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://127.0.0.1:8000**.

## Pushing to GitHub (do this yourself, ~2 minutes)

```bash
cd Sentiment-Analysis
git init
git add .
git commit -m "Initial commit: full sentiment analysis pipeline"
```
Then on github.com: click **New repository**, name it (e.g. `sentiment-analysis`), don't initialize with a README (you already have one), then run the two commands it shows you, e.g.:
```bash
git remote add origin https://github.com/<your-username>/sentiment-analysis.git
git branch -M main
git push -u origin main
```

## Deploying (do this yourself — needs your accounts)

**Backend (Render.com, free tier):**
1. Push this repo to GitHub (above).
2. Go to render.com → New → Web Service → connect your GitHub repo.
3. Root directory: `backend`. Build command: `pip install -r requirements.txt`.
   Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Deploy — Render gives you a public URL like `https://your-app.onrender.com`.

Since `main.py` already serves the frontend directly (`/` route), the single
Render deployment covers both backend and frontend — no separate frontend
hosting needed. Just share the Render URL.

## YouTube comment sentiment analysis (added feature)

The app can also fetch comments from any YouTube video and run them through
the same trained model.

### Setup

1. Get a free YouTube Data API v3 key: console.cloud.google.com → create/select
   a project → **APIs & Services → Library** → enable "YouTube Data API v3" →
   **APIs & Services → Credentials → Create Credentials → API key**.
2. In `backend/`, copy `.env.example` to `.env`:
   ```bash
   cd backend
   cp .env.example .env
   ```
3. Edit `.env` and paste your key:
   ```
   YOUTUBE_API_KEY=AIzaSyA9xaZEeIXvZ-o4vbo9WDwiK9lfyQCmXnQ
   ```
4. Reinstall requirements (adds `requests` and `python-dotenv`):
   ```bash
   pip install -r requirements.txt
   ```
5. Start the server as usual — the "YouTube Comments" tab in the UI will now work.

### API

```
POST /api/youtube/analyze
{ "video_url": "https://www.youtube.com/watch?v=VIDEO_ID", "max_comments": 50 }

-> {
  "video_id": "VIDEO_ID",
  "total_comments_analyzed": 50,
  "sentiment_summary": { "positive": 62.0, "neutral": 20.0, "negative": 18.0 },
  "comments": [
    { "author": "...", "text": "...", "like_count": 12,
      "label": "positive", "emoji": "😊", "confidence": 0.87 },
    ...
  ]
}
```

Accepts full URLs (`youtube.com/watch?v=`, `youtu.be/`, `/shorts/`) or a bare
11-character video ID.

### Notes

- **Quota**: YouTube's free tier gives 10,000 units/day; fetching comments
  costs ~1 unit per 100 comments, so this is very cheap to run.
- **Common errors handled gracefully** (not a crash): comments disabled on
  the video, invalid/missing API key, video not found, quota exhausted —
  all return a clear JSON error message instead of a 500.
- **Never commit your `.env` file** — it's already in `.gitignore`.
- This was tested with a mocked YouTube API response (since this environment
  can't reach googleapis.com) — the parsing, aggregation, and error-handling
  logic is verified, but you should do one real end-to-end test with your own
  API key before considering it fully validated.



The dataset is template-generated for a clean teaching example, so the model
scores ~99% on its own held-out test — a property of the dataset being easy,
not a guarantee on messy real text. The failed long-paragraph test case above
is real evidence of this: sentences with a mixed emotional arc (skeptical →
positive) can fool a TF-IDF + linear/tree model since it has no sense of
sentence order or contrast ("skeptical... but"). For production-grade
robustness, swap in a real-world dataset (IMDB, Sentiment140) and/or a
transformer-based model (e.g. fine-tuned DistilBERT).
