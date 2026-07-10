Before start pls use these instruction
 🤖 Sentiment Analyzer

An end-to-end sentiment analysis web app — built from scratch, not wrapped
around an external AI API. Type any sentence, or paste a YouTube video link,
and see it classified as **Positive 😊 / Neutral 😐 / Negative 😞** by a
machine learning model trained on labeled data.

 ✨ Features

Custom-trained ML model TF-IDF features + 4 candidate algorithms
  (Logistic Regression, Naive Bayes, Linear SVM, Random Forest), best one
  auto-selected by accuracy
Text sentiment analysis: instant classification with a confidence
  breakdown shown as segmented level-meter bars
YouTube comment analysis: paste any video URL, get sentiment scored
  across all its top-level comments, with an aggregate summary
Live model-accuracy: badge pulled directly from the last training
  run, not hardcoded
NLP pipeline: NLTK tokenization, stopword removal (with negation
  words like "not"/"never" deliberately preserved), lemmatization
FastAPI backend:clean REST API, JSON in/out
No external AI APIs used for inference: the model is 100% self-trained

 🖥️ Tech stack

Python · Pandas · NumPy · scikit-learn · NLTK · FastAPI · HTML/CSS/JavaScript
 📁 Project structure

```
Sentiment-Analysis/
├── backend/
│   ├── main.py              # FastAPI app & routes
│   ├── utils.py             # shared text preprocessing
│   ├── youtube_service.py   # YouTube Data API integration
│   ├── model.pkl            # trained classifier
│   ├── vectorizer.pkl       # fitted TF-IDF vectorizer
│   ├── model_info.json      # live accuracy/metrics
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── dataset/
│   ├── build_dataset.py
│   └── sentiment.csv
├── training/
│   ├── train_model.py       # full pipeline: EDA → clean → train → evaluate
│   ├── train_model.ipynb    # same pipeline, runnable in Google Colab
│   └── label_distribution.png
├── test_cases.py
└── requirements.txt
```

 🚀 Getting started

 1. Clone the repo
```bash
git clone https://github.com//Sentiment-Analysis.git
cd Sentiment-Analysis
```

 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

 4. (Optional) Set up YouTube comment analysis
Get a free API key from [Google Cloud Console](https://console.cloud.google.com/)
(enable **YouTube Data API v3** → create an API key), then:
```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```
Paste your key into `.env`:
```
YOUTUBE_API_KEY=your_key_here
```

 5. Train the model
```bash
cd ..
python dataset/build_dataset.py
python training/train_model.py
```
 6. Run the tests
```bash
python test_cases.py
```

 7. Start the server
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

 8. Open the app
```
http://127.0.0.1:8000
```

 📊 Model performance

| Model | Accuracy |
|---|---|
|Random Forest |⭐99.45%|
| Logistic Regression | 99.18% |
| Linear SVM | 98.91% |
| Multinomial Naive Bayes | 98.36% |

*(Regenerated automatically every time `train_model.py` runs — see `backend/model_info.json`)*

 ⚠️ Limitations

The training dataset is synthetically generated for demonstration purposes,
so these accuracy numbers reflect a clean, controlled test set — not
necessarily messy real-world text (sarcasm, slang, mixed sentiment). For
production use, swap in a real-world dataset (e.g. IMDB, Sentiment140) and/or
a transformer-based model.
