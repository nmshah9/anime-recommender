# 🎌 Anime Recommendation System — Content-Based Filtering

End-to-end project built on the [Kaggle "Anime Recommendations Database"](https://www.kaggle.com/code/benroshan/content-collaborative-anime-recommendation)
(`anime.csv`), with full EDA, skewness/transformation analysis, a content-based
recommender, and a Streamlit app to serve it.

## Project structure

```
anime_recsys/
├── data/
│   └── anime.csv                          # raw dataset (anime_id, name, genre, type, episodes, rating, members)
├── notebooks/
│   └── Anime_Recommendation_System.ipynb  # EDA, skewness analysis, model build (fully executed, no errors)
├── models/                                # generated model artifacts (created by notebook or recommender.py)
│   ├── anime_clean.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_matrix.npz
├── recommender.py                         # shared data-cleaning + recommender logic (used by notebook AND app)
├── app.py                                 # Streamlit app
├── requirements.txt
└── README.md
```

## Why one shared `recommender.py`?

Both the notebook and the Streamlit app import the same `recommender.py` module for
data cleaning, feature engineering, and the recommendation function. This guarantees
the model you explore/validate in the notebook is *exactly* the model the app serves —
no drift between "research" and "production" code.

## Setup

```bash
cd anime_recsys
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Run / explore the notebook

```bash
jupyter notebook notebooks/Anime_Recommendation_System.ipynb
```

This notebook:
1. Loads `anime.csv`
2. Performs EDA (missing values, categorical + numeric distributions, correlations, boxplots)
3. Computes skewness (`scipy.stats.skew`) for `rating`, `episodes`, `members` and applies a
   `log1p` transform to the highly-skewed `episodes` and `members` columns
4. Builds a **content-based recommender**: TF-IDF over genre+type tags → cosine `NearestNeighbors`
5. Saves the trained artifacts to `models/` for the app to consume

Running the last "Persist Model Artifacts" cell (or the whole notebook) regenerates `models/`.

## 2. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with two pages:
- **🔍 Get Recommendations** — pick any anime title, get the top-N most similar anime
  (by genre + format), with optional type/popularity filters
- **📊 Dataset Explorer** — interactive recap of the EDA (type/genre distributions, rating
  distribution, raw-vs-log members skewness comparison)

If `models/` doesn't exist yet, the app will build it automatically on first load
(reading `data/anime.csv`) — you don't have to run the notebook first, though doing so
is recommended to see the full analysis.

## Rebuilding the model from the command line

```bash
python recommender.py
```

This re-runs the full clean → build → save pipeline and prints a quick smoke-test
recommendation, without needing Jupyter or Streamlit.

## Notes / scope

- The dataset provided only contains anime metadata (`anime.csv`), not the separate
  `rating.csv` user-ratings file used for collaborative filtering in the original Kaggle
  notebook — so this project implements the **content-based** half only, as requested.
- Recommendations are generated with `NearestNeighbors` (cosine metric) over a TF-IDF
  matrix rather than a full N×N similarity matrix, to keep memory/startup time small
  enough to run comfortably in Streamlit.
