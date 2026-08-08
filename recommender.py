"""
recommender.py
----------------
Core, reusable logic for the Anime Content-Based Recommender System.

This module is imported by BOTH:
  1. The Jupyter notebook (notebooks/Anime_Recommendation_System.ipynb) - for EDA/experimentation
  2. The Streamlit app (app.py) - for serving recommendations

Keeping the logic in one place means the model the notebook builds is
exactly the model the app serves (no drift between "research" and "production").
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
import joblib
import os

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# --------------------------------------------------------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Kaggle anime.csv file."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw anime dataframe:
      - 'episodes' has literal 'Unknown' strings -> convert to numeric NaN
      - Missing genre/type -> fill with 'Unknown' (small % of rows, safe to keep)
      - Missing rating -> impute with the median rating for that anime's 'type'
      - Missing episodes -> impute with the median episodes for that anime's 'type'
      - Drop exact duplicate anime_id rows if any
      - Reset index
    """
    df = df.copy()

    # episodes: 'Unknown' -> NaN -> numeric
    df["episodes"] = df["episodes"].replace("Unknown", np.nan)
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")

    # genre / type: small number of missing values -> fill with 'Unknown'
    df["genre"] = df["genre"].fillna("Unknown")
    df["type"] = df["type"].fillna("Unknown")

    # rating: impute using median rating within the same 'type' (TV shows rate
    # differently to Movies/Specials on average, so group-wise median is more
    # faithful than a single global median)
    df["rating"] = df.groupby("type")["rating"].transform(
        lambda s: s.fillna(s.median())
    )
    # any leftover NaNs (a whole 'type' group with no ratings at all) -> global median
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # episodes: same group-wise imputation strategy
    df["episodes"] = df.groupby("type")["episodes"].transform(
        lambda s: s.fillna(s.median())
    )
    df["episodes"] = df["episodes"].fillna(df["episodes"].median())

    df = df.drop_duplicates(subset="anime_id").reset_index(drop=True)

    return df


def add_transformed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    'members' and 'episodes' are strongly right-skewed (a handful of mega-hit
    anime and a handful of extremely long-running series dominate the range).
    Add log1p-transformed versions used later for skewness comparison and as
    (optional) numeric features for the recommender.
    """
    df = df.copy()
    df["members_log"] = np.log1p(df["members"])
    df["episodes_log"] = np.log1p(df["episodes"])
    return df


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING FOR THE CONTENT-BASED MODEL
# --------------------------------------------------------------------------
def _clean_genre_string(genre: str) -> str:
    """
    Turn 'Action, Adventure, Comedy' into 'Action Adventure Comedy' while
    keeping multi-word genre tags (e.g. 'Sci-Fi', 'Slice of Life') intact as
    single tokens, so the vectorizer doesn't split them incorrectly.
    """
    tags = [t.strip().replace(" ", "_").replace("-", "_") for t in genre.split(",")]
    return " ".join(tags)


def build_soup(df: pd.DataFrame) -> pd.Series:
    """
    Build the 'content soup' text used by the TF-IDF vectorizer:
    genre tags + anime type (TV/Movie/OVA/...), each genre/type kept as a
    single token. This is the classic content-based approach: two anime are
    'similar' if they share genres and format.
    """
    genre_clean = df["genre"].apply(_clean_genre_string)
    type_clean = df["type"].fillna("Unknown").str.replace(" ", "_")
    soup = genre_clean + " " + type_clean
    return soup


def build_content_model(df: pd.DataFrame):
    """
    Fit the content-based model:
      1. TF-IDF vectorize the genre+type 'soup'
      2. Fit a NearestNeighbors index (cosine distance) over the TF-IDF matrix

    Returns (tfidf_vectorizer, tfidf_matrix, nn_model)

    Why NearestNeighbors instead of a full cosine_similarity matrix?
    A dense NxN similarity matrix for ~12,000 anime would be ~12000x12000
    floats (~1GB+) which is wasteful to store/ship with the Streamlit app.
    NearestNeighbors on the sparse TF-IDF matrix answers "top-k similar"
    queries on demand in milliseconds with a tiny memory footprint.
    """
    soup = build_soup(df)
    vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")  # tokens are already space-separated
    tfidf_matrix = vectorizer.fit_transform(soup)

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    return vectorizer, tfidf_matrix, nn_model


# --------------------------------------------------------------------------
# 3. RECOMMENDATION LOGIC
# --------------------------------------------------------------------------
def get_recommendations(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix,
    nn_model,
    top_n: int = 10,
    min_members: int = 0,
    anime_type: str = "Any",
):
    """
    Return the top_n anime most similar (by genre/type content) to `title`.

    Parameters
    ----------
    title : exact (case-insensitive) or partial match against df['name']
    df : cleaned anime dataframe (same row order used to fit tfidf_matrix!)
    tfidf_matrix : fitted TF-IDF matrix from build_content_model
    nn_model : fitted NearestNeighbors model from build_content_model
    top_n : number of recommendations to return
    min_members : optional popularity filter (only recommend anime with
                  at least this many community members)
    anime_type : optional filter, one of 'Any', 'TV', 'Movie', 'OVA', ...

    Returns
    -------
    pandas.DataFrame of recommended anime (or an empty DataFrame + message
    if the title isn't found)
    """
    matches = df[df["name"].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df["name"].str.lower().str.contains(re.escape(title.lower()), na=False)]
    if matches.empty:
        return pd.DataFrame(), f"No anime found matching '{title}'."

    idx = matches.index[0]
    matched_name = df.loc[idx, "name"]

    # Ask for extra neighbors so we still have top_n left after filtering out
    # the anime itself and applying the type/popularity filters.
    n_query = min(top_n + 50, tfidf_matrix.shape[0])
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n_query)

    rec_idx = [i for i in indices.flatten() if i != idx]
    rec_dist = [d for d, i in zip(distances.flatten(), indices.flatten()) if i != idx]

    recs = df.iloc[rec_idx].copy()
    recs["similarity"] = 1 - np.array(rec_dist)  # cosine distance -> similarity

    if anime_type != "Any":
        recs = recs[recs["type"] == anime_type]
    if min_members > 0:
        recs = recs[recs["members"] >= min_members]

    recs = recs.sort_values("similarity", ascending=False).head(top_n)
    recs = recs[["name", "genre", "type", "episodes", "rating", "members", "similarity"]]
    recs = recs.reset_index(drop=True)

    return recs, matched_name


# --------------------------------------------------------------------------
# 4. PERSISTENCE (used by the notebook to save, and app.py to load)
# --------------------------------------------------------------------------
def save_artifacts(df, vectorizer, tfidf_matrix, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(df, os.path.join(models_dir, "anime_clean.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    sparse.save_npz(os.path.join(models_dir, "tfidf_matrix.npz"), tfidf_matrix)
    print(f"Artifacts saved to {models_dir}")


def load_artifacts(models_dir: str):
    df = joblib.load(os.path.join(models_dir, "anime_clean.pkl"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    tfidf_matrix = sparse.load_npz(os.path.join(models_dir, "tfidf_matrix.npz"))

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    return df, vectorizer, tfidf_matrix, nn_model


if __name__ == "__main__":
    # Quick smoke test / can also be run as a standalone script to (re)build
    # the model artifacts used by the Streamlit app:
    #   python recommender.py
    RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "anime.csv")
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

    print("Loading raw data...")
    raw = load_raw_data(RAW_PATH)
    print("Cleaning...")
    clean = clean_data(raw)
    clean = add_transformed_features(clean)
    print("Building content model...")
    vec, mat, nn = build_content_model(clean)
    print("Saving artifacts...")
    save_artifacts(clean, vec, mat, MODELS_DIR)

    print("\nSmoke test: recommendations for 'Naruto'")
    recs, matched = get_recommendations("Naruto", clean, mat, nn, top_n=5)
    print("Matched:", matched)
    print(recs)
