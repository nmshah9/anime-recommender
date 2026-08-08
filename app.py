"""
Streamlit App - Anime Content-Based Recommender System
--------------------------------------------------------
Run locally with:
    streamlit run app.py

The app loads pre-computed model artifacts from ./models (built by running
`python recommender.py` once, or by executing the notebook). It never
re-trains the model on every page load - only on demand via the sidebar
button - so the UI stays fast.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from recommender import (
    load_raw_data,
    clean_data,
    add_transformed_features,
    build_content_model,
    save_artifacts,
    load_artifacts,
    get_recommendations,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "anime.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
)

# ============================================================
# LOAD BANNER
# ============================================================

banner = Image.open("banner.png")

# ============================================================
# DISPLAY BANNER
# ============================================================

st.image(banner, use_container_width=True)

# --------------------------------------------------------------------------
# Cached loaders - Streamlit reruns the whole script on every interaction,
# so caching prevents re-loading/re-fitting the model on every click.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommender model...")
def get_model():
    """
    Load pre-built artifacts if they exist; otherwise build them once from
    the raw CSV (first run, or if /models was deleted) and cache the result
    for the lifetime of the app process.
    """
    required = ["anime_clean.pkl", "tfidf_vectorizer.pkl", "tfidf_matrix.npz"]
    if all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required):
        df, vectorizer, tfidf_matrix, nn_model = load_artifacts(MODELS_DIR)
    else:
        raw = load_raw_data(DATA_PATH)
        df = clean_data(raw)
        df = add_transformed_features(df)
        vectorizer, tfidf_matrix, nn_model = build_content_model(df)
        save_artifacts(df, vectorizer, tfidf_matrix, MODELS_DIR)
    return df, vectorizer, tfidf_matrix, nn_model


def rebuild_model():
    st.cache_resource.clear()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🎌 Anime Recommender")
st.sidebar.markdown(
    "Content-based recommendations using **genre + format (TV/Movie/OVA...) "
    "similarity** (TF-IDF + cosine nearest-neighbors)."
)

page = st.sidebar.radio("Navigate", ["🔍 Get Recommendations", "📊 Dataset Explorer"])

if st.sidebar.button("🔄 Rebuild model from raw data"):
    rebuild_model()
    st.sidebar.success("Cache cleared - model will rebuild on next load.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: MyAnimeList anime metadata (anime_id, name, genre, type, "
    "episodes, rating, members)."
)

df, vectorizer, tfidf_matrix, nn_model = get_model()


# --------------------------------------------------------------------------
# Page 1: Recommendations
# --------------------------------------------------------------------------
if page == "🔍 Get Recommendations":
    st.title("Find anime similar to your favorite")

    all_names = sorted(df["name"].unique().tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        title_input = st.selectbox(
            "Pick an anime (or type to search)",
            options=all_names,
            index=all_names.index("Naruto") if "Naruto" in all_names else 0,
        )
    with col2:
        top_n = st.slider("Number of recommendations", 5, 25, 10)

    with st.expander("Advanced filters"):
        c1, c2 = st.columns(2)
        with c1:
            type_filter = st.selectbox(
                "Restrict to type", ["Any"] + sorted(df["type"].dropna().unique().tolist())
            )
        with c2:
            min_members = st.number_input(
                "Minimum popularity (members)", min_value=0, value=0, step=1000
            )

    if st.button("Get Recommendations", type="primary"):
        result = get_recommendations(
            title_input,
            df,
            tfidf_matrix,
            nn_model,
            top_n=top_n,
            min_members=min_members,
            anime_type=type_filter,
        )
        recs, matched = result

        if recs.empty:
            st.warning(matched)
        else:
            st.success(f"Because you liked **{matched}**, you might enjoy:")

            display = recs.copy()
            display["similarity"] = (display["similarity"] * 100).round(1).astype(str) + "%"
            display.columns = [
                "Name", "Genre", "Type", "Episodes", "Rating", "Members", "Similarity",
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)

            fig = px.bar(
                recs.sort_values("similarity"),
                x="similarity",
                y="name",
                orientation="h",
                color="rating",
                color_continuous_scale="Viridis",
                labels={"similarity": "Content Similarity", "name": "", "rating": "Rating"},
                title="Similarity score of recommended titles",
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Page 2: Dataset Explorer / EDA recap
# --------------------------------------------------------------------------
else:
    st.title("📊 Dataset Explorer")
    st.caption("A quick interactive recap of the EDA performed in the notebook.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total anime", f"{len(df):,}")
    m2.metric("Unique genres", df["genre"].str.split(", ").explode().nunique())
    m3.metric("Avg rating", f"{df['rating'].mean():.2f}")
    m4.metric("Median members", f"{int(df['members'].median()):,}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Type distribution", "Rating distribution", "Members (skewness)", "Top genres"]
    )

    with tab1:
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig = px.bar(type_counts, x="type", y="count", color="type", title="Anime count by type")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x="rating", nbins=40, title="Distribution of ratings")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Skewness: {df['rating'].skew():.3f} (close to 0 = fairly symmetric)")

    with tab3:
        colA, colB = st.columns(2)
        with colA:
            fig = px.histogram(df, x="members", nbins=50, title="Members (raw) — right-skewed")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Skewness (raw): {df['members'].skew():.3f}")
        with colB:
            fig = px.histogram(
                df, x="members_log", nbins=50, title="log1p(members) — after transform"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Skewness (log1p): {df['members_log'].skew():.3f}")

    with tab4:
        genre_counts = df["genre"].str.split(", ").explode().value_counts().head(20).reset_index()
        genre_counts.columns = ["genre", "count"]
        fig = px.bar(
            genre_counts.sort_values("count"),
            x="count", y="genre", orientation="h", title="Top 20 most common genres",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Browse the cleaned data")
    st.dataframe(df.head(200), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center;'>

###  🤖 Anime Content-Based Recommender System using Machine Learning

Built with ❤️ using Streamlit | Developed by nmshah9

</div>
""", unsafe_allow_html=True)
