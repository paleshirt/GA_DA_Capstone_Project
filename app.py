import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Podcast Recommender",
    page_icon="🎙️",
    layout="wide"
)

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("outputs/models/logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw_df   = pd.read_csv("data_raw/episodes_raw.csv")
    clean_df = pd.read_csv("data_clean/episodes_clean.csv")
    return raw_df, clean_df

tfidf, model     = load_model()
raw_df, clean_df = load_data()
feature_names    = tfidf.get_feature_names_out()
coefficients     = model.coef_[0]
stop_words       = set(ENGLISH_STOP_WORDS)

# ── Text cleaning ──────────────────────────────────────────────────────────
def clean_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# ── Recommendation logic ───────────────────────────────────────────────────
def recommend(user_input, n=2):
    cleaned      = clean_input(user_input)
    user_vec     = tfidf.transform([cleaned])
    pred_label   = model.predict(user_vec)[0]
    pred_proba   = model.predict_proba(user_vec)[0]
    confidence   = round(max(pred_proba) * 100, 1)
    podcast_name = "Hidden Brain" if pred_label == 0 else "CNA Deep Dive"

    feature_list = list(feature_names)
    coef_sign    = coefficients if pred_label == 1 else -coefficients
    input_words  = cleaned.split()
    word_scores  = {
        w: coef_sign[feature_list.index(w)]
        for w in input_words if w in feature_list
    }
    top_keywords = sorted(word_scores, key=word_scores.get, reverse=True)[:5]
    theme_str    = ", ".join(top_keywords) if top_keywords else "social issues"

    matched_clean = clean_df[clean_df["label"] == pred_label].copy()
    matched_raw   = raw_df[raw_df["podcast"] == podcast_name].copy()
    matched_raw   = matched_raw[
        matched_raw["title"].isin(matched_clean["title"])
    ].copy()
    matched_clean = matched_clean.reset_index(drop=True)
    matched_raw   = matched_raw.reset_index(drop=True)

    ep_vecs    = tfidf.transform(matched_clean["clean_text"].fillna(""))
    cos_scores = cosine_similarity(user_vec, ep_vecs)[0]

    def keyword_overlap(ep_text, keywords):
        if not keywords:
            return 0
        ep_words = set(str(ep_text).lower().split())
        return sum(1 for kw in keywords if kw in ep_words) / len(keywords)

    overlap_scores = matched_clean["clean_text"].apply(
        lambda x: keyword_overlap(x, input_words)).values
    title_scores   = matched_raw["title"].apply(
        lambda x: keyword_overlap(str(x).lower(), input_words)).values

    combined_scores = 0.6 * cos_scores + 0.2 * overlap_scores + 0.2 * title_scores
    matched_clean   = matched_clean.copy()
    matched_clean["score"] = combined_scores

    top_idx      = matched_clean.nlargest(n, "score").index.tolist()
    top_episodes = matched_raw.loc[top_idx]

    return podcast_name, confidence, theme_str, top_episodes, matched_clean, top_idx

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🎙️ Social Issues Podcast Recommender")
st.markdown("*Transparent recommendations powered by NLP — Hidden Brain vs CNA Deep Dive*")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔍 Find My Podcast", "📊 Model Insights"])

# ── Tab 1: Recommender ─────────────────────────────────────────────────────
with tab1:
    st.subheader("What topics are you interested in?")
    user_input = st.text_area(
        "Describe your interests:",
        placeholder="e.g. mental health stigma and how society treats people with anxiety...",
        height=100
    )

    if st.button("🎯 Find My Podcast", type="primary"):
        if user_input.strip():
            with st.spinner("Analysing your interests..."):
                podcast_name, confidence, theme_str, top_episodes, matched_clean, top_idx = recommend(user_input)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎙️ Best Matching Podcast", podcast_name)
            with col2:
                st.metric("🎯 Confidence", f"{confidence}%")

            st.markdown(f"**🔑 Key themes detected:** `{theme_str}`")
            st.markdown("---")
            st.subheader(f"🎧 Top 2 Recommended Episodes from {podcast_name}")

            for rank, (idx, row) in enumerate(top_episodes.iterrows(), 1):
                score = matched_clean.loc[top_idx[rank-1], "score"]
                with st.expander(f"#{rank}: {row['title']} — Match: {round(score*100,1)}%"):
                    st.markdown(f"**Released:** {row['release_date']}")
                    st.markdown(f"**Why recommended:** This episode matches your interest in *{theme_str}*.")
                    st.markdown(f"**Listen on Spotify:** [Open Episode]({row['spotify_url']})")
        else:
            st.warning("Please enter something about your interests!")

# ── Tab 2: Model Insights ──────────────────────────────────────────────────
with tab2:
    st.subheader("Model Performance Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Accuracy", "98.73%")
    col2.metric("F1 Score", "98.70%")
    col3.metric("Misclassifications", "1 / 79")

    st.markdown("---")
    st.subheader("All 9 Model Combinations")
    model_df = pd.read_csv("outputs/tableau_model_comparison.csv")
    model_df["Accuracy %"] = (model_df["Accuracy"] * 100).round(2)
    model_df["F1 %"]       = (model_df["F1"] * 100).round(2)
    st.dataframe(
        model_df[["Method", "Model", "Accuracy %", "F1 %"]]
        .sort_values("Accuracy %", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Top Predictive Words per Podcast")
    features_df = pd.read_csv("outputs/tableau_top_features.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Hidden Brain**")
        hb = features_df[features_df["podcast"] == "Hidden Brain"].nlargest(10, "abs_score")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(hb["word"], hb["abs_score"], color="steelblue")
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        st.pyplot(fig)

    with col2:
        st.markdown("**CNA Deep Dive**")
        cna = features_df[features_df["podcast"] == "CNA Deep Dive"].nlargest(10, "abs_score")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(cna["word"], cna["abs_score"], color="coral")
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm_df = pd.read_csv("outputs/tableau_confusion_matrix.csv")
    st.dataframe(cm_df, use_container_width=True)