import pandas as pd
import numpy as np
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Load model and vectorizer ──────────────────────────────────────────────
with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("outputs/models/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = tfidf.get_feature_names_out()
coefficients  = model.coef_[0]

# ── Load episode data ──────────────────────────────────────────────────────
raw_df   = pd.read_csv("data_raw/episodes_raw.csv")
clean_df = pd.read_csv("data_clean/episodes_clean.csv")

# ── Reuse same cleaning logic as preprocessing ─────────────────────────────
stop_words = set(ENGLISH_STOP_WORDS)

def clean_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# ── Recommendation function ────────────────────────────────────────────────
def recommend(user_input, n=2, debug=False):
    print("\n" + "="*60)
    print("PODCAST RECOMMENDER v2")
    print("="*60)
    print(f"Input: \"{user_input}\"")

    # Clean input
    cleaned = clean_input(user_input)
    if debug:
        print(f"Cleaned input: {cleaned}")

    # Predict podcast
    user_vec   = tfidf.transform([cleaned])
    pred_label = model.predict(user_vec)[0]
    pred_proba = model.predict_proba(user_vec)[0]
    confidence = round(max(pred_proba) * 100, 1)
    podcast_name = "Hidden Brain" if pred_label == 0 else "CNA Deep Dive"

    print(f"\n📻 Best matching podcast : {podcast_name}")
    print(f"🎯 Confidence            : {confidence}%")

    # Get driving keywords from model coefficients
    feature_list = list(feature_names)
    coef_sign    = coefficients if pred_label == 1 else -coefficients
    input_words  = cleaned.split()
    word_scores  = {
        w: coef_sign[feature_list.index(w)]
        for w in input_words if w in feature_list
    }
    top_keywords = sorted(word_scores, key=word_scores.get, reverse=True)[:5]
    theme_str    = ", ".join(top_keywords) if top_keywords else "social issues"
    print(f"🔑 Key themes detected   : {theme_str}")

    # ── Episode scoring (3 signals) ────────────────────────────────────────
    # Filter episodes for predicted podcast
    matched_clean = clean_df[clean_df["label"] == pred_label].copy()
    matched_raw   = raw_df[raw_df["podcast"] == podcast_name].copy()

    # Align both to same size using title as key
    matched_raw = matched_raw[
        matched_raw["title"].isin(matched_clean["title"])
    ].copy()

    # Reset index for alignment
    matched_clean = matched_clean.reset_index(drop=True)
    matched_raw   = matched_raw.reset_index(drop=True)

    # Signal 1: Cosine similarity on description
    ep_vecs    = tfidf.transform(matched_clean["clean_text"].fillna(""))
    cos_scores = cosine_similarity(user_vec, ep_vecs)[0]

    # Signal 2: Keyword overlap on description
    def keyword_overlap(ep_text, keywords):
        if not keywords:
            return 0
        ep_words = set(str(ep_text).lower().split())
        return sum(1 for kw in keywords if kw in ep_words) / len(keywords)

    overlap_scores = matched_clean["clean_text"].apply(
        lambda x: keyword_overlap(x, input_words)
    ).values

    # Signal 3: Title keyword overlap
    title_scores = matched_raw["title"].apply(
        lambda x: keyword_overlap(str(x).lower(), input_words)
    ).values

    # Combined score (60% cosine + 20% description overlap + 20% title overlap)
    combined_scores = (
        0.6 * cos_scores +
        0.2 * overlap_scores +
        0.2 * title_scores
    )
    matched_clean = matched_clean.copy()
    matched_clean["score"] = combined_scores

    # Get top n episodes
    top_idx      = matched_clean.nlargest(n, "score").index.tolist()
    top_episodes = matched_raw.loc[top_idx]

    print(f"\n🎧 Top {n} Recommended Episodes from {podcast_name}:")
    print("-" * 60)

    for rank, (idx, row) in enumerate(top_episodes.iterrows(), 1):
        score = matched_clean.loc[idx, "score"]
        print(f"\n#{rank}: {row['title']}")
        print(f"    Released    : {row['release_date']}")
        print(f"    Match score : {round(score * 100, 1)}%")
        print(f"    Spotify     : {row['spotify_url']}")
        print(f"    Why         : This episode matches your interest in "
              f"{theme_str}.")

    print("\n" + "="*60)

# ── Run tests ──────────────────────────────────────────────────────────────
recommend("I'm interested in why people make irrational decisions and how emotions affect behavior")
recommend("I want to learn about Singapore housing policy and government initiatives")
recommend("mental health stigma and how society treats people with anxiety and depression")
recommend("climate change environment sustainability and green policy")
recommend("workplace stress burnout and managing career pressure")