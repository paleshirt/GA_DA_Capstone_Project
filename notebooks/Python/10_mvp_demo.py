import pandas as pd
import numpy as np
import pickle

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

# ── Core recommendation function ──────────────────────────────────────────
def run_mvp(user_input):
    print("\n" + "="*60)
    print("PODCAST RECOMMENDER — MVP DEMO")
    print("="*60)
    print(f"Your input: \"{user_input}\"")

    # Clean input the same way as training data
    import re
    text = user_input.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Predict
    user_vec   = tfidf.transform([text])
    pred_label = model.predict(user_vec)[0]
    pred_proba = model.predict_proba(user_vec)[0]
    confidence = round(max(pred_proba) * 100, 1)
    podcast_name = "Hidden Brain" if pred_label == 0 else "CNA Deep Dive"

    print(f"\n📻 Best matching podcast : {podcast_name}")
    print(f"🎯 Confidence            : {confidence}%")

    # Get driving keywords
    feature_list = list(feature_names)
    coef_sign    = coefficients if pred_label == 1 else -coefficients
    input_words  = text.split()
    word_scores  = {w: coef_sign[feature_list.index(w)]
                    for w in input_words if w in feature_list}
    top_keywords = sorted(word_scores, key=word_scores.get, reverse=True)[:5]
    theme_str    = ", ".join(top_keywords) if top_keywords else "social issues"
    print(f"🔑 Key themes detected   : {theme_str}")

    # Score all episodes from matched podcast by similarity to input
    matched_raw   = raw_df[raw_df["podcast"] == podcast_name].copy()
    matched_clean = clean_df[clean_df["label"] == pred_label].copy()

    # Vectorize all episode descriptions
    ep_vecs    = tfidf.transform(matched_clean["clean_text"].fillna(""))
    user_vec_d = tfidf.transform([text])

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(user_vec_d, ep_vecs)[0]
    matched_clean = matched_clean.copy()
    matched_clean["score"] = scores

    # Get top 2 by similarity
    top2_idx = matched_clean.nlargest(2, "score").index
    top2_raw = matched_raw[matched_raw.index.isin(
        matched_clean.loc[top2_idx, "clean_text"].index
    )]

    # Fallback: just use top 2 from raw if index matching fails
    top2_episodes = matched_raw.head(2) if len(top2_raw) < 2 else top2_raw

    print(f"\n🎧 Top 2 Recommended Episodes from {podcast_name}:")
    print("-" * 60)
    for i, (_, row) in enumerate(top2_episodes.iterrows(), 1):
        print(f"\n#{i}: {row['title']}")
        print(f"    Released : {row['release_date']}")
        print(f"    Spotify  : {row['spotify_url']}")
        print(f"    Why      : Matches your interest in {theme_str}.")
    print("\n" + "="*60)

# ── Run demo with 3 test inputs ────────────────────────────────────────────
run_mvp("I'm interested in why people make irrational decisions and how emotions affect behavior")
run_mvp("I want to learn about Singapore housing policy and government initiatives")
run_mvp("mental health stigma and how society treats people with anxiety and depression")