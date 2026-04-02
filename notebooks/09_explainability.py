import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ── Load saved model and vectorizer ───────────────────────────────────────
with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("outputs/models/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Extract top features per class ────────────────────────────────────────
feature_names = tfidf.get_feature_names_out()
coefficients  = model.coef_[0]

coef_df = pd.DataFrame({
    "word":        feature_names,
    "coefficient": coefficients
})

top_hidden_brain = coef_df.nsmallest(20, "coefficient")
top_cna          = coef_df.nlargest(20, "coefficient")

print("── Top 20 words → Hidden Brain ──")
print(top_hidden_brain[["word", "coefficient"]].to_string(index=False))

print("\n── Top 20 words → CNA Deep Dive ──")
print(top_cna[["word", "coefficient"]].to_string(index=False))

# ── Plot top features ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(top_hidden_brain["word"], top_hidden_brain["coefficient"], color="steelblue")
axes[0].set_title("Top 20 Words → Hidden Brain")
axes[0].invert_yaxis()

axes[1].barh(top_cna["word"], top_cna["coefficient"], color="coral")
axes[1].set_title("Top 20 Words → CNA Deep Dive")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/figures/top_features_per_class.png")
plt.close()
print("\nFeature importance plot saved.")

# ── Save top features ──────────────────────────────────────────────────────
top_hidden_brain.to_csv("outputs/top_features_hidden_brain.csv", index=False)
top_cna.to_csv("outputs/top_features_cna.csv", index=False)

# ── Recommendation template ────────────────────────────────────────────────
def recommend_episodes(user_text, n=2):
    df = pd.read_csv("data_clean/episodes_clean.csv")
    raw_df = pd.read_csv("data_raw/episodes_raw.csv")

    # Predict podcast
    user_vec   = tfidf.transform([user_text])
    pred_label = model.predict(user_vec)[0]
    pred_proba = model.predict_proba(user_vec)[0]
    confidence = round(max(pred_proba) * 100, 1)

    podcast_name = "Hidden Brain" if pred_label == 0 else "CNA Deep Dive"
    print(f"\n── Prediction ──")
    print(f"Best matching podcast: {podcast_name} ({confidence}% confidence)")

    # Get top features that drove prediction
    user_words   = user_text.lower().split()
    feature_list = list(feature_names)
    coef_sign    = coefficients if pred_label == 1 else -coefficients
    word_scores  = {w: coef_sign[feature_list.index(w)] 
                    for w in user_words if w in feature_list}
    top_keywords = sorted(word_scores, key=word_scores.get, reverse=True)[:5]

    print(f"Key themes detected: {', '.join(top_keywords) if top_keywords else 'general content'}")

    # Find top matching episodes
    matched = df[df["label"] == pred_label].copy()
    raw_matched = raw_df[raw_df["podcast"] == podcast_name].copy()

    recommendations = raw_matched.head(n)
    print(f"\n── Top {n} Recommended Episodes ──")
    for _, row in recommendations.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Spotify: {row['spotify_url']}")
        print(f"Recommended because: This episode is from {podcast_name}, "
              f"which matches your interest in: {', '.join(top_keywords) if top_keywords else 'social issues'}.")

# ── Test with sample inputs ────────────────────────────────────────────────
print("\n═══ Test 1: Psychology-style input ═══")
recommend_episodes("human behavior psychology bias decision making unconscious")

print("\n═══ Test 2: Policy-style input ═══")
recommend_episodes("government policy education public health social inequality")