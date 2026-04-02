import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# ── Load data ──────────────────────────────────────────────────────────────
train_df  = pd.read_csv("data_clean/train.csv")
test_df   = pd.read_csv("data_clean/test.csv")
clean_df  = pd.read_csv("data_clean/episodes_clean.csv")
raw_df    = pd.read_csv("data_raw/episodes_raw.csv")

with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("outputs/models/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

X_train, y_train = train_df["clean_text"], train_df["label"]
X_test,  y_test  = test_df["clean_text"],  test_df["label"]

X_test_t = tfidf.transform(X_test)
y_pred   = model.predict(X_test_t)
y_proba  = model.predict_proba(X_test_t)

# ── Export 1: Model comparison table ──────────────────────────────────────
model_comparison = pd.DataFrame([
    {"Method": "CountVectorizer", "Model": "Logistic Regression", "Accuracy": 0.9241, "F1": 0.9167},
    {"Method": "CountVectorizer", "Model": "Naive Bayes",         "Accuracy": 0.9873, "F1": 0.9873},
    {"Method": "CountVectorizer", "Model": "Linear SVM",          "Accuracy": 0.9241, "F1": 0.9167},
    {"Method": "N-grams",         "Model": "Logistic Regression", "Accuracy": 0.9241, "F1": 0.9167},
    {"Method": "N-grams",         "Model": "Naive Bayes",         "Accuracy": 0.9747, "F1": 0.9750},
    {"Method": "N-grams",         "Model": "Linear SVM",          "Accuracy": 0.9241, "F1": 0.9167},
    {"Method": "TF-IDF",          "Model": "Logistic Regression", "Accuracy": 0.9873, "F1": 0.9870},
    {"Method": "TF-IDF",          "Model": "Naive Bayes",         "Accuracy": 0.9873, "F1": 0.9870},
    {"Method": "TF-IDF",          "Model": "Linear SVM",          "Accuracy": 0.9747, "F1": 0.9737},
])
model_comparison.to_csv("outputs/tableau_model_comparison.csv", index=False)
print("✅ Saved: tableau_model_comparison.csv")

# ── Export 2: Confusion matrix data ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame([
    {"Actual": "Hidden Brain",  "Predicted": "Hidden Brain",  "Count": cm[0][0]},
    {"Actual": "Hidden Brain",  "Predicted": "CNA Deep Dive", "Count": cm[0][1]},
    {"Actual": "CNA Deep Dive", "Predicted": "Hidden Brain",  "Count": cm[1][0]},
    {"Actual": "CNA Deep Dive", "Predicted": "CNA Deep Dive", "Count": cm[1][1]},
])
cm_df.to_csv("outputs/tableau_confusion_matrix.csv", index=False)
print("✅ Saved: tableau_confusion_matrix.csv")

# ── Export 3: Prediction confidence per episode ────────────────────────────
test_episodes = test_df.copy()
test_episodes["predicted_label"]   = y_pred
test_episodes["confidence_hb"]     = y_proba[:, 0]
test_episodes["confidence_cna"]    = y_proba[:, 1]
test_episodes["correct"]           = (y_pred == y_test).astype(int)
test_episodes["predicted_podcast"] = test_episodes["predicted_label"].map(
    {0: "Hidden Brain", 1: "CNA Deep Dive"}
)
test_episodes["actual_podcast"] = y_test.map(
    {0: "Hidden Brain", 1: "CNA Deep Dive"}
)
test_episodes.to_csv("outputs/tableau_predictions.csv", index=False)
print("✅ Saved: tableau_predictions.csv")

# ── Export 4: Top features per class ──────────────────────────────────────
feature_names = tfidf.get_feature_names_out()
coefficients  = model.coef_[0]

top_hb = pd.DataFrame({
    "word":      feature_names[np.argsort(coefficients)[:20]],
    "score":     np.sort(coefficients)[:20],
    "podcast":   "Hidden Brain"
})
top_cna = pd.DataFrame({
    "word":      feature_names[np.argsort(coefficients)[-20:][::-1]],
    "score":     np.sort(coefficients)[-20:][::-1],
    "podcast":   "CNA Deep Dive"
})
top_features = pd.concat([top_hb, top_cna])
top_features["abs_score"] = top_features["score"].abs()
top_features.to_csv("outputs/tableau_top_features.csv", index=False)
print("✅ Saved: tableau_top_features.csv")

# ── Export 5: Episode dataset overview ────────────────────────────────────
raw_df["description_length"] = raw_df["description"].str.len()
raw_df["release_year"]       = pd.to_datetime(
    raw_df["release_date"], errors="coerce"
).dt.year
raw_df.to_csv("outputs/tableau_episodes.csv", index=False)
print("✅ Saved: tableau_episodes.csv")

print("\nAll Tableau exports complete — check outputs/ folder.")