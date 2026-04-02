import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

train_df = pd.read_csv("data_clean/train.csv")
test_df  = pd.read_csv("data_clean/test.csv")

X_train, y_train = train_df["clean_text"], train_df["label"]
X_test,  y_test  = test_df["clean_text"],  test_df["label"]

# ── Vectorise with bigrams ─────────────────────────────────────────────────
ng = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_ng = ng.fit_transform(X_train)
X_test_ng  = ng.transform(X_test)

# ── EDA: top bigrams per podcast ───────────────────────────────────────────
full_df = pd.read_csv("data_clean/episodes_clean.csv")
for label, name in [(0, "Hidden Brain"), (1, "CNA Deep Dive")]:
    subset = full_df[full_df["label"] == label]["clean_text"]
    vec = CountVectorizer(max_features=20, ngram_range=(2, 2))
    vec.fit_transform(subset)
    top_phrases = pd.Series(
        vec.transform(subset).toarray().sum(axis=0),
        index=vec.get_feature_names_out()
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    top_phrases.plot(kind="bar")
    plt.title(f"Top 20 Bigrams — {name} (N-grams)")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/ng_top_bigrams_{name.replace(' ', '_')}.png")
    plt.close()
    print(f"\nTop 20 bigrams for {name}:")
    print(top_phrases)

# ── Train and evaluate 3 models ────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         MultinomialNB(),
    "Linear SVM":          LinearSVC(max_iter=1000)
}

print("\n── N-grams Results ──")
ng_results = {}
for model_name, model in models.items():
    model.fit(X_train_ng, y_train)
    y_pred = model.predict(X_test_ng)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    ng_results[model_name] = {"accuracy": acc, "f1": f1}
    print(f"{model_name}: Accuracy={acc:.4f}, F1={f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Hidden Brain", "CNA Deep Dive"])
    disp.plot()
    plt.title(f"Confusion Matrix — {model_name} (N-grams)")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/ng_cm_{model_name.replace(' ', '_')}.png")
    plt.close()