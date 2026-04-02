import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ── Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv("data_clean/train.csv")
test_df  = pd.read_csv("data_clean/test.csv")

X_train, y_train = train_df["clean_text"], train_df["label"]
X_test,  y_test  = test_df["clean_text"],  test_df["label"]

# ── Vectorise ──────────────────────────────────────────────────────────────
cv = CountVectorizer(max_features=5000)
X_train_cv = cv.fit_transform(X_train)
X_test_cv  = cv.transform(X_test)

# ── EDA: top words per podcast ─────────────────────────────────────────────
full_df = pd.read_csv("data_clean/episodes_clean.csv")
for label, name in [(0, "Hidden Brain"), (1, "CNA Deep Dive")]:
    subset = full_df[full_df["label"] == label]["clean_text"]
    vec = CountVectorizer(max_features=20)
    vec.fit_transform(subset)
    top_words = pd.Series(
        vec.transform(subset).toarray().sum(axis=0),
        index=vec.get_feature_names_out()
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    top_words.plot(kind="bar")
    plt.title(f"Top 20 Words — {name} (CountVectorizer)")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/cv_top_words_{name.replace(' ', '_')}.png")
    plt.close()
    print(f"\nTop 20 words for {name}:")
    print(top_words)

# ── Train and evaluate 3 models ────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         MultinomialNB(),
    "Linear SVM":          LinearSVC(max_iter=1000)
}

print("\n── CountVectorizer Results ──")
cv_results = {}
for model_name, model in models.items():
    model.fit(X_train_cv, y_train)
    y_pred = model.predict(X_test_cv)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    cv_results[model_name] = {"accuracy": acc, "f1": f1}
    print(f"{model_name}: Accuracy={acc:.4f}, F1={f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Hidden Brain", "CNA Deep Dive"])
    disp.plot()
    plt.title(f"Confusion Matrix — {model_name} (CountVectorizer)")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/cv_cm_{model_name.replace(' ', '_')}.png")
    plt.close()