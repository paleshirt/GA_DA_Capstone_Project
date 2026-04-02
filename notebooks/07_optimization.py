import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train_df = pd.read_csv("data_clean/train.csv")
test_df  = pd.read_csv("data_clean/test.csv")

X_train, y_train = train_df["clean_text"], train_df["label"]
X_test,  y_test  = test_df["clean_text"],  test_df["label"]

# ── Experiment with TF-IDF parameters ─────────────────────────────────────
experiments = [
    {"max_features": 5000, "ngram_range": (1,1), "min_df": 1},
    {"max_features": 5000, "ngram_range": (1,2), "min_df": 2},
    {"max_features": 8000, "ngram_range": (1,2), "min_df": 2},
    {"max_features": 10000,"ngram_range": (1,3), "min_df": 2},
]

print("── TF-IDF + Logistic Regression Optimization ──\n")
best_acc = 0
best_config = None

for config in experiments:
    tfidf = TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=config["ngram_range"],
        min_df=config["min_df"]
    )
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t  = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"Config {config} → Accuracy={acc:.4f}, F1={f1:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_config = config
        best_tfidf = tfidf
        best_model = model
        best_pred = y_pred

print(f"\n✅ Best config: {best_config}")
print(f"✅ Best Accuracy: {best_acc:.4f}")

# ── Save confusion matrix for best model ───────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Hidden Brain", "CNA Deep Dive"])
disp.plot()
plt.title(f"Confusion Matrix — Best TF-IDF + LR\n{best_config}")
plt.tight_layout()
plt.savefig("outputs/figures/best_model_confusion_matrix.png")
plt.close()
print("Confusion matrix saved.")

# ── Save results ───────────────────────────────────────────────────────────
results = pd.DataFrame([
    {**config, "accuracy": accuracy_score(y_test, 
        LogisticRegression(max_iter=1000).fit(
            TfidfVectorizer(**{k:v for k,v in config.items()}).fit_transform(X_train),
            y_train
        ).predict(
            TfidfVectorizer(**{k:v for k,v in config.items()}).fit(X_train).transform(X_test)
        )), 
     "f1": f1_score(y_test,
        LogisticRegression(max_iter=1000).fit(
            TfidfVectorizer(**{k:v for k,v in config.items()}).fit_transform(X_train),
            y_train
        ).predict(
            TfidfVectorizer(**{k:v for k,v in config.items()}).fit(X_train).transform(X_test)
        ))}
    for config in experiments
])
results.to_csv("outputs/optimization_results.csv", index=False)
print("Results saved to outputs/optimization_results.csv")