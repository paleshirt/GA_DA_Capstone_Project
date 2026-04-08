import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
import pickle

train_df = pd.read_csv("data_clean/train.csv")
test_df  = pd.read_csv("data_clean/test.csv")

X_train, y_train = train_df["clean_text"], train_df["label"]
X_test,  y_test  = test_df["clean_text"],  test_df["label"]

# ── Train final model ──────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,1), min_df=1)
X_train_t = tfidf.fit_transform(X_train)
X_test_t  = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_t, y_train)
y_pred = model.predict(X_test_t)

# ── Print metrics ──────────────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
print(f"Final Accuracy: {acc:.4f}")
print(f"Final F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=["Hidden Brain", "CNA Deep Dive"]))

# ── Save confusion matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Hidden Brain", "CNA Deep Dive"])
disp.plot(cmap="Blues")
plt.title("Final Model: TF-IDF + Logistic Regression")
plt.tight_layout()
plt.savefig("outputs/figures/final_confusion_matrix.png")
plt.close()
print("Confusion matrix saved.")

# ── Save final metrics table ───────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    "model": "TF-IDF + Logistic Regression",
    "accuracy": acc,
    "f1": f1,
    "vectorizer": "TF-IDF",
    "max_features": 5000,
    "ngram_range": "(1,1)",
    "min_df": 1
}])
metrics_df.to_csv("outputs/final_metrics.csv", index=False)
print("Metrics saved to outputs/final_metrics.csv")

# ── Save model and vectorizer ──────────────────────────────────────────────
with open("outputs/models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("outputs/models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model and vectorizer saved to outputs/models/")