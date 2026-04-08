import pandas as pd

# ── Paste your actual results here after running scripts 03, 04, 05 ────────
results = {
    "CountVectorizer + LR":  {"accuracy": 0.9241, "f1": 0.9167},
    "CountVectorizer + NB":  {"accuracy": 0.9873, "f1": 0.9873},
    "CountVectorizer + SVM": {"accuracy": 0.9241, "f1": 0.9167},
    "N-grams + LR":          {"accuracy": 0.9241, "f1": 0.9167},
    "N-grams + NB":          {"accuracy": 0.9873, "f1": 0.9873},
    "N-grams + SVM":         {"accuracy": 0.9241, "f1": 0.9167},
    "TF-IDF + LR":           {"accuracy": 0.9873, "f1": 0.9870},
    "TF-IDF + NB":           {"accuracy": 0.9873, "f1": 0.9870},
    "TF-IDF + SVM":          {"accuracy": 0.9747, "f1": 0.9737},
}

df = pd.DataFrame(results).T
df = df.sort_values("accuracy", ascending=False)
print(df.to_string())
df.to_csv("outputs/model_comparison.csv")
print("\nSaved to outputs/model_comparison.csv")