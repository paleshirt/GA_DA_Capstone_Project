# load raw data
import pandas as pd

df = pd.read_csv("data_raw/episodes_raw.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())

# build text cleaning pipeline

import re
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords

def load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        # Try one quiet download attempt; this can fail in restricted SSL/network environments.
        if nltk.download("stopwords", quiet=True):
            try:
                return set(stopwords.words("english"))
            except LookupError:
                pass

    print("Warning: NLTK stopwords unavailable; using sklearn ENGLISH_STOP_WORDS fallback.")
    return set(ENGLISH_STOP_WORDS)

stop_words = load_stop_words()

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stop words
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

df["clean_text"] = df["description"].apply(clean_text)
print(df[["description", "clean_text"]].head())

# remove duplicates

before = len(df)
df = df.drop_duplicates(subset="clean_text")
after = len(df)
print(f"Removed {before - after} duplicates. Remaining: {after}")

# create label column

df["label"] = df["podcast"].map({
    "Hidden Brain": 0,
    "CNA Deep Dive": 1
})
print(df["label"].value_counts())

# create fixed train/test split
from sklearn.model_selection import train_test_split

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # ensures equal class balance in both splits
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Train label balance:\n{y_train.value_counts()}")
print(f"Test label balance:\n{y_test.value_counts()}")

# save the clean dataset and split

# Save full clean dataset
df[["podcast", "label", "clean_text", "title", "release_date"]].to_csv(
    "data_clean/episodes_clean.csv", index=False
)

# Save train and test splits
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("data_clean/train.csv", index=False)
test_df.to_csv("data_clean/test.csv", index=False)

print("Files saved to data_clean/")