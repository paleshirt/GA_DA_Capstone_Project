import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

# ── Load raw data ──────────────────────────────────────────────────────────
df = pd.read_csv("data_raw/episodes_raw.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())

# ── Stop words ─────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        if nltk.download("stopwords", quiet=True):
            stop_words = set(stopwords.words("english"))
        else:
            raise LookupError
except Exception:
    print("Warning: NLTK stopwords unavailable; using sklearn fallback.")
    stop_words = set(ENGLISH_STOP_WORDS)

# ── Text cleaning pipeline ─────────────────────────────────────────────────
def clean_text(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove Hidden Brain boilerplate
    text = re.sub(r"hosted by simplecast.*", "", text)
    text = re.sub(r"see pcm\.adswizz\.com.*", "", text)
    text = re.sub(r"pcmadswizzcom.*", "", text)
    text = re.sub(r"adswizz.*", "", text)
    # Remove CNA Deep Dive boilerplate
    text = re.sub(r"see omnystudio.*", "", text)
    text = re.sub(r"omnystudio.*", "", text)
    # Remove host names
    text = re.sub(r"\b(steven|chia|crispina|robert|tiffany|ang|otelli|edwards)\b", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stop words
    text = " ".join([w for w in text.split() if w not in stop_words])
    # Remove podcast name references
    text = re.sub(r"\b(hidden|brain|podcast|episode|episodes|week|series|favorite)\b", "", text)
    # Remove CNA identity words
    text = re.sub(r"\b(singapore|deep dive|cna)\b", "", text)
    # Remove generic guest titles and common filler words
    text = re.sub(r"\b(dr|mr|ms|professor|prof|speaks|speak|says|like|check|consider)\b", "", text)
    # Remove very common Singapore surnames used as identifiers
    text = re.sub(r"\b(tan|lee|lim|ng|wong)\b", "", text)
    # Remove Hidden Brain's subscription link that appears in descriptions
    text = re.sub(r"support\S+", "", text)
    # Remove Hidden Brain subscription/newsletter boilerplate
    text = re.sub(r"\b(subscribe|newsletter|listening|thanks|apple|podcasts|unsung|hero|revisit)\b", "", text)
    text = re.sub(r"newshiddenbrainorg\S*", "", text)
    # Remove specific CNA Deep Dive guest/person names (Identity Leakage)
    text = re.sub(r"\b(walter|theseira|lin|suling|kuan|yew)\b", "", text)
    # Remove CNA Deep Dive host names
    text = re.sub(r"\b(steven|chia|crispina|robert|tiffany|ang|otelli|edwards)\b", "", text)
    # Remove Hidden Brain host name
    text = re.sub(r"\b(shankar|vedantam)\b", "", text)
    return text

df["clean_text"] = df["description"].apply(clean_text)
print(df[["description", "clean_text"]].head())

# ── Remove duplicates ──────────────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset="clean_text")
after = len(df)
print(f"Removed {before - after} duplicates. Remaining: {after}")

# ── Create label column ────────────────────────────────────────────────────
df["label"] = df["podcast"].map({
    "Hidden Brain": 0,
    "CNA Deep Dive": 1
})
print(df["label"].value_counts())

# ── Train/test split ───────────────────────────────────────────────────────
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Train label balance:\n{y_train.value_counts()}")
print(f"Test label balance:\n{y_test.value_counts()}")

# ── Save files ─────────────────────────────────────────────────────────────
df[["podcast", "label", "clean_text", "title", "release_date"]].to_csv(
    "data_clean/episodes_clean.csv", index=False
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

train_df.to_csv("data_clean/train.csv", index=False)
test_df.to_csv("data_clean/test.csv",   index=False)

print("Files saved to data_clean/")