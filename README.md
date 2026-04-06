# Social-Issues Podcast Recommender with Transparency (Spotify Platform)

## Goal
Build a performance-first, transparent podcast recommender that helps users discover social commentary / social issues content through Spotify.

The project starts with a simple MVP (Minimum Viable Product):
- compare **2 podcasts only**:
  - **Hidden Brain**
  - **CNA Deep Dive**
- train an NLP classifier to predict which podcast an episode belongs to
- aim for **90% accuracy or higher**
- recommend **2 episodes** from the predicted better-matching podcast
- explain each recommendation using **theme tags, keywords, and phrases**

After the MVP is successful, the project can be scaled up with more podcasts, multilingual support, and stronger Spotify integration.

---

## MVP Problem Statement
Can I build an NLP model that classifies whether an episode belongs to **Hidden Brain** or **CNA Deep Dive** with **90%+ accuracy**, and then recommend **2 episodes** from the better-matching podcast with clear explanations?

---

## Chosen Podcasts
### 1) Hidden Brain
An evidence-based podcast hosted by science journalist Shankar Vedantam that explores the unconscious patterns driving human behaviour, social biases, and psychological research. Rated the #1 science podcast in the United States.

### 2) CNA Deep Dive
A newsroom-style podcast that unpacks current affairs and social issues, especially from a Singapore and public-interest perspective.

---

## Final Model Performance
| Metric | Score |
|---|---|
| Accuracy | **98.73%** |
| F1 Score | **98.70%** |
| Misclassifications | **1 out of 79 test episodes** |

**Chosen model:** TF-IDF + Logistic Regression

---

## Dataset(s)

### Primary training dataset
**Spotify Web API**
Episode metadata pulled directly from Spotify for both podcasts.

Fields used:
- podcast name (label)
- episode title
- episode description
- release date
- duration
- Spotify episode URL

**Dataset size:** 400 episodes (200 per podcast), balanced to avoid class imbalance.
After cleaning: 395 episodes (5 duplicates removed).

### Spotify Web API (platform layer)
Spotify is also used as the **platform** for:
- checking podcast availability
- mapping recommendations back to Spotify links
- optional future personalization/demo features

---

## Project Approach

### Phase 1: Performance-first MVP
The first objective is to achieve strong model performance before adding more features.

#### Step 1: Data collection
- collect episode metadata and text for **Hidden Brain** and **CNA Deep Dive** via Spotify API
- balance to 200 episodes each (400 total)
- combine into one clean dataset

#### Step 2: Text preprocessing (4 rounds of cleaning)
- Round 1: Standard NLP — lowercase, remove URLs, special characters, stop words
- Round 2: Targeted leakage removal — platform boilerplate (Simplecast, AdsWizz, OmnyStudio), host names, podcast identity words
- Round 3: Residual leakage — host name completion (shankar, vedantam), subscription/newsletter text
- Round 4: Final cleanup — removed `episodes` which was still appearing as an identity word

Full cleaning decisions documented in `docs/cleaning_decisions.md`.

#### Step 3: NLP experimentation
Test **3 vectorisation methods**:
1. **CountVectorizer**
2. **N-grams** (unigrams and bigrams)
3. **TF-IDF**

#### Step 4: EDA for each method
For each vectorisation method, examine:
- top words and phrases per podcast
- distinctive terms
- differences between the two podcasts

#### Step 5: Modelling
Train and compare baseline models using the same fixed train/test split (80/20, stratified).

Models tested:
- Logistic Regression
- Naive Bayes
- Linear SVM

#### Step 6: Model selection
**TF-IDF + Logistic Regression** chosen as the final model because:
- Tied highest accuracy (98.73%) with Naive Bayes
- Provides probability scores per prediction (e.g. "94% Hidden Brain")
- Provides feature coefficients — shows exactly which words drove each prediction
- More suitable for explainability requirements

#### Step 7: Explainability
For the final model:
- top 20 words per podcast extracted from model coefficients
- theme tags generated from user input
- "Recommended because..." explanations attached to each episode

**Top predictive words:**
- Hidden Brain: `psychologist`, `lives`, `explore`, `conversation`, `relationships`, `purpose`
- CNA Deep Dive: `director`, `university`, `minister`, `government`, `policy`, `law`

#### Step 8: Recommendation output
After predicting which podcast best matches a user's interests, recommend **2 episodes** using a 3-signal scoring system:
- 60% cosine similarity (TF-IDF vectors)
- 20% keyword overlap on episode description
- 20% keyword overlap on episode title

---

## Known Limitations
- **Vocabulary mismatch:** recommender relies on keyword overlap — works less well when user vocabulary differs from podcast vocabulary (e.g. "burnout" vs "worker wellbeing")
- **Dataset size:** 196 CNA Deep Dive episodes — niche topics may have limited coverage
- **Short descriptions:** episode descriptions are brief; full transcripts would give richer signal

Potential future fixes: synonym expansion, larger dataset, full transcript data, semantic similarity (SBERT).

---

## Dashboard
A Tableau dashboard has been built covering:
- Model accuracy comparison across all 9 method/model combinations
- Confusion matrix for the final model (TF-IDF + Logistic Regression)
- Top 20 predictive words per podcast (Hidden Brain and CNA Deep Dive separately)
- Episode distribution by year

Dashboard file: `Tableau - Capstone.twb` (open with Tableau Desktop or Tableau Public)

---

## How to Run the Streamlit App

### Local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Public Demo
[Link to be added after Streamlit Cloud deployment]

---

## Success Metrics

### Primary technical metric
- **Accuracy ≥ 90%** ✅ achieved 98.73%
- also reported:
  - **F1 score** — 98.70%
  - **confusion matrix** — 1 misclassification out of 79

### Secondary recommendation goals
- recommend **2 episodes** from the predicted better-matching podcast ✅
- each recommendation includes:
  - theme tags ✅
  - keywords/phrases ✅
  - a short explanation ✅
  - Spotify link ✅

---

## Repo Structure
data_raw/               → raw extracted data (episodes_raw.csv)
data_clean/             → cleaned data (episodes_clean.csv, train.csv, test.csv)
notebooks/              → 12 Python scripts covering full pipeline
outputs/figures/        → confusion matrices, feature importance plots
outputs/models/         → tfidf_vectorizer.pkl, logistic_regression_model.pkl
outputs/                → model comparison, final metrics, Tableau export CSVs
docs/                   → labeling_guidelines.md, cleaning_decisions.md
app.py                  → Streamlit web application
requirements.txt        → Python dependencies

---

## How to Reproduce

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Spotify credentials to a `.env` file: