# Social-Issues Podcast Recommender with Transparency (Spotify Platform)

## Goal
Build a performance-first, transparent podcast recommender that helps users discover social commentary / social issues content through Spotify.

The project starts with a simple MVP:
- compare **2 podcasts only**:
  - **Harvard Thinking**
  - **CNA Deep Dive**
- train an NLP classifier to predict which podcast an episode belongs to
- aim for **90% accuracy or higher**
- recommend **2 episodes** from the predicted better-matching podcast
- explain each recommendation using **theme tags, keywords, and phrases**

After the MVP is successful, the project can be scaled up with more podcasts, multilingual support, and stronger Spotify integration.

---

## MVP Problem Statement
Can I build an NLP model that classifies whether an episode belongs to **Harvard Thinking** or **CNA Deep Dive** with **90%+ accuracy**, and then recommend **2 episodes** from the better-matching podcast with clear explanations?

---

## Chosen Podcasts
### 1) Harvard Thinking
An academic and expert-led podcast that discusses ideas, research, and social issues through a university / scholarly lens.

### 2) CNA Deep Dive
A newsroom-style podcast that unpacks current affairs and social issues, especially from a Singapore and public-interest perspective.

---

## Dataset(s)

### Primary training dataset
**Podcast Index Database Dump (SQLite)**  
Used as the main text source for model training.

Planned fields:
- show title
- episode title
- episode description
- published date
- language
- categories (if available)

This dataset is used because Spotify content should not be used as the primary training source for ML/NLP.

### Spotify Web API (platform layer)
Spotify is used as the **platform** for:
- checking podcast availability
- mapping recommendations back to Spotify links
- optional future personalization/demo features

Spotify is **not** the main training dataset.

---

## Project Approach

### Phase 1: Performance-first MVP
The first objective is to achieve strong model performance before adding more features.

#### Step 1: Data collection
- collect episode metadata and text for **Harvard Thinking** and **CNA Deep Dive**
- combine them into one clean dataset

#### Step 2: Text preprocessing
- lowercase text
- remove punctuation / unnecessary symbols
- remove stop words
- remove stop phrases if needed
- remove duplicate or empty records

#### Step 3: NLP experimentation
Test **3 vectorisation methods**:
1. **CountVectorizer**
2. **N-grams** (unigrams and bigrams)
3. **TF-IDF**

#### Step 4: EDA for each method
For each vectorisation method, perform EDA to examine:
- top words
- top phrases
- distinctive terms
- differences between the two podcasts

#### Step 5: Modelling
Train and compare baseline models using the same train/test split for fairness.

Possible baseline models:
- Logistic Regression
- Naive Bayes
- Linear SVM

#### Step 6: Model selection
Choose the best-performing method and optimize it to reach **90%+ accuracy**.

#### Step 7: Explainability
For the final chosen model, show:
- the most important words/phrases that distinguish each podcast
- theme tags
- recommendation reasons such as:
  - “Recommended because this episode contains themes related to policy, inequality, and social change.”

#### Step 8: Recommendation output
After predicting which podcast best matches a user’s interests, recommend **2 episodes** from that podcast with explanations.

---

## Stretch Goals (only after MVP works)
Once the MVP reaches the target accuracy, possible extensions include:
- adding more podcasts
- multilingual podcast discovery
- stronger Spotify integration
- richer recommender logic
- dashboard improvements

---

## Success Metrics

### Primary technical metric
- **Accuracy ≥ 90%**
- also report:
  - **F1 score**
  - **confusion matrix**

### Secondary recommendation goals
- recommend **2 episodes** from the predicted better-matching podcast
- each recommendation must include:
  - theme tags
  - keywords/phrases
  - a short explanation

### Optional stretch metrics
- HitRate@K
- Precision@K
- small user relevance testing

---

## Repo Structure
- `data_raw/` : raw extracted data
- `data_clean/` : cleaned data ready for modelling
- `notebooks/` : EDA and modelling notebooks
- `outputs/figures/` : charts, confusion matrices, visuals
- `outputs/models/` : saved models and results tables
- `docs/` : notes, rubric, and supporting project documents

---

## Notes
- This project is intentionally scoped to **2 podcasts first** to maximize model performance.
- The main goal of the MVP is to prove the concept with a strong classifier before scaling up.
- Spotify is used as the platform layer, while the training text comes from a non-Spotify source.