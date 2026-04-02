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

#### Step 2: Text preprocessing (3 rounds of cleaning)
- Round 1: Standard NLP — lowercase, remove URLs, special characters, stop words
- Round 2: Targeted leakage removal — platform boilerplate, host names, podcast identity words
- Round 3: Residual leakage — host name completion (shankar, vedantam)

#### Step 3: NLP experimentation
Test **3 vectorisation methods**:
1. **CountVectorizer**
2. **N-grams** (unigrams and bigrams)
3. **TF-IDF**

###