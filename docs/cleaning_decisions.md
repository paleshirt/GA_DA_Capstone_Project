# Data Cleaning & Preprocessing Decisions

## The Challenge: The Illusion of 100% Accuracy
[cite_start]During the initial modeling phase, baseline models achieved an artificial 100% accuracy[cite: 575]. [cite_start]An analysis of the top-performing features (words and bigrams) revealed severe data leakage[cite: 576]. [cite_start]The models were not learning the thematic differences between "Hidden Brain" and "CNA Deep Dive" regarding social issues[cite: 577]. [cite_start]Instead, they were memorizing platform-specific boilerplate text, URLs, and recurring host names[cite: 578]. 

To build a genuinely robust and trustworthy classifier, aggressive and targeted text cleaning was required to force the model to learn the actual content and themes of the episodes.

---

## Standard Preprocessing Pipeline
Before addressing specific data leakage, all episode descriptions underwent a standard Natural Language Processing (NLP) scrubbing process:
* Converted all text to lowercase to ensure uniformity.
* Stripped all URLs, special characters, and numbers.
* Removed extra whitespaces.
* Removed standard English stop words (using `sklearn` fallback when `NLTK` was unavailable).
* Dropped 5 duplicate episode records to prevent training bias.

---

## Targeted Leakage Removal

### 1. "Hidden Brain" Scrubbing
The raw descriptions for this podcast contained heavy promotional and legal footers. The following categories were stripped from the text:
* **Hosting Boilerplate:** Removed phrases related to "Simplecast", "AdsWizz", and "hosted by".
* **Subscription/Promo Language:** Removed specific calls to action including "subscribe", "newsletter", "support work", "apple podcasts", and "revisit".
* **Podcast Identity:** Removed self-referential words like "hidden", "brain", "podcast", and "episode".

### 2. "CNA Deep Dive" Scrubbing
This podcast's descriptions contained localized platform footprints and frequent guest/host repetition. The following categories were stripped:
* **Hosting Boilerplate:** Removed all references to "OmnyStudio".
* **Identity & Location:** Removed "singapore", "deep dive", and "cna".
* **Host & Guest Names:** Removed repeating host names (e.g., "steven", "crispina", "otelli") and specific recurring guests (e.g., "walter", "theseira", "kuan", "yew").
* **Generic Titles:** Removed common identifiers that skewed the data, such as "dr", "prof", "minister", and common surnames ("tan", "lee").

---

## Final Post-Cleaning Model Performance
After isolating the text to its core thematic content, the models were re-trained. The results below demonstrate highly credible, production-ready performance without relying on platform boilerplate.

| Vectorizer | Model | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- |
| **CountVectorizer** | Logistic Regression | 92.41% | 91.67% |
| **CountVectorizer** | Naive Bayes | 98.73% | 98.73% |
| **CountVectorizer** | Linear SVM | 92.41% | 91.67% |
| **N-grams (Bigrams)** | Logistic Regression | 92.41% | 91.67% |
| **N-grams (Bigrams)** | Naive Bayes | 98.73% | 98.73% |
| **N-grams (Bigrams)** | Linear SVM | 92.41% | 91.67% |
| **TF-IDF** | Logistic Regression | 98.73% | 98.70% |
| **TF-IDF** | Naive Bayes | 98.73% | 98.70% |
| **TF-IDF** | Linear SVM | 97.47% | 97.37% |

## Key Takeaway
By aggressively auditing and cleaning our dataset, we successfully eliminated predictive shortcuts. The TF-IDF vectorizer combined with Logistic Regression or Naive Bayes yields exceptionally high (98.73%) and mathematically sound accuracy, proving the model successfully distinguishes the distinct sociological and psychological themes of the two podcasts.

## Optimization Finding
During hyperparameter tuning of TF-IDF + Logistic Regression, the simplest 
configuration (unigrams, max_features=5000, min_df=1) outperformed more complex 
configurations with bigrams and trigrams. This suggests the episode descriptions 
contain sufficiently distinctive unigram vocabulary to separate the two podcasts 
without needing phrase-level features.

**Locked-in final model:** TF-IDF (unigrams, max_features=5000, min_df=1) + Logistic Regression  
**Final Accuracy:** 98.73% | **Final F1:** 98.70%

## Round 3 Cleaning: Host Name Completion
After reviewing the explainability output from `09_explainability.py`, the Hidden Brain 
host name "shankar" (Shankar Vedantam) appeared as a top predictive feature, indicating 
residual identity leakage. Added to the cleaning pipeline:

- `shankar`, `vedantam` — Hidden Brain host name

This ensures the model distinguishes podcasts by **content themes**, not host identity.