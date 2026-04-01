# Labeling Guidelines for Capstone MVP
## Hidden Brain vs CNA Deep Dive

## Purpose
This project focuses on social commentary / social issues podcast content.

For the MVP, the main task is a **binary classification problem**:
- **Class 0 = Hidden Brain**
- **Class 1 = CNA Deep Dive**

The purpose of this labeling guide is to:
1. keep the dataset aligned with the project's social-issues theme
2. define what kinds of episodes should be included
3. ensure the dataset is clean and consistent before modelling

---

## MVP Label Definition

### Class 0: Hidden Brain
Assign this label when the episode belongs to **Hidden Brain**.

Typical characteristics may include:
- science journalism and evidence-based discussion
- psychology, neuroscience, and behavioural science lens
- research-informed perspective on human behaviour and social patterns
- broad social, psychological, and cultural themes
- explanatory or conceptual framing backed by academic studies

### Class 1: CNA Deep Dive
Assign this label when the episode belongs to **CNA Deep Dive**.

Typical characteristics may include:
- newsroom or journalism-style discussion
- current affairs / issue explainer format
- stronger Singapore relevance
- public-interest or public-policy angle

---

## Thematic Scope: What counts as "social issues / social commentary"
Even though the MVP predicts podcast identity, the project should stay within a social-issues theme.

Include episodes that discuss topics such as:
- public policy
- politics and governance
- social inequality
- housing and cost of living
- healthcare
- education
- labour and work
- climate and environment
- gender and women's rights
- migration and identity
- media, misinformation, and public discourse
- culture and society
- human behaviour and psychology with social implications

---

## Exclude or review carefully
These types of episodes should be excluded or flagged for review:
- trailers
- promotional announcements
- very short filler episodes
- duplicate uploads
- missing or nearly empty descriptions
- content with no clear social issue / social commentary angle
- purely entertainment or lifestyle content with no broader societal discussion

---

## Inclusion Rules
Include an episode if:
1. it clearly belongs to either **Hidden Brain** or **CNA Deep Dive**
2. it has enough usable text (title + description)
3. it fits the broad project theme of social commentary / social issues

---

## Exclusion Rules
Exclude an episode if:
1. the title/description is empty or too short to be meaningful
2. it is a duplicate of another record
3. it is only a trailer, promo, or housekeeping update
4. it does not fit the broader project theme at all

---

## Label Values
- `0` = Hidden Brain
- `1` = CNA Deep Dive

---

## Handling Borderline Cases
If an episode belongs to one of the two podcasts but does not strongly fit the social-issues theme:
- keep a note in the `notes` column
- decide whether to exclude it based on how much off-topic content you want in the dataset

Rule of thumb:
- if the episode still discusses society, institutions, public life, or collective issues, include it
- if it is too far from the theme and may weaken the project narrative, exclude it

---

## Why this rubric matters
This rubric helps the project in 3 ways:
1. it keeps the dataset aligned with the capstone theme
2. it makes preprocessing and modelling more consistent
3. it supports explainability later, because the recommendations can point to meaningful themes and keywords

---

## Notes for future scaling
This MVP only focuses on **2 podcasts** for performance.

Later, this rubric can be expanded to:
- more podcasts
- multilingual podcasts
- theme-level classification
- stronger recommendation logic