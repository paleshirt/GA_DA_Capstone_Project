import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# ── Credentials ────────────────────────────────────────────────────────────
try:
    SPOTIFY_CLIENT_ID     = st.secrets["SPOTIFY_CLIENT_ID"]
    SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
except Exception:
    load_dotenv()
    SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Better Questions — Podcast Recommender",
    page_icon="🎙️",
    layout="wide"
)

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("outputs/models/logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

@st.cache_data
def load_data():
    raw_df   = pd.read_csv("data_raw/episodes_raw.csv")
    clean_df = pd.read_csv("data_clean/episodes_clean.csv")
    return raw_df, clean_df

tfidf, model     = load_model()
raw_df, clean_df = load_data()
feature_names    = tfidf.get_feature_names_out()
coefficients     = model.coef_[0]
stop_words       = set(ENGLISH_STOP_WORDS)

# ── Synonym expansion map ──────────────────────────────────────────────────
SYNONYM_MAP = {
    "stress":        ["work", "pressure", "mental", "health", "emotion"],
    "burnout":       ["work", "pressure", "mental", "health"],
    "anxiety":       ["mental", "health", "fear", "psychological"],
    "depression":    ["mental", "health", "psychological", "mood", "emotion"],
    "bias":          ["unconscious", "psychological", "behavior", "decision"],
    "irrational":    ["behavior", "decision", "psychological", "emotion"],
    "housing":       ["public", "policy", "government", "national", "social"],
    "hdb":           ["public", "policy", "government", "national"],
    "cost":          ["public", "policy", "social", "national"],
    "jobs":          ["workers", "business", "national", "social", "policy"],
    "inequality":    ["social", "policy", "public", "national", "workers"],
    "racism":        ["social", "policy", "bias", "psychological", "behavior"],
    "climate":       ["environment", "policy", "national", "public", "social"],
    "relationships": ["conversation", "people", "psychological", "lives"],
    "happiness":     ["psychological", "people", "lives", "purpose", "emotion"],
    "leadership":    ["psychological", "people", "work", "behavior", "decision"],
    "mental":        ["psychological", "health", "emotion", "people", "lives"],
    "fear":          ["psychological", "emotion", "behavior", "people"],
    "grief":         ["psychological", "emotion", "lives", "people"],
    "identity":      ["psychological", "social", "people", "behavior"],
    "education":     ["school", "national", "public", "policy", "social"],
    "poverty":       ["social", "policy", "public", "national", "workers"],
}

def expand_input(text):
    words = text.lower().split()
    expansions = []
    for word in words:
        if word in SYNONYM_MAP:
            expansions.extend(SYNONYM_MAP[word])
    if expansions:
        return text + " " + " ".join(expansions)
    return text

# ── Text cleaning ──────────────────────────────────────────────────────────
def clean_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# ── Spotify episode ID extractor ───────────────────────────────────────────
def get_spotify_episode_id(url):
    match = re.search(r"episode/([a-zA-Z0-9]+)", url)
    return match.group(1) if match else None

# ── Podcast info ────────────────────────────────────────────────────────────
PODCAST_INFO = {
    "Hidden Brain": {
        "host": "Shankar Vedantam",
        "style": "Academic · Psychology · Human Behaviour",
        "description": (
            "Hidden Brain explores the unconscious patterns that drive human behaviour. "
            "Hosted by science journalist Shankar Vedantam, each episode draws on research "
            "from psychology, neuroscience, and social science to explain why we think and "
            "act the way we do."
        ),
        "spotify_show_id": "20Gf4IAauFrfj7RBkjcWxh",
        "spotify_show":    "https://open.spotify.com/show/20Gf4IAauFrfj7RBkjcWxh",
        "color": "🟠",
        "identity": "psychology and human behaviour",
        "lens": "It explores the psychological and behavioural science behind this topic.",
        "best_for": [
            "Why people make irrational decisions",
            "The psychology of relationships and emotions",
            "Unconscious bias and behaviour",
            "Motivation, happiness, and purpose",
            "Social patterns and human connection",
        ],
    },
    "CNA Deep Dive": {
        "host": "Steven Chia & Tiffany Ang",
        "style": "Newsroom · Public Policy · Singapore & Asia",
        "description": (
            "CNA Deep Dive unpacks Singapore's most pressing social, economic, and political "
            "issues. Hosted by Steven Chia and Tiffany Ang, each episode brings in expert "
            "guests to explain the context behind the headlines — from housing policy to "
            "mental health legislation."
        ),
        "spotify_show_id": "2hcojizvVOLz8dTRblRuSC",
        "spotify_show":    "https://open.spotify.com/show/2hcojizvVOLz8dTRblRuSC",
        "color": "🔵",
        "identity": "Singapore social issues and public policy",
        "lens": "It examines this topic through a Singapore and public policy lens.",
        "best_for": [
            "Singapore housing and cost of living",
            "Government policy and public services",
            "Mental health in Singapore society",
            "Climate change and sustainability",
            "Education, workforce, and social inequality",
        ],
    },
}

# ── Recommendation logic ───────────────────────────────────────────────────
def recommend(user_input, n=2):
    expanded   = expand_input(user_input)
    cleaned    = clean_input(expanded)
    user_vec   = tfidf.transform([cleaned])

    pred_label   = model.predict(user_vec)[0]
    pred_proba   = model.predict_proba(user_vec)[0]
    confidence   = round(max(pred_proba) * 100, 1)
    podcast_name = "Hidden Brain" if pred_label == 0 else "CNA Deep Dive"

    feature_list = list(feature_names)
    coef_sign    = coefficients if pred_label == 1 else -coefficients
    input_words  = cleaned.split()
    word_scores  = {
        w: coef_sign[feature_list.index(w)]
        for w in input_words if w in feature_list
    }
    top_keywords = sorted(word_scores, key=word_scores.get, reverse=True)[:5]
    theme_str    = ", ".join(top_keywords) if top_keywords else "social issues"

    matched_clean = clean_df[clean_df["label"] == pred_label].copy()
    matched_raw   = raw_df[raw_df["podcast"] == podcast_name].copy()
    matched_raw   = matched_raw[
        matched_raw["title"].isin(matched_clean["title"])
    ].copy()
    matched_clean = matched_clean.reset_index(drop=True)
    matched_raw   = matched_raw.reset_index(drop=True)

    ep_vecs    = tfidf.transform(matched_clean["clean_text"].fillna(""))
    cos_scores = cosine_similarity(user_vec, ep_vecs)[0]

    def keyword_overlap(ep_text, keywords):
        if not keywords:
            return 0
        ep_words = set(str(ep_text).lower().split())
        return sum(1 for kw in keywords if kw in ep_words) / len(keywords)

    overlap_scores = matched_clean["clean_text"].apply(
        lambda x: keyword_overlap(x, input_words)).values
    title_scores   = matched_raw["title"].apply(
        lambda x: keyword_overlap(str(x).lower(), input_words)).values

    combined_scores = 0.6 * cos_scores + 0.2 * overlap_scores + 0.2 * title_scores
    matched_clean["score"] = combined_scores

    top_idx      = matched_clean.nlargest(n, "score").index.tolist()
    top_episodes = matched_raw.loc[top_idx]

    return podcast_name, confidence, theme_str, top_keywords, top_episodes, matched_clean, top_idx


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.title("🎙️ Better Questions")
st.markdown("### *Find your next social issues podcast — with transparent recommendations*")
st.markdown(
    "**High-quality mental health and social resources are often gated by financial barriers, "
    "making free podcasts a vital stepping stone for self-discovery and healing.** "
    "Type what you want to learn about — our engine scans the actual conversations and themes "
    "of hundreds of episodes on Spotify to find your best match, and explains exactly why."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Find My Podcast", "🎧 About the Podcasts", "📊 Data Insights"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("What topics are you interested in?")
    st.markdown(
        "Describe a topic, a feeling, or a question you have. "
        "The recommender will match you to the podcast whose vocabulary "
        "most closely mirrors your interests — and show you exactly which words drove that choice."
    )
    st.caption("💡 **Mac:** ⌘ + Enter to search &nbsp;|&nbsp; **Windows:** Ctrl + Enter to search")

    # st.form enables Enter key (⌘+Enter / Ctrl+Enter) to submit
    with st.form(key="search_form", clear_on_submit=False):
        user_input = st.text_area(
            "Your interests:",
            placeholder="e.g. why do people make irrational decisions... or Singapore housing policy...",
            height=100,
        )
        submitted = st.form_submit_button("🎯 Find My Podcast", type="primary")

    if submitted:
        if user_input.strip():
            with st.spinner("Matching your interests..."):
                podcast_name, confidence, theme_str, top_keywords, top_episodes, matched_clean, top_idx = recommend(user_input)

            info = PODCAST_INFO[podcast_name]
            st.markdown("---")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"## {info['color']} Matched to: **{podcast_name}**")
                st.markdown(f"*{info['style']}* · Hosted by {info['host']}")
            with col2:
                st.metric(
                    "Match Confidence", f"{confidence}%",
                    help="How confident the model is that this podcast matches your interests"
                )

            st.info(
                f"💡 **Why this podcast?** Your input contained themes related to "
                f"**{theme_str}** — words strongly associated with *{podcast_name}*'s "
                f"focus on {info['identity']}."
            )

            if top_keywords:
                st.markdown("**🔑 Key phrases detected in your input:**")
                cols = st.columns(len(top_keywords))
                for i, kw in enumerate(top_keywords):
                    cols[i].success(f"**{kw}**")

            st.markdown("---")
            st.subheader("🎧 Top 2 Episodes to Listen To")

            for rank, (idx, row) in enumerate(top_episodes.iterrows(), 1):
                score      = matched_clean.loc[top_idx[rank - 1], "score"]
                episode_id = get_spotify_episode_id(row["spotify_url"])

                with st.expander(
                    f"**#{rank}: {row['title']}** — {round(score * 100, 1)}% match",
                    expanded=True
                ):
                    st.markdown(f"**📅 Released:** {row['release_date']}")
                    st.markdown(
                        f"**🔍 Why this episode?** This episode from *{podcast_name}* "
                        f"closely matches your interest in **{theme_str}**. "
                        f"{info['lens']}"
                    )
                    if episode_id:
                        components.html(
                            f"""<iframe style="border-radius:12px"
                                src="https://open.spotify.com/embed/episode/{episode_id}?utm_source=generator"
                                width="100%" height="152" frameBorder="0" allowfullscreen=""
                                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                                loading="lazy"></iframe>""",
                            height=160
                        )
                    else:
                        st.markdown(f"[🎵 Listen on Spotify]({row['spotify_url']})")
        else:
            st.warning("Please describe what you're interested in!")

    st.markdown("---")
    st.markdown("### 💡 Not sure what to type? Try these examples:")
    col1, col2 = st.columns(2)
    examples = {
        "🟠 Hidden Brain": [
            "the psychology behind why people avoid making decisions",
            "unconscious bias and how it shapes human behavior",
            "the science of happiness and what truly motivates people",
        ],
        "🔵 CNA Deep Dive": [
            "government policy on housing and cost of living in Singapore",
            "mental health support and social services for young people",
            "Singapore's approach to climate change and sustainability",
        ],
    }
    for col, (podcast, phrases) in zip([col1, col2], examples.items()):
        with col:
            st.markdown(f"**{podcast}**")
            for phrase in phrases:
                st.markdown(f"- *{phrase}*")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: ABOUT THE PODCASTS
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("About the Podcasts")
    st.markdown(
        "This recommender covers two expert-led social issues podcasts. "
        "Both explore themes that matter to society — but from very different angles."
    )

    col1, col2 = st.columns(2)

    with col1:
        info = PODCAST_INFO["Hidden Brain"]
        st.markdown(f"### {info['color']} Hidden Brain")
        st.markdown(f"**Host:** {info['host']}")
        st.markdown(f"**Style:** {info['style']}")
        st.markdown(info["description"])
        st.markdown("**Best for topics like:**")
        for topic in info["best_for"]:
            st.markdown(f"- {topic}")
        components.html(
            """<iframe style="border-radius:12px"
                src="https://open.spotify.com/embed/show/20Gf4IAauFrfj7RBkjcWxh?utm_source=generator"
                width="100%" height="152" frameBorder="0" allowfullscreen=""
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"></iframe>""",
            height=160
        )

    with col2:
        info = PODCAST_INFO["CNA Deep Dive"]
        st.markdown(f"### {info['color']} CNA Deep Dive")
        st.markdown(f"**Host:** {info['host']}")
        st.markdown(f"**Style:** {info['style']}")
        st.markdown(info["description"])
        st.markdown("**Best for topics like:**")
        for topic in info["best_for"]:
            st.markdown(f"- {topic}")
        components.html(
            """<iframe style="border-radius:12px"
                src="https://open.spotify.com/embed/show/2hcojizvVOLz8dTRblRuSC?utm_source=generator"
                width="100%" height="152" frameBorder="0" allowfullscreen=""
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"></iframe>""",
            height=160
        )

    st.markdown("---")
    st.markdown("### 🔬 What separates them?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Hidden Brain** approaches social issues from the **inside out** — "
            "starting with individual psychology and working outward to explain collective behaviour. "
            "Episodes are topic-focused and timeless."
        )
    with col2:
        st.markdown(
            "**CNA Deep Dive** approaches social issues from the **outside in** — "
            "starting with current events, policy, and institutions and examining their impact on people. "
            "Episodes are timely, objective, and Singapore-focused."
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How the Recommender Works")
    st.markdown(
        "This recommender was trained on 395 real podcast episode descriptions. "
        "It learned to tell the two podcasts apart by their distinctive vocabulary — "
        "without being told what to look for. Here's what it discovered."
    )

    st.markdown("---")
    st.markdown("### ✅ How reliable is it?")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "98.73%",
                help="78 out of 79 test episodes were correctly matched")
    col2.metric("Balanced Score (F1)", "98.70%",
                help="Equally reliable for both podcasts")
    col3.metric("Wrong predictions", "1 out of 79",
                help="Only 1 misclassification in testing")
    st.success("✅ The recommender exceeds the 90% accuracy target.")

    st.markdown("---")
    st.markdown("### 🔑 The key phrases that define each podcast")
    st.markdown(
        "The longer the bar, the more that word is a signature of that show. "
        "These were discovered by the model — not manually chosen."
    )

    features_df = pd.read_csv("outputs/tableau_top_features.csv")
    col1, col2  = st.columns(2)

    with col1:
        st.markdown("#### 🟠 Hidden Brain's signature words")
        hb  = features_df[features_df["podcast"] == "Hidden Brain"].nlargest(10, "abs_score")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(hb["word"], hb["abs_score"], color="#FF6B35")
        ax.invert_yaxis()
        ax.set_xlabel("Word Importance")
        ax.set_title("Psychology · Behaviour · Emotion")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.caption("The language of internal human experience")

    with col2:
        st.markdown("#### 🔵 CNA Deep Dive's signature words")
        cna = features_df[features_df["podcast"] == "CNA Deep Dive"].nlargest(10, "abs_score")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(cna["word"], cna["abs_score"], color="#0099CC")
        ax.invert_yaxis()
        ax.set_xlabel("Word Importance")
        ax.set_title("Policy · Institutions · Society")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.caption("The language of external societal structures")

    st.markdown("---")
    st.markdown("### 🧪 The 9 approaches tested")
    model_df = pd.read_csv("outputs/tableau_model_comparison.csv")
    model_df["Accuracy %"]  = (model_df["Accuracy"] * 100).round(2)
    model_df["F1 %"]        = (model_df["F1"] * 100).round(2)
    model_df["✅ Above 90%"] = model_df["Accuracy %"].apply(lambda x: "✅" if x >= 90 else "❌")
    st.dataframe(
        model_df[["Method", "Model", "Accuracy %", "F1 %", "✅ Above 90%"]]
        .sort_values("Accuracy %", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )
    st.caption("Final model: **TF-IDF + Logistic Regression** — chosen for explainability, not just accuracy.")

    st.markdown("---")
    st.markdown("### 📊 Full Interactive Dashboard")
    st.markdown("Built with Tableau — scroll to explore the full data story.")

    # ── Tableau dashboard embed ────────────────────────────────────────────
    st.markdown("### 📊 Full Interactive Dashboard")
    st.markdown(
        "Explore the full data story below — from the growth of podcast content "
        "to the distinct identities of each show. Built with Tableau."
    )

    # Clean URL with parameters forcing desktop layout
    embed_url = (
        "https://public.tableau.com/views/GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft"
        "?%3Aembed=y"
        "&%3AshowVizHome=no"
        "&%3Adisplay_count=no"
        "&%3AshowTabs=no"
        "&%3Adevice=desktop"
    )

    # Tightened CSS: padding set to 0 and display set to block to remove uneven gaps
    html_code = f"""
    <div style="background-color: #FFFFFF; border-radius: 8px; padding: 0px; margin: 0px; overflow: hidden;">
        <iframe 
            src="{embed_url}" 
            width="100%" 
            height="2030" 
            frameborder="0" 
            style="border: none; display: block; margin: 0 auto;">
        </iframe>
    </div>
    """

    # Synced height to eliminate extra scrolling/white space at the bottom
    components.html(html_code, height=2030, scrolling=False)