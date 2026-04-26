import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
st.set_page_config(page_title="Better Questions", page_icon="🎙️", layout="wide")

# ── Colours — fixed dark Spotify-inspired theme ────────────────────────────
HB_COLOR  = "#FF6B35"   # Hidden Brain   — Orange
CNA_COLOR = "#4BC8E8"   # CNA Deep Dive  — Blue
GREEN     = "#1DB954"   # Spotify green

BG        = "#0D0D0D"
BG2       = "#1A1A1A"
BG3       = "#141414"
TEXT      = "#FFFFFF"
TEXT2     = "#AAAAAA"
TEXT3     = "#CCCCCC"
TEXT4     = "#888888"
BORDER    = "#2A2A2A"
PLOT_BG   = "#1A1A1A"
PLOT_TICK = "#CCCCCC"
PLOT_GRID = "#2A2A2A"

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

html, body, [class*="css"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 3rem !important; max-width: 1200px; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {TEXT4} !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: none !important;
    padding: 8px 22px;
    border-radius: 20px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {TEXT} !important;
    color: {BG} !important;
}}

/* Form submit button */
.stFormSubmitButton > button {{
    background: {GREEN} !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 32px !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.stFormSubmitButton > button:hover {{ background: #1ed760 !important; }}

/* Text area */
.stTextArea textarea {{
    background: {BG2} !important;
    color: {TEXT} !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 16px !important;
}}
.stTextArea textarea:focus {{ border-color: {GREEN} !important; }}

/* Expanders */
.streamlit-expanderHeader {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    color: {TEXT} !important;
    font-weight: 700;
}}
.streamlit-expanderContent {{
    background: {BG3} !important;
    border: 1px solid {BORDER} !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}}

/* Misc */
hr {{ border-color: {BORDER} !important; }}
.stCaption {{ color: {TEXT4} !important; font-size: 12px !important; }}
.stSpinner > div {{ border-top-color: {GREEN} !important; }}
[data-testid="stDataFrame"] {{ font-size: 14px !important; }}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}

/* Custom classes */
.pill-hb  {{ display:inline-block; background:{HB_COLOR};  color:#FFFFFF; font-size:13px; font-weight:700; padding:6px 14px; border-radius:20px; margin:3px; }}
.pill-cna {{ display:inline-block; background:{CNA_COLOR}; color:#000000; font-size:13px; font-weight:700; padding:6px 14px; border-radius:20px; margin:3px; }}

.stat-card  {{ background:{BG2}; border:1px solid {BORDER}; border-radius:16px; padding:24px; text-align:center; }}
.stat-num   {{ font-size:38px; font-weight:900; color:{GREEN}; line-height:1; margin-bottom:8px; }}
.stat-label {{ font-size:11px; color:{TEXT4}; text-transform:uppercase; letter-spacing:0.1em; }}

.podcast-card     {{ background:{BG2}; border:1px solid {BORDER}; border-radius:20px; padding:28px; }}
.podcast-card-hb  {{ border-top:4px solid {HB_COLOR}; }}
.podcast-card-cna {{ border-top:4px solid {CNA_COLOR}; }}

.result-header    {{ background:#111111; border:1px solid {GREEN}; border-radius:20px; padding:28px; margin:16px 0; }}
.confidence-num   {{ font-size:52px; font-weight:900; color:{GREEN}; line-height:1; }}
.confidence-label {{ font-size:11px; color:{TEXT4}; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px; }}

.why-box       {{ background:#0A150A; border-left:3px solid {GREEN}; border-radius:0 12px 12px 0; padding:16px 20px; margin:16px 0; font-size:14px; line-height:1.7; color:{TEXT3}; }}
.section-label {{ font-size:11px; font-weight:700; color:{TEXT4}; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px; }}
.example-phrase{{ background:{BG2}; border:1px solid {BORDER}; border-radius:10px; padding:10px 14px; font-size:13px; color:{TEXT3}; margin-bottom:6px; line-height:1.5; }}
</style>
""", unsafe_allow_html=True)

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

# ── Synonym expansion ──────────────────────────────────────────────────────
SYNONYM_MAP = {
    "stress":        ["work","pressure","mental","health","emotion"],
    "burnout":       ["work","pressure","mental","health"],
    "anxiety":       ["mental","health","fear","psychological"],
    "depression":    ["mental","health","psychological","mood","emotion"],
    "bias":          ["unconscious","psychological","behavior","decision"],
    "irrational":    ["behavior","decision","psychological","emotion"],
    "housing":       ["public","policy","government","national","social"],
    "hdb":           ["public","policy","government","national"],
    "cost":          ["public","policy","social","national"],
    "jobs":          ["workers","business","national","social","policy"],
    "inequality":    ["social","policy","public","national","workers"],
    "racism":        ["social","policy","bias","psychological","behavior"],
    "climate":       ["environment","policy","national","public","social"],
    "relationships": ["conversation","people","psychological","lives"],
    "happiness":     ["psychological","people","lives","purpose","emotion"],
    "leadership":    ["psychological","people","work","behavior","decision"],
    "mental":        ["psychological","health","emotion","people","lives"],
    "fear":          ["psychological","emotion","behavior","people"],
    "grief":         ["psychological","emotion","lives","people"],
    "identity":      ["psychological","social","people","behavior"],
    "education":     ["school","national","public","policy","social"],
    "poverty":       ["social","policy","public","national","workers"],
}

def expand_input(text):
    words = text.lower().split()
    expansions = [exp for w in words if w in SYNONYM_MAP for exp in SYNONYM_MAP[w]]
    return text + " " + " ".join(expansions) if expansions else text

def clean_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in stop_words)

def get_spotify_episode_id(url):
    m = re.search(r"episode/([a-zA-Z0-9]+)", url)
    return m.group(1) if m else None

# ── Podcast metadata ───────────────────────────────────────────────────────
PODCAST_INFO = {
    "Hidden Brain": {
        "host":         "Shankar Vedantam",
        "style":        "Psychology · Behaviour · Human Experience",
        "description":  "Hidden Brain explores the unconscious patterns that drive human behaviour. Hosted by science journalist Shankar Vedantam, each episode draws on psychology, neuroscience, and social science to explain why we think and act the way we do.",
        "show_id":      "20Gf4IAauFrfj7RBkjcWxh",
        "color":        HB_COLOR,
        "pill_class":   "pill-hb",
        "card_class":   "podcast-card-hb",
        "label":        "The Inside View",
        "identity":     "psychology and human behaviour",
        "lens":         "It explores the psychological and behavioural science behind this topic.",
        "best_for": [
            "Why people make irrational decisions",
            "The psychology of relationships and emotions",
            "Unconscious bias and behaviour",
            "Motivation, happiness, and purpose",
            "Social patterns and human connection",
        ],
    },
    "CNA Deep Dive": {
        "host":         "Steven Chia & Tiffany Ang",
        "style":        "Policy · Society · Singapore & Asia",
        "description":  "CNA Deep Dive unpacks Singapore's most pressing social, economic, and political issues. Hosted by Steven Chia and Tiffany Ang, each episode brings in expert guests to explain the context behind the headlines.",
        "show_id":      "2hcojizvVOLz8dTRblRuSC",
        "color":        CNA_COLOR,
        "pill_class":   "pill-cna",
        "card_class":   "podcast-card-cna",
        "label":        "The Outside View",
        "identity":     "Singapore social issues and public policy",
        "lens":         "It examines this topic through a Singapore and public policy lens.",
        "best_for": [
            "Singapore housing and cost of living",
            "Government policy and public services",
            "Mental health in Singapore society",
            "Climate change and sustainability",
            "Education, workforce, and social inequality",
        ],
    },
}

# ── Recommendation engine ──────────────────────────────────────────────────
def recommend(user_input, n=2):
    expanded     = expand_input(user_input)
    cleaned      = clean_input(expanded)
    user_vec     = tfidf.transform([cleaned])
    pred_label   = model.predict(user_vec)[0]
    pred_proba   = model.predict_proba(user_vec)[0]
    confidence   = round(max(pred_proba) * 100, 1)
    podcast_name = "Hidden Brain" if pred_label == 0 else "CNA Deep Dive"

    feature_list = list(feature_names)
    coef_sign    = coefficients if pred_label == 1 else -coefficients
    input_words  = cleaned.split()
    word_scores  = {w: coef_sign[feature_list.index(w)] for w in input_words if w in feature_list}
    top_keywords = sorted(word_scores, key=word_scores.get, reverse=True)[:5]
    theme_str    = ", ".join(top_keywords) if top_keywords else "social issues"

    matched_clean = clean_df[clean_df["label"] == pred_label].copy().reset_index(drop=True)
    matched_raw   = raw_df[raw_df["podcast"] == podcast_name].copy()
    matched_raw   = matched_raw[matched_raw["title"].isin(matched_clean["title"])].copy().reset_index(drop=True)

    ep_vecs         = tfidf.transform(matched_clean["clean_text"].fillna(""))
    cos_scores      = cosine_similarity(user_vec, ep_vecs)[0]

    def overlap(ep_text, kws):
        if not kws: return 0
        words = set(str(ep_text).lower().split())
        return sum(1 for k in kws if k in words) / len(kws)

    desc_overlap    = matched_clean["clean_text"].apply(lambda x: overlap(x, input_words)).values
    title_overlap   = matched_raw["title"].apply(lambda x: overlap(str(x).lower(), input_words)).values
    combined        = 0.6 * cos_scores + 0.2 * desc_overlap + 0.2 * title_overlap
    matched_clean["score"] = combined

    top_idx      = matched_clean.nlargest(n, "score").index.tolist()
    top_episodes = matched_raw.loc[top_idx]
    return podcast_name, confidence, theme_str, top_keywords, top_episodes, matched_clean, top_idx


# ══════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding:2rem 0 1rem">
  <div style="font-size:11px;font-weight:700;color:{GREEN};text-transform:uppercase;
              letter-spacing:0.15em;margin-bottom:8px">
    Social-Issues Podcast Recommender
  </div>
  <div style="font-size:48px;font-weight:900;letter-spacing:-0.03em;
              line-height:1.05;margin-bottom:16px;color:{TEXT}">
    Better Questions
  </div>
  <div style="font-size:15px;color:{TEXT2};max-width:620px;line-height:1.7">
    Free podcasts are a vital stepping stone for self-discovery and healing.
    Type what's on your mind — we'll find the expert conversation that meets you there.
  </div>
</div>
<hr style="border-color:{BORDER};margin:0 0 24px">
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Find My Podcast", "About the Podcasts", "Data Insights"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — FIND MY PODCAST
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What are you curious about?</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:14px;color:{TEXT2};margin-bottom:20px">'
        'Describe a topic, a feeling, or a question. We match you to the podcast whose '
        'vocabulary mirrors yours — and show exactly which words drove that choice.</div>',
        unsafe_allow_html=True
    )
    st.caption("💡 Mac: ⌘ + Enter  |  Windows: Ctrl + Enter")

    with st.form(key="search_form", clear_on_submit=False):
        user_input = st.text_area(
            "",
            placeholder='"why do people make irrational decisions"  or  "Singapore housing policy"',
            height=110,
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Find My Podcast →", type="primary")

    if submitted:
        if user_input.strip():
            with st.spinner("Matching your interests..."):
                podcast_name, confidence, theme_str, top_keywords, \
                top_episodes, matched_clean, top_idx = recommend(user_input)

            info = PODCAST_INFO[podcast_name]

            # Result card
            st.markdown(f"""
            <div class="result-header">
              <div style="display:flex;justify-content:space-between;
                          align-items:flex-start;flex-wrap:wrap;gap:20px">
                <div>
                  <div style="font-size:11px;font-weight:700;color:{info['color']};
                              text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px">
                    {info['label']}
                  </div>
                  <div style="font-size:32px;font-weight:900;letter-spacing:-0.02em;
                              margin-bottom:4px;color:{TEXT}">
                    {podcast_name}
                  </div>
                  <div style="font-size:13px;color:{TEXT4}">
                    {info['style']} · {info['host']}
                  </div>
                </div>
                <div style="text-align:right">
                  <div class="confidence-num">{confidence}%</div>
                  <div class="confidence-label">Match confidence</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Why explanation
            st.markdown(
                f'<div class="why-box">'
                f'<strong style="color:{GREEN}">Why this podcast?</strong><br>'
                f'Your input contained themes related to <strong>{theme_str}</strong> — '
                f'words strongly associated with <em>{podcast_name}</em>\'s focus on '
                f'{info["identity"]}.</div>',
                unsafe_allow_html=True
            )

            # Key phrase pills
            if top_keywords:
                st.markdown(
                    '<div class="section-label" style="margin-top:20px">Key phrases detected</div>',
                    unsafe_allow_html=True
                )
                pills = " ".join(
                    f'<span class="{info["pill_class"]}">{kw}</span>'
                    for kw in top_keywords
                )
                st.markdown(pills, unsafe_allow_html=True)

            # Episode recommendations
            st.markdown(
                '<div class="section-label" style="margin-top:24px">Top 2 episodes for you</div>',
                unsafe_allow_html=True
            )
            for rank, (idx, row) in enumerate(top_episodes.iterrows(), 1):
                score      = matched_clean.loc[top_idx[rank - 1], "score"]
                episode_id = get_spotify_episode_id(row["spotify_url"])
                with st.expander(
                    f"#{rank}  {row['title']}  —  {round(score * 100, 1)}% match",
                    expanded=True
                ):
                    st.markdown(
                        f'<div style="color:{TEXT4};font-size:13px;margin-bottom:8px">'
                        f'Released {row["release_date"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="font-size:14px;color:{TEXT3};line-height:1.6;margin-bottom:16px">'
                        f'This episode from <em>{podcast_name}</em> closely matches your interest in '
                        f'<strong>{theme_str}</strong>. {info["lens"]}</div>',
                        unsafe_allow_html=True
                    )
                    if episode_id:
                        components.html(
                            f'<iframe style="border-radius:12px" '
                            f'src="https://open.spotify.com/embed/episode/{episode_id}'
                            f'?utm_source=generator&theme=0" '
                            f'width="100%" height="152" frameBorder="0" allowfullscreen="" '
                            f'allow="autoplay; clipboard-write; encrypted-media; '
                            f'fullscreen; picture-in-picture" loading="lazy"></iframe>',
                            height=160
                        )
                    else:
                        st.markdown(f"[Listen on Spotify →]({row['spotify_url']})")
        else:
            st.warning("Please describe what you're interested in.")

    # Example phrases
    st.markdown(f'<hr style="border-color:{BORDER};margin:32px 0 20px">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Not sure what to type? Try these</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<span class="pill-hb" style="margin-bottom:12px;display:inline-block">Hidden Brain</span>',
            unsafe_allow_html=True
        )
        for p in [
            "the psychology behind why people avoid making difficult decisions",
            "unconscious bias and how it shapes human behavior",
            "the science of happiness and what truly motivates people",
        ]:
            st.markdown(f'<div class="example-phrase">"{p}"</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<span class="pill-cna" style="margin-bottom:12px;display:inline-block">CNA Deep Dive</span>',
            unsafe_allow_html=True
        )
        for p in [
            "government policy on housing and cost of living in Singapore",
            "mental health support and social services for young people",
            "Singapore's approach to climate change and sustainability",
        ]:
            st.markdown(f'<div class="example-phrase">"{p}"</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT THE PODCASTS
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:15px;color:{TEXT2};margin-bottom:28px">'
        'Two podcasts. Both cover social issues. Neither sounds the same.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")
    for col, key in zip([col1, col2], ["Hidden Brain", "CNA Deep Dive"]):
        info = PODCAST_INFO[key]
        with col:
            best_for_html = "".join(
                f'<div style="font-size:13px;color:{TEXT3};padding:6px 0;'
                f'border-bottom:1px solid {BORDER}">→ {t}</div>'
                for t in info["best_for"]
            )
            st.markdown(f"""
            <div class="podcast-card {info['card_class']}">
              <div style="font-size:11px;font-weight:700;color:{info['color']};
                          text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px">
                {info['label']}
              </div>
              <div style="font-size:26px;font-weight:900;margin-bottom:4px;color:{TEXT}">
                {key}
              </div>
              <div style="font-size:13px;color:{TEXT4};margin-bottom:16px">
                {info['host']}
              </div>
              <div style="font-size:14px;color:{TEXT3};line-height:1.7;margin-bottom:20px">
                {info['description']}
              </div>
              <div style="font-size:11px;font-weight:700;color:{TEXT4};
                          text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
                Best for
              </div>
              {best_for_html}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            components.html(
                f'<iframe style="border-radius:12px" '
                f'src="https://open.spotify.com/embed/show/{info["show_id"]}'
                f'?utm_source=generator&theme=0" '
                f'width="100%" height="152" frameBorder="0" allowfullscreen="" '
                f'allow="autoplay; clipboard-write; encrypted-media; '
                f'fullscreen; picture-in-picture" loading="lazy"></iframe>',
                height=160
            )

    st.markdown(f'<hr style="border-color:{BORDER};margin:32px 0 20px">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What separates them?</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            f'<div style="background:{BG2};border-left:3px solid {HB_COLOR};'
            f'border-radius:0 12px 12px 0;padding:16px 20px;font-size:14px;'
            f'color:{TEXT3};line-height:1.7">'
            f'<strong style="color:{HB_COLOR}">Hidden Brain</strong> approaches social issues '
            f'from the <strong>inside out</strong> — starting with individual psychology and '
            f'working outward. Episodes are topic-focused and timeless.</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div style="background:{BG2};border-left:3px solid {CNA_COLOR};'
            f'border-radius:0 12px 12px 0;padding:16px 20px;font-size:14px;'
            f'color:{TEXT3};line-height:1.7">'
            f'<strong style="color:{CNA_COLOR}">CNA Deep Dive</strong> approaches social issues '
            f'from the <strong>outside in</strong> — starting with policy, institutions, and '
            f'current events. Episodes are timely and Singapore-focused.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:15px;color:{TEXT2};margin-bottom:28px">'
        'Trained on 395 real episode descriptions. The model learned the vocabulary of '
        'each podcast — without being told what to look for.</div>',
        unsafe_allow_html=True
    )

    # Accuracy stats
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">98.73%</div>'
            f'<div class="stat-label">Accuracy</div>'
            f'<div style="font-size:12px;color:{TEXT4};margin-top:8px">78 of 79 episodes correct</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">98.70%</div>'
            f'<div class="stat-label">F1 Score</div>'
            f'<div style="font-size:12px;color:{TEXT4};margin-top:8px">Balanced across both podcasts</div></div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">1/79</div>'
            f'<div class="stat-label">Wrong predictions</div>'
            f'<div style="font-size:12px;color:{TEXT4};margin-top:8px">One understandable edge case</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:#0A150A;border:1px solid {GREEN};border-radius:12px;'
        f'padding:14px 20px;font-size:14px;color:{GREEN};font-weight:700;margin-bottom:32px">'
        f'✓ Exceeds the 90% accuracy target set at project start</div>',
        unsafe_allow_html=True
    )

    # Signature words charts
    st.markdown('<div class="section-label">The signature words of each podcast</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:14px;color:{TEXT2};margin-bottom:20px">'
        'Discovered by the model — never manually labelled. '
        'The longer the bar, the more that word is a signature of that show.</div>',
        unsafe_allow_html=True
    )
    try:
        features_df = pd.read_csv("outputs/tableau_top_features.csv")
        col1, col2  = st.columns(2, gap="large")
        chart_meta  = [
            (col1, "Hidden Brain",  HB_COLOR,  "pill-hb",  "The language of internal experience"),
            (col2, "CNA Deep Dive", CNA_COLOR, "pill-cna", "The language of external structures"),
        ]
        for col, podcast, color, pill_class, subtitle in chart_meta:
            with col:
                st.markdown(
                    f'<span class="{pill_class}">{podcast}</span>'
                    f'<div style="font-size:12px;color:{TEXT4};margin:8px 0 12px;'
                    f'font-style:italic">{subtitle}</div>',
                    unsafe_allow_html=True
                )
                df  = features_df[features_df["podcast"] == podcast].nlargest(10, "abs_score")
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor(PLOT_BG)
                ax.set_facecolor(PLOT_BG)
                ax.barh(df["word"], df["abs_score"], color=color, height=0.65)
                ax.invert_yaxis()
                ax.set_xlabel("Word Importance", color=PLOT_TICK, fontsize=10)
                ax.tick_params(colors=PLOT_TICK, labelsize=11)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.grid(axis="x", color=PLOT_GRID, linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    except Exception:
        st.info("Feature chart data not found — check outputs/tableau_top_features.csv")

    st.markdown(f'<hr style="border-color:{BORDER};margin:32px 0 20px">', unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="section-label">All 9 model approaches tested</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:14px;color:{TEXT2};margin-bottom:20px">'
        'Every combination exceeded 90% accuracy. TF-IDF + Logistic Regression was chosen '
        'for explainability, not just accuracy.</div>',
        unsafe_allow_html=True
    )
    try:
        model_df = pd.read_csv("outputs/tableau_model_comparison.csv")
        model_df["Accuracy %"] = (model_df["Accuracy"] * 100).round(2)
        model_df["F1 %"]       = (model_df["F1"] * 100).round(2)
        model_df["Result"]     = model_df["Accuracy %"].apply(lambda x: "✅" if x >= 90 else "❌")
        st.dataframe(
            model_df[["Method", "Model", "Accuracy %", "F1 %", "Result"]]
            .sort_values("Accuracy %", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Final model: TF-IDF + Logistic Regression — "
            "same accuracy as Naive Bayes, but explains its reasoning."
        )
    except Exception:
        st.info("Model comparison data not found — check outputs/tableau_model_comparison.csv")

    st.markdown(f'<hr style="border-color:{BORDER};margin:32px 0 20px">', unsafe_allow_html=True)

    # Tableau dashboard — static preview + link
    # Note: Tableau Public's embed JS cannot render correctly inside Streamlit's
    # sandboxed iframe, which causes the white background and broken legend you see.
    # The solution is a static preview image (always pixel-perfect) + a button to
    # open the full interactive dashboard in a new tab where it renders perfectly.
    st.markdown('<div class="section-label">Full interactive dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:14px;color:{TEXT2};margin-bottom:16px">'
        'Built with Tableau. The preview below shows the full dashboard — '
        'click the button to explore it interactively.</div>',
        unsafe_allow_html=True
    )

    TABLEAU_URL = (
        "https://public.tableau.com/app/profile/muhammad.nuruddin6371/viz/"
        "GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft"
    )
    TABLEAU_PREVIEW = (
        "https://public.tableau.com/static/images/GA/"
        "GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft/1.png"
    )

    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:16px;
                padding:28px;text-align:center">
      <a href="{TABLEAU_URL}" target="_blank"
         style="display:inline-block;background:{GREEN};color:#000000;
                font-weight:700;font-size:14px;padding:14px 32px;border-radius:30px;
                text-decoration:none;letter-spacing:0.04em;text-transform:uppercase;
                margin-bottom:24px">
        → Open Interactive Dashboard
      </a>
      <br>
      <a href="{TABLEAU_URL}" target="_blank">
        <img src="{TABLEAU_PREVIEW}"
             style="width:100%;border-radius:12px;border:1px solid {BORDER};
                    display:block;margin-top:8px"
             alt="Tableau dashboard preview" />
      </a>
      <div style="font-size:12px;color:{TEXT4};margin-top:12px">
        Click the image or button to open the full interactive version
      </div>
    </div>
    """, unsafe_allow_html=True)
