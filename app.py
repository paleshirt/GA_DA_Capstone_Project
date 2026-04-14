import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# ── Page config (Must be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="Better Questions",
    page_icon="🎙️",
    layout="wide"
)

# ── Theme State Management ──────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Create a top-row layout for Title and Toggle
col_title, col_theme = st.columns([0.8, 0.2])

with col_theme:
    # Custom toggle for light/dark mode
    light_mode = st.toggle("🌙 Dark / ☀️ Light", value=(st.session_state.theme == 'light'))
    st.session_state.theme = 'light' if light_mode else 'dark'

# Define Theme Colors
if st.session_state.theme == 'dark':
    bg_color = "#0D0D0D"
    text_color = "#FFFFFF"
    secondary_text = "#888888"
    card_bg = "#1A1A1A"
    border_color = "#2A2A2A"
    input_bg = "#1A1A1A"
else:
    bg_color = "#FAF9F6"  # Soft Off-white
    text_color = "#1A1A1A"
    secondary_text = "#555555"
    card_bg = "#FFFFFF"
    border_color = "#E0E0E0"
    input_bg = "#FFFFFF"

# ── Global CSS (150% Scaling & Theme) ──────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

/* Global Scaling (150% effect) */
html, body, [class*="st-"] {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'Inter', sans-serif;
    font-size: 24px !important; /* Standard is 16px; 24px = 150% */
}}

/* Scaling Headers */
h1 {{ font-size: 4rem !important; margin-bottom: 1.5rem !important; }}
h2 {{ font-size: 3rem !important; }}
h3 {{ font-size: 2.2rem !important; }}
h4 {{ font-size: 1.8rem !important; }}

.block-container {{
    padding: 3rem 5rem !important;
    max-width: 1400px;
}}

/* Input Box Scaling */
.stTextInput>div>div>input {{
    background-color: {input_bg} !important;
    color: {text_color} !important;
    border: 2px solid {border_color} !important;
    border-radius: 12px;
    padding: 20px !important;
    font-size: 24px !important;
}}

/* Podcast Cards */
.podcast-card {{
    background-color: {card_bg};
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 30px;
    border: 2px solid {border_color};
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}

.theme-badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 24px;
    font-size: 18px;
    font-weight: bold;
    margin-right: 12px;
    margin-bottom: 12px;
}}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {{ gap: 20px; }}
.stTabs [data-baseweb="tab"] {{
    font-size: 24px !important;
    font-weight: 700;
    padding: 15px 0px;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Load model & Data ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Ensuring pathing matches your environment
    with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("outputs/models/logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

@st.cache_data
def load_data():
    raw_df = pd.read_csv("data_raw/episodes_raw.csv")
    clean_df = pd.read_csv("data_clean/episodes_clean.csv")
    df = pd.merge(clean_df, raw_df[['spotify_url', 'release_date', 'title']], on='title', how='left')
    return df

tfidf, model = load_model()
df = load_data()

# ── Logic Helpers ──────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b(omnystudio|adswizz|listener|privacy|information|click|here|podcast)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

# ── Main Content ────────────────────────────────────────────────────────────
with col_title:
    st.title("Better Questions")
    st.markdown(f"<p style='font-size: 28px; color:{secondary_text};'>Transparent, NLP-driven podcast recommendations.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Search & Recommend", "📊 Data & Methodology"])

# ── TAB 1: Search ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_input(
        "What are you curious about?",
        placeholder="e.g., 'Singapore housing policy' or 'Emotional behavior'",
        label_visibility="collapsed"
    )
    
    if user_input:
        with st.spinner("Analyzing..."):
            cleaned_input = clean_text(user_input)
            input_vec = tfidf.transform([cleaned_input])
            prediction = model.predict(input_vec)
            probabilities = model.predict_proba(input_vec)
            confidence = max(probabilities) * 100
            
            podcast_df = df[df['podcast'] == prediction].copy()
            corpus_vecs = tfidf.transform(podcast_df['description_clean'].fillna(""))
            similarities = cosine_similarity(input_vec, corpus_vecs).flatten()
            podcast_df['similarity'] = similarities
            top_episodes = podcast_df.sort_values(by='similarity', ascending=False).head(2)
            
            # Explainability logic
            input_words = cleaned_input.split()
            feature_names = tfidf.get_feature_names_out()
            word_coefs = [(w, model.coef_[np.where(feature_names == w)]) for w in input_words if w in feature_names]
            
            # FIXED: CNA is Blue, Hidden Brain is Orange
            highlight_color = "#0072B2" if "CNA" in prediction else "#E68A00"
            badge_bg = "rgba(0, 114, 178, 0.15)" if "CNA" in prediction else "rgba(230, 138, 0, 0.15)"
            
            st.markdown(f"### Best Match: **<span style='color:{highlight_color}'>{prediction}</span>** ({confidence:.1f}% confidence)", unsafe_allow_html=True)
            
            if word_coefs:
                st.markdown("**Key Themes Detected:**")
                badges_html = "".join([f"<span class='theme-badge' style='background-color:{badge_bg}; color:{highlight_color};'>{w.upper()}</span>" for w, c in word_coefs])
                st.markdown(badges_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_ep1, col_ep2 = st.columns(2)
            for i, (_, row) in enumerate(top_episodes.iterrows()):
                target_col = col_ep1 if i == 0 else col_ep2
                with target_col:
                    st.markdown(f"""
                    <div class="podcast-card" style="border-top: 8px solid {highlight_color};">
                        <h3 style="margin-top:0; color:{text_color};">{row['title']}</h3>
                        <p style="color:{secondary_text}; font-size:20px;">📅 {row['release_date']}</p>
                        <p style="color:{text_color}; font-size:22px; line-height:1.6;">{str(row['description'])[:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    spotify_url = str(row['spotify_url'])
                    if "spotify.com" in spotify_url:
                        embed_url = spotify_url.replace("/episode/", "/embed/episode/")
                        components.iframe(embed_url, height=200)

# ── TAB 2: Data ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Interactive Dashboard")
    # Using the fixed height from previous iteration
    tableau_html = """
    <div style='width: 100%; height: 2050px;'>
        <script type='module' src='https://public.tableau.com/javascripts/api/viz_v1.js'></script>
        <tableau-viz id='tableauViz' src='https://public.tableau.com/views/GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft' toolbar='bottom' hide-tabs></tableau-viz>
    </div>
    """
    components.html(tableau_html, height=2050, scrolling=True)