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

# ── Credentials ────────────────────────────────────────────────────────────
try:
    SPOTIFY_CLIENT_ID     = st.secrets["SPOTIFY_CLIENT_ID"]
    SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
except Exception:
    load_dotenv()
    SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

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
col_title, col_theme = st.columns([0.85, 0.15])

with col_theme:
    # Native toggle, isolated from large CSS scaling below
    light_mode = st.toggle("Light Mode", value=(st.session_state.theme == 'light'))
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
    bg_color = "#FDFBF5"  # Soft, warmer cream
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
html, body, .block-container, p, span, div {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 24px; 
}}

/* Scaling Headers */
h1 {{ font-size: 4rem !important; margin-bottom: 1.5rem !important; font-weight: 900 !important; letter-spacing: -0.5px; }}
h2 {{ font-size: 3rem !important; font-weight: 900 !important; letter-spacing: -0.5px; }}
h3 {{ font-size: 2.2rem !important; font-weight: 900 !important; letter-spacing: -0.5px; }}
h4 {{ font-size: 1.8rem !important; font-weight: 900 !important; letter-spacing: -0.5px; }}

.block-container {{
    padding: 3rem 5rem !important;
    max-width: 1400px;
}}

/* Ensure the Toggle stays small and Apple-like */
[data-testid="stToggle"] * {{
    font-size: 16px !important;
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
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
.podcast-card:hover {{
    border-color: #888888;
    transform: translateY(-4px);
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
.stTabs [data-baseweb="tab-list"] {{ 
    gap: 20px; 
    background: transparent;
    border-bottom: 2px solid {border_color};
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 24px !important;
    font-weight: 700;
    padding: 15px 0px;
    color: {secondary_text};
    background: transparent;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    color: {text_color} !important;
    border-bottom: 3px solid {text_color} !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
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
    df = pd.merge(
        clean_df,
        raw_df[['spotify_url', 'release_date', 'title']],
        on='title',
        how='left'
    )
    return df

tfidf, model = load_model()
df = load_data()

# ── Helpers ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b(omnystudio|adswizz|listener|privacy|information|click|here|podcast)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def get_podcast_color(podcast_name):
    # CNA is Blue, Hidden Brain is Orange
    if "CNA" in podcast_name:
        return "#0072B2" # Blue
    else:
        return "#E68A00" # Orange

def plot_top_features(tfidf, model, top_n=10, theme='dark'):
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = model.coef_
    classes = model.classes_
    
    top_positive_idx = np.argsort(coef)[-top_n:]
    top_negative_idx = np.argsort(coef)[:top_n]
    
    top_positive_features = feature_names[top_positive_idx]
    top_positive_coefs = coef[top_positive_idx]
    
    top_negative_features = feature_names[top_negative_idx]
    top_negative_coefs = coef[top_negative_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Dynamic background colors based on mode
    plot_bg = '#0D0D0D' if theme == 'dark' else '#FDFBF5'
    plot_text = 'white' if theme == 'dark' else 'black'
    spine_color = '#333333' if theme == 'dark' else '#CCCCCC'
    
    fig.patch.set_facecolor(plot_bg)
    
    color_pos = get_podcast_color(classes)
    ax1.barh(top_positive_features, top_positive_coefs, color=color_pos)
    ax1.set_title(f"Strongest words for\n{classes}", color=plot_text, pad=20, weight='bold', fontsize=16)
    ax1.tick_params(colors=plot_text, labelsize=12)
    ax1.set_facecolor(plot_bg)
    ax1.spines['bottom'].set_color(spine_color)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(spine_color)
    
    color_neg = get_podcast_color(classes)
    ax2.barh(top_negative_features[::-1], np.abs(top_negative_coefs[::-1]), color=color_neg)
    ax2.set_title(f"Strongest words for\n{classes}", color=plot_text, pad=20, weight='bold', fontsize=16)
    ax2.tick_params(colors=plot_text, labelsize=12)
    ax2.set_facecolor(plot_bg)
    ax2.spines['bottom'].set_color(spine_color)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(spine_color)
    
    plt.tight_layout()
    return fig

# ── App Layout ─────────────────────────────────────────────────────────────
with col_title:
    st.title("Better Questions")
    st.markdown(f"<p style='font-size: 28px; color:{secondary_text}; font-weight:700;'>Transparent, NLP-driven podcast recommendations.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Search & Recommend", "Data & Methodology"])

# ── TAB 1: Search ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_input(
        "What are you curious about?",
        placeholder="e.g., 'I want to understand Singapore housing policy' or 'Why do we make irrational decisions?'",
        label_visibility="collapsed"
    )
    
    if user_input:
        with st.spinner("Analyzing semantics..."):
            cleaned_input = clean_text(user_input)
            input_vec = tfidf.transform([cleaned_input])
            
            prediction = model.predict(input_vec)
            probabilities = model.predict_proba(input_vec)
            confidence = max(probabilities) * 100
            
            predicted_podcast = prediction
            podcast_df = df[df['podcast'] == predicted_podcast].copy()
            
            # Find best episodes via cosine similarity
            corpus_vecs = tfidf.transform(podcast_df['description_clean'].fillna(""))
            similarities = cosine_similarity(input_vec, corpus_vecs).flatten()
            podcast_df['similarity'] = similarities
            top_episodes = podcast_df.sort_values(by='similarity', ascending=False).head(2)
            
            # Extract keywords (explainability)
            input_words = cleaned_input.split()
            feature_names = tfidf.get_feature_names_out()
            word_coefs = []
            for word in input_words:
                if word in feature_names:
                    idx = np.where(feature_names == word)
                    coef = model.coef_[idx]
                    word_coefs.append((word, coef))
            
            highlight_color = "#E68A00" if "Hidden Brain" in predicted_podcast else "#0072B2"
            badge_bg = "rgba(230, 138, 0, 0.15)" if "Hidden Brain" in predicted_podcast else "rgba(0, 114, 178, 0.15)"
            
            st.markdown(f"### Best Match: **<span style='color:{highlight_color}'>{predicted_podcast}</span>** ({confidence:.1f}% match)", unsafe_allow_html=True)
            
            if word_coefs:
                st.markdown(f"<strong style='color:{text_color};'>Why we chose this (Key Themes):</strong>", unsafe_allow_html=True)
                badges_html = ""
                for word, coef in word_coefs:
                    badges_html += f"<span class='theme-badge' style='background-color:{badge_bg}; color:{highlight_color}; border: 1px solid {highlight_color}'>{word.upper()}</span>"
                st.markdown(badges_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display Episodes
            col1, col2 = st.columns(2)
            cols = [col1, col2]
            
            for i, (_, row) in enumerate(top_episodes.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="podcast-card" style="border-top: 6px solid {highlight_color};">
                        <h4 style="margin-top:0; color:{text_color};">{row['title']}</h4>
                        <p style="color:{secondary_text}; font-size:18px; margin-bottom:16px;">📅 {row['release_date']}</p>
                        <p style="color:{text_color}; font-size:20px; line-height:1.5;">{str(row['description'])[:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    spotify_url = str(row['spotify_url'])
                    if "spotify.com/episode/" in spotify_url or "spotify.com" in spotify_url:
                        embed_url = spotify_url.replace("/episode/", "/embed/episode/")
                        components.iframe(embed_url, height=152)
                    else:
                        st.markdown(f"[Listen on Spotify]({spotify_url})")

# ── TAB 2: Data ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Under the Hood")
    
    st.markdown("#### 1. What the Model Learned")
    st.markdown(
        f"<p style='color:{secondary_text}; font-size:20px;'>By analyzing 400 episode descriptions, the TF-IDF model organically separated the vocabulary "
        "of psychology (Hidden Brain) from the vocabulary of policy (CNA Deep Dive).</p>",
        unsafe_allow_html=True
    )
    
    # REINSTATED: The Matplotlib feature plot, now dynamically colored based on theme
    fig = plot_top_features(tfidf, model, theme=st.session_state.theme)
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("#### 2. Model Evaluation")
    st.markdown(
        f"<p style='color:{secondary_text}; font-size:20px;'>To ensure the most transparent approach, 9 different combinations of text analysis methods "
        "and classifiers were tested. Every single one exceeded the 90% accuracy target.</p>",
        unsafe_allow_html=True
    )

    # REINSTATED: The Model Evaluation Table
    try:
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
    except Exception:
        st.info("Evaluation data will appear here once 'outputs/tableau_model_comparison.csv' is generated.")
        
    st.caption(
        "**Final chosen approach: TF-IDF + Logistic Regression** — "
        "selected not just for accuracy, but because it can explain its reasoning in plain language."
    )

    st.markdown("---")

    # ── Tableau dashboard embed ────────────────────────────────────────────
    st.markdown("### 📊 Full Interactive Dashboard")
    st.markdown(
        f"<p style='color:{secondary_text}; font-size:20px;'>Explore the full data story below — from the growth of podcast content "
        "to the distinct identities of each show. Built with Tableau.</p>",
        unsafe_allow_html=True
    )

    # REINSTATED: The full Tableau embed code with 100% width fix
    tableau_html = """
    <div class='tableauPlaceholder' id='viz1775658569626' style='position: relative; width: 100%;'>
        <noscript><a href='#'><img alt='1 ' src='https://public.tableau.com/static/images/GA/GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft/1_rss.png' style='border: none' /></a></noscript>
        <object class='tableauViz'  style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
            <param name='embed_code_version' value='3' /> 
            <param name='site_root' value='' />
            <param name='name' value='GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/GA/GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-GB' />
            <param name='filter' value='publish=yes' />
        </object>
    </div>
    <script type='text/javascript'>
        var divElement = document.getElementById('viz1775658569626');
        var vizElement = divElement.getElementsByTagName('object');
        if (divElement.offsetWidth > 800) { vizElement.style.width='100%'; vizElement.style.height='2027px'; }
        else if (divElement.offsetWidth > 500) { vizElement.style.width='100%'; vizElement.style.height='2027px'; }
        else { vizElement.style.width='100%'; vizElement.style.height='2727px'; }
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
    
    components.html(tableau_html, height=2050, scrolling=True)