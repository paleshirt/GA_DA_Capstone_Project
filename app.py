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

tfidf, model = load_model()
raw_df, clean_df = load_data()

# ── Navigation ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Find My Podcast", "📖 About the Podcasts", "📊 Data Insights"])

# ── TAB 1: RECOMMENDER ─────────────────────────────────────────────────────
with tab1:
    st.title("🎙️ Better Questions")
    st.markdown("""
    ### Find your next stepping stone for self-healing.
    *Type a topic or feeling below. Our engine scans the actual conversations and themes of hundreds of episodes 
    to find your best match, rather than just relying on broad category tags.*
    """)
    st.write("---")

    # Using st.form ensures that pressing "Enter" triggers the submit button natively on all OS
    with st.form(key="search_form"):
        user_input = st.text_input(
            "What would you like to explore today?",
            placeholder="e.g., How to manage anxiety, or social inequality in Singapore..."
        )
        submit_button = st.form_submit_button("🔍 Find My Podcast")

    # Example inputs outside the form
    st.markdown("**Or try an example:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1: ex1 = st.button("💡 I feel like I'm falling behind my peers", use_container_width=True)
    with col_ex2: ex2 = st.button("💡 Why is housing so expensive now?", use_container_width=True)
    with col_ex3: ex3 = st.button("💡 I want to communicate better in relationships", use_container_width=True)

    # Determine which query to run
    active_query = ""
    if submit_button and user_input.strip():
        active_query = user_input
    elif ex1: active_query = "I feel like I'm falling behind my peers"
    elif ex2: active_query = "Why is housing so expensive now?"
    elif ex3: active_query = "I want to communicate better in relationships"
    elif submit_button and not user_input.strip():
        st.warning("Please enter a topic to get a recommendation!")

    if active_query:
        st.markdown(f"**Searching for:** *{active_query}*")
        st.write("---")
        
        # 1. Vectorize input
        input_vec = tfidf.transform([active_query])
        
        # 2. Predict podcast
        pred_idx = model.predict(input_vec)
        podcasts = model.classes_
        final_podcast = podcasts[pred_idx]
        
        # 3. Find top keywords driving the decision
        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_
        if final_podcast == podcasts:
            coefs = -coefs
            
        nonzero_idx = input_vec.nonzero()
        word_scores = [(feature_names[i], coefs[i]) for i in nonzero_idx]
        word_scores.sort(key=lambda x: x, reverse=True)
        top_words = [w for w in word_scores[:3] if w > 0]
        
        # --- UI OUTPUT ---
        st.subheader("🎯 Your Recommendation:")
        st.header(final_podcast)

        if top_words:
            badges = " ".join([f"<span style='background-color:#d1fae5; color:#065f46; padding:4px 8px; border-radius:12px; font-size:14px; margin-right:5px;'>{word}</span>" for word in top_words])
            st.markdown(f"**Key themes detected:** {badges}", unsafe_allow_html=True)
            st.write("") # spacing

        # Spotify Embed
        st.markdown("##### 🎧 Listen to the Show on Spotify")
        if final_podcast == "Hidden Brain":
            spotify_url = "https://open.spotify.com/embed/show/20AdxCWTik21B3s27K2s8Z"
        else:
            spotify_url = "https://open.spotify.com/embed/show/7BvA1Kq2uV82y3Vtc3jGjQ"
            
        components.html(
            f'<iframe style="border-radius:12px" src="{spotify_url}" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=160
        )

# ── TAB 2: ABOUT THE PODCASTS ──────────────────────────────────────────────
with tab2:
    st.title("📖 Podcast Context")
    st.write("Learn more about the distinct personalities of our two featured shows.")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("CNA Deep Dive")
        st.write("""
        **Host:** Steven Chia & Tiffany Ang  
        **Focus:** Current affairs, public policy, and societal trends in Singapore and Asia.
        
        **Why this matters:** CNA Deep Dive tackles the structural side of the human experience. 
        It focuses on the 'how' and 'why' of the world around us, looking at policy, economics, 
        and national conversations. It's the perfect resource for understanding societal context.
        """)
        components.html(
            '<iframe style="border-radius:12px" src="https://open.spotify.com/embed/show/7BvA1Kq2uV82y3Vtc3jGjQ" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=160
        )
        
    with col2:
        st.header("Hidden Brain")
        st.write("""
        **Host:** Shankar Vedantam  
        **Focus:** Psychology, human behavior, and the unconscious patterns that drive us.
        
        **Why this matters:** Hidden Brain tackles the personal side of the human experience. 
        It explores the inner workings of the mind, helping listeners reflect on their own habits, 
        emotions, and relationships. It is a powerful tool for self-understanding and healing.
        """)
        components.html(
            '<iframe style="border-radius:12px" src="https://open.spotify.com/embed/show/20AdxCWTik21B3s27K2s8Z" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=160
        )

# ── TAB 3: DATA INSIGHTS ───────────────────────────────────────────────────
with tab3:
    st.title("⚙️ Model Engine & Analytics")
    st.info(
        "This section is for those curious about how the recommendation engine works. "
        "It details the model performance and the data story behind the scenes."
    )
    
    st.markdown("### 🏆 How Accurate is the Recommender?")
    st.markdown(
        "To find the most reliable approach, 9 different combinations of text analysis methods "
        "and classifiers were tested. Every single one exceeded the 90% accuracy target."
    )

    model_df = pd.read_csv("outputs/tableau_model_comparison.csv")
    model_df["Accuracy %"]  = (model_df["Accuracy"] * 100).round(2)
    model_df["Balanced Score %"] = (model_df["F1"] * 100).round(2)
    model_df["✅ Above 90%"] = model_df["Accuracy %"].apply(lambda x: "✅" if x >= 90 else "❌")
    st.dataframe(
        model_df[["Method", "Model", "Accuracy %", "Balanced Score %", "✅ Above 90%"]]
        .sort_values("Accuracy %", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )
    st.caption(
        "**Final chosen approach: TF-IDF + Logistic Regression** — "
        "selected not just for its high accuracy, but because it is an 'explainable' model, "
        "allowing us to show the exact key phrases driving your recommendation."
    )

    st.markdown("---")

    # ── Tableau dashboard embed ────────────────────────────────────────────
    st.markdown("### 📊 Full Interactive Dashboard")
    st.markdown(
        "Explore the full data story below — from the growth of podcast content "
        "to the distinct identities of each show. Built with Tableau."
    )

    # Updated URL with parameters to force desktop layout and remove toolbars 
    # to fix the white background and legend spacing issues.
    tableau_url = (
        "https://public.tableau.com/views/"
        "GADACapstoneSocialIssuesPodcastRecommenderDashboard/FinalDraft"
        "?:showVizHome=no&:embed=true&:device=desktop&:toolbar=no"
    )
    
    # Using raw HTML iframe gives us more robust control over the width boundaries
    html_code = f"""
    <iframe src="{tableau_url}" width="100%" height="2050" style="border:none;" scrolling="no"></iframe>
    """
    components.html(html_code, height=2100)