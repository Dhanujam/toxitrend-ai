import streamlit as st
import pickle
import re
import os
import nltk
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from googleapiclient.discovery import build
from supabase import create_client, Client

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ToxiTrend AI",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# NLTK DOWNLOADS
# =====================================================

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# =====================================================
# SUPABASE CONNECTION
# =====================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# =====================================================
# SESSION STATE
# =====================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# =====================================================
# REAL AUTHENTICATION
# =====================================================

if not st.session_state.authenticated:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            "<h1 style='text-align:center;'>🔐 ToxiTrend AI</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center; color:gray;'>Secure AI Moderation Dashboard</p>",
            unsafe_allow_html=True
        )

        auth_choice = st.selectbox(
            "Choose Option",
            ["Login", "Signup"]
        )

        email = st.text_input("Email")

        password = st.text_input(
            "Password",
            type="password"
        )

        # =====================================================
        # SIGNUP
        # =====================================================

        if auth_choice == "Signup":

            if st.button("Create Account"):

                if email.strip() == "" or password.strip() == "":
                    st.warning("Please enter email and password.")

                else:
                    try:

                        with st.spinner("Creating account..."):

                            supabase.auth.sign_up({
                                "email": email,
                                "password": password
                            })

                        st.success("Account created successfully 😎")

                    except Exception as e:
                        st.error(f"Signup Error: {e}")

        # =====================================================
        # LOGIN
        # =====================================================

        elif auth_choice == "Login":

            if st.button("Login"):

                if email.strip() == "" or password.strip() == "":
                    st.warning("Please enter email and password.")

                else:
                    try:

                        with st.spinner("Logging in..."):

                            supabase.auth.sign_in_with_password({
                                "email": email,
                                "password": password
                            })

                        st.session_state.authenticated = True
                        st.session_state.user_email = email

                        st.success("Login successful 😎")

                        st.rerun()

                    except Exception as e:
                        st.error(f"Login Error: {e}")

# =====================================================
# SHOW APP ONLY AFTER LOGIN
# =====================================================

if st.session_state.authenticated:

    st.sidebar.success(f"Logged in as {st.session_state.user_email}")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.rerun()

    # =====================================================
    # SIDEBAR NAVIGATION
    # =====================================================

    menu = st.sidebar.radio(
        "📌 Navigation",
        [
            "Single Comment Analysis",
            "CSV Analysis",
            "Reddit Analysis",
            "YouTube Analysis"
        ]
    )

    # =====================================================
    # CUSTOM CSS
    # =====================================================

    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3 {
        color: white;
    }

    .stTextArea textarea {
        background-color: #262730;
        color: white;
        border-radius: 12px;
        font-size: 16px;
    }

    .stButton button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        height: 3em;
        font-size: 18px;
        border: none;
    }

    .stButton button:hover {
        background-color: #ff1e1e;
        color: white;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # LOAD MODEL AND VECTORIZER
    # =====================================================

    model = pickle.load(open("models/toxicity_model.pkl", "rb"))
    tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

    # =====================================================
    # YOUTUBE API
    # =====================================================

    API_KEY = st.secrets["YOUTUBE_API_KEY"]

    youtube = build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )

    # =====================================================
    # NLP CLEANING
    # =====================================================

    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def analyze_dataframe(df):

        df["cleaned_comment"] = df["comment_text"].apply(clean_text)

        vectors = tfidf.transform(df["cleaned_comment"])

        predictions = model.predict(vectors)
        probabilities = model.predict_proba(vectors)[:, 1]

        df["Prediction"] = predictions

        df["Prediction"] = df["Prediction"].map({
            0: "Non-Toxic",
            1: "Toxic"
        })

        df["Toxicity Score"] = probabilities * 100

        df["Flag Status"] = df["Toxicity Score"].apply(
            lambda score: "🚩 Flagged Comment" if score >= 90 else "Normal"
        )

        return df

    def show_dashboard(df, title):

        st.subheader(title)

        total_comments = len(df)
        toxic_count = len(df[df["Prediction"] == "Toxic"])
        non_toxic_count = len(df[df["Prediction"] == "Non-Toxic"])
        flagged_count = len(df[df["Flag Status"] == "🚩 Flagged Comment"])

        toxicity_percentage = (
            toxic_count / total_comments
        ) * 100 if total_comments > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("📄 Total Comments", total_comments)
        col2.metric("🚨 Toxic", toxic_count)
        col3.metric("✅ Non-Toxic", non_toxic_count)
        col4.metric("🚩 Flagged", flagged_count)
        col5.metric("📈 Toxicity %", f"{toxicity_percentage:.2f}%")

        st.subheader("📋 Prediction Results")

        st.dataframe(
            df[
                [
                    "comment_text",
                    "Prediction",
                    "Toxicity Score",
                    "Flag Status"
                ]
            ]
        )

        counts = df["Prediction"].value_counts()

        chart_df = pd.DataFrame({
            "Category": counts.index,
            "Count": counts.values
        })

        fig = px.bar(
            chart_df,
            x="Category",
            y="Count",
            title="Toxicity Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Trending Toxic Keywords")

        toxic_comments = df[df["Prediction"] == "Toxic"]

        toxic_text = " ".join(
            toxic_comments["cleaned_comment"]
        )

        toxic_words = toxic_text.split()

        word_counts = Counter(toxic_words)

        top_words = word_counts.most_common(10)

        trend_df = pd.DataFrame(
            top_words,
            columns=["Word", "Frequency"]
        )

        if not trend_df.empty:

            st.dataframe(trend_df)

            fig_trend = px.bar(
                trend_df,
                x="Word",
                y="Frequency",
                title="Top Trending Toxic Words"
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True
            )

    # =====================================================
    # TITLE
    # =====================================================

    st.markdown(
        "<h1 style='text-align:center;'>🤖 ToxiTrend AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:gray;'>AI-Powered Toxicity Detection Dashboard</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # SINGLE COMMENT ANALYSIS
    # =====================================================

    if menu == "Single Comment Analysis":

        st.subheader("📝 Single Comment Analysis")

        user_input = st.text_area(
            "Enter a comment",
            height=150
        )

        analyze_comment = st.button("Analyze Comment")

        if analyze_comment:

            with st.spinner("Analyzing comment..."):

                if user_input.strip() != "":

                    cleaned = clean_text(user_input)

                    vectorized = tfidf.transform([cleaned])

                    prediction = model.predict(vectorized)

                    probability = model.predict_proba(vectorized)[0][1]

                    toxicity_score = probability * 100

                    if prediction[0] == 1:

                        if toxicity_score >= 90:

                            st.markdown(
                                f"""
                                <div class='result-box' style='background-color:#b00020;'>
                                🚩 Flagged Toxic Comment<br><br>
                                Toxicity Score: {toxicity_score:.2f}%
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                f"""
                                <div class='result-box' style='background-color:#ff4b4b;'>
                                🚨 Toxic Comment Detected<br><br>
                                Toxicity Score: {toxicity_score:.2f}%
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    else:

                        st.markdown(
                            f"""
                            <div class='result-box' style='background-color:#00C853;'>
                            ✅ Non-Toxic Comment<br><br>
                            Toxicity Score: {toxicity_score:.2f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # =====================================================
    # CSV ANALYSIS
    # =====================================================

    if menu == "CSV Analysis":

        st.subheader("📁 CSV Toxicity Analysis")

        uploaded_file = st.file_uploader(
            "Upload CSV file with column name 'comment_text'",
            type=["csv"]
        )

        csv_button = st.button("Analyze CSV File")

        if uploaded_file is not None and csv_button:

            with st.spinner("Analyzing CSV file..."):

                df_upload = pd.read_csv(uploaded_file)

                if "comment_text" in df_upload.columns:

                    df_upload = analyze_dataframe(df_upload)

                    show_dashboard(
                        df_upload,
                        "📊 CSV Dashboard Metrics"
                    )

                else:

                    st.error(
                        "CSV must contain a column named 'comment_text'"
                    )

    # =====================================================
    # REDDIT ANALYSIS
    # =====================================================

    if menu == "Reddit Analysis":

        st.subheader("🌍 Manual Reddit Comment Analyzer")

        reddit_input = st.text_area(
            "Paste Reddit comments here",
            height=250
        )

        reddit_button = st.button("Analyze Reddit Comments")

        if reddit_button:

            with st.spinner("Analyzing Reddit comments..."):

                reddit_comments = [
                    comment.strip()
                    for comment in reddit_input.split("\n")
                    if comment.strip() != ""
                ]

                reddit_df = pd.DataFrame({
                    "comment_text": reddit_comments
                })

                reddit_df = analyze_dataframe(reddit_df)

                show_dashboard(
                    reddit_df,
                    "📊 Reddit Dashboard"
                )

    # =====================================================
    # YOUTUBE ANALYSIS
    # =====================================================

    if menu == "YouTube Analysis":

        st.subheader("🎥 YouTube Comment Analyzer")

        video_id = st.text_input(
            "Enter YouTube Video ID"
        )

        youtube_button = st.button(
            "Analyze YouTube Comments"
        )

        if youtube_button:

            with st.spinner("Fetching YouTube comments..."):

                try:

                    request = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        textFormat="plainText"
                    )

                    response = request.execute()

                    comments = []

                    for item in response.get("items", []):

                        comment = item[
                            "snippet"
                        ][
                            "topLevelComment"
                        ][
                            "snippet"
                        ][
                            "textDisplay"
                        ]

                        comments.append(comment)

                    youtube_df = pd.DataFrame({
                        "comment_text": comments
                    })

                    youtube_df = analyze_dataframe(youtube_df)

                    show_dashboard(
                        youtube_df,
                        "📊 YouTube Dashboard"
                    )

                except Exception as e:

                    st.error(f"Error: {e}")

    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.markdown(
        """
        <p style='text-align:center; color:gray;'>
        Made by Dhanuja | ToxiTrend AI
        </p>
        """,
        unsafe_allow_html=True
    )