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

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top, #111827 0%, #05070d 45%, #02040a 100%);
        color: white;
    }

    .stButton button {
        background: linear-gradient(135deg, #ff4b4b, #ff1e1e);
        color: white;
        border-radius: 14px;
        border: none;
        height: 3.2em;
        font-weight: 800;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.25);
    }

    .stTextInput input {
        background-color: #151822 !important;
        color: white !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    </style>
    """, unsafe_allow_html=True)

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

                        st.success("Account created successfully. Now login 😎")

                    except Exception as e:
                        st.error(f"Signup Error: {e}")

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
                        st.rerun()

                    except Exception as e:
                        st.error(f"Login Error: {e}")

# =====================================================
# SHOW APP ONLY AFTER LOGIN
# =====================================================

if st.session_state.authenticated:

    # =====================================================
    # PREMIUM DARK UI CSS
    # =====================================================

    st.markdown("""
    <style>

    .stApp {
        background: radial-gradient(circle at top, #111827 0%, #05070d 45%, #02040a 100%);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b14 0%, #0b1020 100%);
        border-right: 1px solid rgba(255, 75, 75, 0.25);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }

    .sidebar-logo {
        padding: 22px;
        border-bottom: 1px solid rgba(255, 75, 75, 0.5);
        margin-bottom: 25px;
    }

    .brand-title {
        font-size: 30px;
        font-weight: 900;
        color: white;
    }

    .brand-title span {
        color: #ff4b4b;
    }

    .brand-subtitle {
        color: #9ca3af;
        font-size: 14px;
    }

    .user-card {
        background: rgba(255, 75, 75, 0.08);
        border: 1px solid rgba(255, 75, 75, 0.35);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(255, 75, 75, 0.08);
    }

    .user-email {
        color: #ff5b5b;
        font-weight: 700;
        word-break: break-word;
    }

    .auth-badge {
        display: inline-block;
        background: rgba(0, 200, 83, 0.15);
        color: #00c853;
        border: 1px solid rgba(0, 200, 83, 0.4);
        padding: 7px 14px;
        border-radius: 10px;
        font-weight: 700;
        margin-top: 12px;
    }

    .stButton button {
        background: linear-gradient(135deg, #ff4b4b, #ff1e1e);
        color: white;
        border-radius: 14px;
        border: none;
        height: 3.2em;
        font-weight: 800;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.25);
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #ff1e1e, #b00020);
        color: white;
        transform: scale(1.01);
    }

    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        padding: 14px 16px;
        border-radius: 14px;
        margin-bottom: 10px;
        transition: 0.2s;
    }

    div[role="radiogroup"] label:hover {
        border-color: #ff4b4b;
        background: rgba(255, 75, 75, 0.08);
    }

    .main-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 0 35px rgba(0,0,0,0.35);
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 36px;
        font-weight: 900;
        color: #ffffff;
    }

    .section-subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 20px;
    }

    .stTextArea textarea,
    .stTextInput input {
        background-color: #151822 !important;
        color: white !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    .result-box {
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 0 25px rgba(255, 75, 75, 0.15);
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 25px;
    }

    .footer span {
        color: #ff4b4b;
        font-weight: 800;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # PREMIUM SIDEBAR
    # =====================================================

    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="brand-title">🤖 ToxiTrend <span>AI</span></div>
        <div class="brand-subtitle">AI-Powered Moderation</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(
        f"""
        <div class="user-card">
            <p>Welcome back,</p>
            <div class="user-email">{st.session_state.user_email}</div>
            <div class="auth-badge">🛡️ Authenticated</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button("↪ Logout"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.rerun()

    st.sidebar.markdown("### 📌 NAVIGATION")

    menu = st.sidebar.radio(
        "",
        [
            "💬 Single Comment Analysis",
            "📊 CSV Analysis",
            "🌍 Reddit Analysis",
            "🎥 YouTube Analysis"
        ]
    )

    st.sidebar.markdown("""
    <div class="user-card">
        <h3>🛡️ ToxiTrend AI</h3>
        <p style="color:#9ca3af;">Detect. Analyze. Protect.</p>
        <p style="color:#9ca3af;">Building a safer digital community.</p>
    </div>

    <div class="footer">
        Made with ❤️ by <span>Dhanuja</span>
    </div>
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

        else:
            st.warning("No toxic keywords found.")

        st.subheader("🚩 Flagged Comments for Moderator Review")

        flagged_df = df[df["Flag Status"] == "🚩 Flagged Comment"]

        if not flagged_df.empty:
            st.dataframe(
                flagged_df[
                    [
                        "comment_text",
                        "Toxicity Score",
                        "Flag Status"
                    ]
                ]
            )
        else:
            st.success("No comments crossed the 90% flagging threshold.")

    # =====================================================
    # MAIN TITLE
    # =====================================================

    st.markdown(
        "<h1 style='text-align:center;'>🤖 ToxiTrend AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:gray;'>AI-Powered Toxicity Detection, Flagging & Trend Analysis Dashboard</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # SINGLE COMMENT ANALYSIS
    # =====================================================

    if menu == "💬 Single Comment Analysis":

        st.markdown("""
        <div class="main-card">
            <div class="section-title">💬 Single Comment Analysis</div>
            <div class="section-subtitle">Analyze the toxicity of any comment in real-time</div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Enter a comment",
            height=150,
            placeholder="Type or paste your comment here..."
        )

        analyze_comment = st.button("🔍 Analyze Comment")

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
                                Toxicity Score: {toxicity_score:.2f}%<br>
                                Status: Needs Moderator Review
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                f"""
                                <div class='result-box' style='background-color:#ff4b4b;'>
                                🚨 Toxic Comment Detected<br><br>
                                Toxicity Score: {toxicity_score:.2f}%<br>
                                Status: Toxic but Not Flagged
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    else:

                        st.markdown(
                            f"""
                            <div class='result-box' style='background-color:#00C853;'>
                            ✅ Non-Toxic Comment<br><br>
                            Toxicity Score: {toxicity_score:.2f}%<br>
                            Status: Safe Comment
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    chart_data = pd.DataFrame({
                        "Category": ["Non-Toxic", "Toxic"],
                        "Score": [1 - probability, probability]
                    })

                    fig = px.bar(
                        chart_data,
                        x="Category",
                        y="Score",
                        title="Toxicity Analysis"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("☁️ Word Cloud")

                    if cleaned.strip() != "":
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color="black"
                        ).generate(cleaned)

                        fig_wc, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig_wc)

                else:
                    st.warning("Please enter a comment.")

    # =====================================================
    # CSV ANALYSIS
    # =====================================================

    if menu == "📊 CSV Analysis":

        st.markdown("""
        <div class="main-card">
            <div class="section-title">📊 CSV Toxicity Analysis</div>
            <div class="section-subtitle">Upload a CSV file and analyze multiple comments at once</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload CSV file with column name 'comment_text'",
            type=["csv"]
        )

        csv_button = st.button("📊 Analyze CSV File")

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

    if menu == "🌍 Reddit Analysis":

        st.markdown("""
        <div class="main-card">
            <div class="section-title">🌍 Reddit Comment Analyzer</div>
            <div class="section-subtitle">Paste Reddit-style comments and detect toxic patterns</div>
        </div>
        """, unsafe_allow_html=True)

        reddit_input = st.text_area(
            "Paste Reddit comments here, one comment per line",
            height=250,
            placeholder="Paste comments here..."
        )

        reddit_button = st.button("🌍 Analyze Reddit Comments")

        if reddit_button:

            with st.spinner("Analyzing Reddit comments..."):

                reddit_comments = [
                    comment.strip()
                    for comment in reddit_input.split("\n")
                    if comment.strip() != ""
                ]

                if len(reddit_comments) > 0:

                    reddit_df = pd.DataFrame({
                        "comment_text": reddit_comments
                    })

                    reddit_df = analyze_dataframe(reddit_df)

                    show_dashboard(
                        reddit_df,
                        "📊 Reddit Dashboard"
                    )

                else:
                    st.warning("Please paste at least one comment.")

    # =====================================================
    # YOUTUBE ANALYSIS
    # =====================================================

    if menu == "🎥 YouTube Analysis":

        st.markdown("""
        <div class="main-card">
            <div class="section-title">🎥 YouTube Comment Analyzer</div>
            <div class="section-subtitle">Fetch live YouTube comments and analyze toxicity</div>
        </div>
        """, unsafe_allow_html=True)

        video_id = st.text_input(
            "Enter YouTube Video ID",
            placeholder="Example: dQw4w9WgXcQ"
        )

        youtube_button = st.button(
            "🎥 Analyze YouTube Comments"
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

                    if len(comments) > 0:

                        youtube_df = pd.DataFrame({
                            "comment_text": comments
                        })

                        youtube_df = analyze_dataframe(youtube_df)

                        show_dashboard(
                            youtube_df,
                            "📊 YouTube Dashboard"
                        )

                    else:
                        st.warning("No comments found for this video.")

                except Exception as e:

                    st.error(f"Error: {e}")

    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.markdown(
        """
        <p style='text-align:center; color:gray;'>
        <span style='color:#ff4b4b; font-weight:800;'>ToxiTrend AI</span> | Made with ❤️ by Dhanuja
        </p>
        """,
        unsafe_allow_html=True
    )