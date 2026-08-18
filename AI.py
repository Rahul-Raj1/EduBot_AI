import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .welcome-box {
        padding: 22px;
        border-radius: 15px;
        background-color: #f5f7fb;
        margin-bottom: 20px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SQLITE DATABASE CONFIGURATION
# ============================================================

DB_NAME = "chat_history.db"


# ============================================================
# SQLITE DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )


# ============================================================
# CREATE CHAT HISTORY TABLE
# ============================================================

def create_chat_table():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            question TEXT NOT NULL,

            answer TEXT NOT NULL,

            similarity_score REAL,

            matched_question TEXT,

            matched INTEGER,

            timestamp TEXT NOT NULL

        )
        """
    )

    conn.commit()

    conn.close()


# Create database and table automatically
create_chat_table()


# ============================================================
# SAVE CHAT TO SQLITE
# ============================================================

def save_chat(
    question,
    answer,
    similarity_score,
    matched_question,
    matched
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history
        (
            question,
            answer,
            similarity_score,
            matched_question,
            matched,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            answer,
            similarity_score,
            matched_question,
            int(matched),
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# LOAD CHAT HISTORY FROM SQLITE
# ============================================================

def load_chat_history():

    conn = get_db_connection()

    query = """
        SELECT
            id,
            question,
            answer,
            similarity_score,
            matched_question,
            matched,
            timestamp
        FROM chat_history
        ORDER BY id ASC
    """

    history_df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return history_df


# ============================================================
# CLEAR CHAT HISTORY FROM SQLITE
# ============================================================

def clear_chat_history():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM chat_history"
    )

    # Reset auto-increment ID
    cursor.execute(
        "DELETE FROM sqlite_sequence "
        "WHERE name='chat_history'"
    )

    conn.commit()

    conn.close()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "knowledge_base.csv"
    )

    # Keep only required columns
    df = df[
        ["question", "answer"]
    ].copy()

    # Convert text columns to string
    df["question"] = (
        df["question"]
        .astype(str)
    )

    df["answer"] = (
        df["answer"]
        .astype(str)
    )

    # Remove leading/trailing spaces
    df["question"] = (
        df["question"]
        .str.strip()
    )

    df["answer"] = (
        df["answer"]
        .str.strip()
    )

    # Replace multiple spaces with one space
    df["question"] = df[
        "question"
    ].str.replace(
        r"\s+",
        " ",
        regex=True
    )

    df["answer"] = df[
        "answer"
    ].str.replace(
        r"\s+",
        " ",
        regex=True
    )

    # Remove duplicate question-answer pairs
    df = df.drop_duplicates(
        subset=[
            "question",
            "answer"
        ]
    ).reset_index(drop=True)

    # Remove empty questions/answers
    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ].reset_index(drop=True)

    return df


# ============================================================
# LOAD SENTENCE TRANSFORMER + CREATE EMBEDDINGS
# ============================================================

@st.cache_resource
def load_model_and_embeddings(df):

    # Load pretrained Sentence Transformer
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Generate embeddings for all questions
    question_embeddings = model.encode(
        df["question"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return model, question_embeddings


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

try:

    df = load_data()

    with st.spinner(
        "🧠 Loading EduBot AI..."
    ):

        model, question_embeddings = (
            load_model_and_embeddings(df)
        )

except Exception as e:

    st.error(
        "❌ Unable to load the chatbot."
    )

    st.error(
        str(e)
    )

    st.stop()


# ============================================================
# CHATBOT FUNCTION
# ============================================================

def get_answer(
    user_question,
    threshold=0.55
):

    # Remove extra spaces
    user_question = (
        user_question.strip()
    )

    # Handle empty input
    if not user_question:

        return {
            "answer": (
                "Please enter a question."
            ),
            "matched_question": None,
            "score": 0.0,
            "matched": False
        }

    # --------------------------------------------------------
    # CONVERT USER QUESTION INTO EMBEDDING
    # --------------------------------------------------------

    query_embedding = model.encode(
        user_question,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        [query_embedding],
        question_embeddings
    )[0]

    # --------------------------------------------------------
    # FIND HIGHEST SIMILARITY
    # --------------------------------------------------------

    best_index = int(
        np.argmax(
            similarity_scores
        )
    )

    # Get highest similarity score
    best_score = float(
        similarity_scores[
            best_index
        ]
    )

    # --------------------------------------------------------
    # GET MATCHING QUESTION
    # --------------------------------------------------------

    best_question = df.iloc[
        best_index
    ]["question"]

    # --------------------------------------------------------
    # GET CORRESPONDING ANSWER
    # --------------------------------------------------------

    best_answer = df.iloc[
        best_index
    ]["answer"]

    # --------------------------------------------------------
    # CHECK SIMILARITY THRESHOLD
    # --------------------------------------------------------

    if best_score >= threshold:

        return {
            "answer": best_answer,
            "matched_question": best_question,
            "score": best_score,
            "matched": True
        }

    else:

        return {
            "answer": (
                "Sorry, I couldn't find a relevant "
                "answer in my educational knowledge base. "
                "Please try asking about AI, Machine "
                "Learning, Deep Learning, Python, Data "
                "Science, Statistics, NLP, Computer "
                "Vision, SQL, or Generative AI."
            ),
            "matched_question": best_question,
            "score": best_score,
            "matched": False
        }


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    # Load permanent history from SQLite
    history_df = load_chat_history()

    st.session_state.messages = []

    # --------------------------------------------------------
    # LOAD OLD CHAT HISTORY
    # --------------------------------------------------------

    if not history_df.empty:

        for _, row in history_df.iterrows():

            # User message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": row["question"]
                }
            )

            # Assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": row["answer"],
                    "score": row[
                        "similarity_score"
                    ],
                    "matched": bool(
                        row["matched"]
                    ),
                    "matched_question": row[
                        "matched_question"
                    ]
                }
            )


# ============================================================
# HEADER
# ============================================================

header_col, history_col = st.columns(
    [7, 2]
)


# ============================================================
# TITLE
# ============================================================

with header_col:

    st.markdown(
        '<div class="main-title">'
        '🤖 EduBot AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your AI-powered educational assistant'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# CHAT HISTORY POPUP BUTTON
# ============================================================

with history_col:

    with st.popover(
        "📜 Chat History",
        use_container_width=True
    ):

        st.subheader(
            "📜 Previous Conversations"
        )

        # Load latest history directly from SQLite
        history_df = load_chat_history()

        # ----------------------------------------------------
        # CHECK HISTORY
        # ----------------------------------------------------

        if history_df.empty:

            st.info(
                "No previous conversations found."
            )

        else:

            # Create display dataframe
            display_df = history_df.copy()

            # Rename columns
            display_df = display_df.rename(
                columns={
                    "id": "ID",
                    "question": "Question",
                    "answer": "Answer",
                    "similarity_score":
                        "Similarity",
                    "matched_question":
                        "Matched Question",
                    "matched":
                        "Matched",
                    "timestamp":
                        "Date & Time"
                }
            )

            # Format similarity score
            display_df[
                "Similarity"
            ] = display_df[
                "Similarity"
            ].apply(
                lambda x:
                f"{x:.2%}"
                if pd.notna(x)
                else "N/A"
            )

            # Format matched value
            display_df[
                "Matched"
            ] = display_df[
                "Matched"
            ].apply(
                lambda x:
                "✅ Yes"
                if x == 1
                else "❌ No"
            )

            # Show newest first
            display_df = display_df.iloc[
                ::-1
            ]

            # ------------------------------------------------
            # DISPLAY TABLE
            # ------------------------------------------------

            st.dataframe(
                display_df[
                    [
                        "ID",
                        "Question",
                        "Answer",
                        "Similarity",
                        "Date & Time"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"📊 Total conversations: "
                f"{len(history_df)}"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ EduBot Settings"
    )

    st.write(
        "EduBot uses a pretrained Sentence Transformer "
        "to understand the semantic meaning of your "
        "question and retrieve the most relevant answer."
    )

    st.divider()


    # ========================================================
    # KNOWLEDGE BASE INFORMATION
    # ========================================================

    st.subheader(
        "📚 Knowledge Base"
    )

    st.metric(
        "Questions",
        len(df)
    )

    st.caption(
        "Model: all-MiniLM-L6-v2"
    )

    st.divider()


    # ========================================================
    # DATABASE INFORMATION
    # ========================================================

    st.subheader(
        "💾 Chat Database"
    )

    history_df = load_chat_history()

    st.metric(
        "Saved Conversations",
        len(history_df)
    )

    st.caption(
        "Storage: SQLite"
    )

    st.divider()


    # ========================================================
    # SUGGESTED QUESTIONS
    # ========================================================

    st.subheader(
        "💡 Try These Questions"
    )

    suggested_questions = [

        "What is Artificial Intelligence?",

        "What is Machine Learning?",

        "What is Deep Learning?",

        "What is Python?",

        "What is NLP?",

        "What is Computer Vision?",

        "What is SQL?",

        "What is Generative AI?"
    ]

    for question in suggested_questions:

        if st.button(
            question,
            use_container_width=True
        ):

            st.session_state.selected_question = (
                question
            )


    st.divider()


    # ========================================================
    # SIMILARITY THRESHOLD
    # ========================================================

    st.subheader(
        "🎯 Similarity Threshold"
    )

    threshold = st.slider(
        "Threshold",

        min_value=0.30,

        max_value=0.90,

        value=0.55,

        step=0.05,

        help=(
            "Higher values make the chatbot "
            "more strict when deciding whether "
            "a question is relevant."
        )
    )


    st.divider()


    # ========================================================
    # CLEAR CHAT HISTORY
    # ========================================================

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):

        # Delete history from SQLite
        clear_chat_history()

        # Clear session history
        st.session_state.messages = []

        # Remove selected question if present
        if (
            "selected_question"
            in st.session_state
        ):

            del st.session_state[
                "selected_question"
            ]

        st.rerun()


# ============================================================
# WELCOME BOX
# ============================================================

if len(
    st.session_state.messages
) == 0:

    st.markdown(
        """
        <div class="welcome-box">

        ### 🎓 Welcome to EduBot AI!

        Ask me anything related to the topics in my
        educational knowledge base.

        **Example:**

        > Can you explain machine learning?

        I'll find the most semantically similar question
        and return its corresponding answer.

        **Topics include:**

        - Artificial Intelligence
        - Machine Learning
        - Deep Learning
        - Python
        - Data Science
        - Statistics
        - NLP
        - Computer Vision
        - SQL
        - Generative AI

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # SHOW RETRIEVAL INFORMATION
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and "score" in message
        ):

            score = message["score"]

            if message.get(
                "matched",
                False
            ):

                st.caption(
                    f"🔎 Semantic similarity: "
                    f"{score:.2%}"
                )

            else:

                st.caption(
                    f"⚠️ Similarity: "
                    f"{score:.2%} — "
                    f"Fallback response"
                )


# ============================================================
# USER INPUT
# ============================================================

user_question = st.chat_input(
    "💬 Ask an educational question..."
)


# ============================================================
# HANDLE SUGGESTED QUESTION
# ============================================================

if (
    "selected_question"
    in st.session_state
    and not user_question
):

    user_question = (
        st.session_state.selected_question
    )

    del st.session_state.selected_question


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_question:

    # Remove unnecessary spaces
    user_question = (
        user_question.strip()
    )

    # --------------------------------------------------------
    # ADD USER MESSAGE TO SESSION STATE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )


    # --------------------------------------------------------
    # GENERATE BOT RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔍 Searching the knowledge base..."
        ):

            result = get_answer(
                user_question,
                threshold
            )


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.markdown(
            result["answer"]
        )


        # ----------------------------------------------------
        # DISPLAY SIMILARITY SCORE
        # ----------------------------------------------------

        if result["matched"]:

            st.caption(
                f"🔎 Semantic similarity: "
                f"{result['score']:.2%}"
            )

        else:

            st.caption(
                f"⚠️ Similarity: "
                f"{result['score']:.2%} — "
                f"Fallback response"
            )


        # ----------------------------------------------------
        # RETRIEVAL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Retrieval Details"
        ):

            if result[
                "matched_question"
            ]:

                st.write(
                    "**Best Matching Question:**"
                )

                st.write(
                    result[
                        "matched_question"
                    ]
                )

            st.write(
                f"**Similarity Score:** "
                f"{result['score']:.4f}"
            )

            st.write(
                f"**Threshold:** "
                f"{threshold:.2f}"
            )


    # --------------------------------------------------------
    # SAVE QUESTION + ANSWER TO SQLITE
    # --------------------------------------------------------

    save_chat(
        question=user_question,
        answer=result["answer"],
        similarity_score=result["score"],
        matched_question=(
            result["matched_question"]
        ),
        matched=result["matched"]
    )


    # --------------------------------------------------------
    # ADD BOT RESPONSE TO SESSION STATE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "score": result["score"],
            "matched": result["matched"],
            "matched_question": (
                result["matched_question"]
            )
        }
    )