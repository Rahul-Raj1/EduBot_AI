import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
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

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SQLITE DATABASE
# ============================================================

DB_NAME = "chat_history.db"


def get_db_connection():
    return sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )


def create_chat_tables():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            similarity_score REAL,
            matched_question TEXT,
            matched INTEGER,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    conn.close()


create_chat_tables()


# ============================================================
# CREATE NEW CONVERSATION
# ============================================================

def create_new_conversation(title="New Chat"):

    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO conversations
        (title, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        (title, now, now)
    )

    conversation_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return conversation_id


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

def get_conversations():

    conn = get_db_connection()

    conversations = pd.read_sql_query(
        """
        SELECT
            id,
            title,
            created_at,
            updated_at
        FROM conversations
        ORDER BY updated_at DESC, id DESC
        """,
        conn
    )

    conn.close()

    return conversations


# ============================================================
# LOAD ONE CONVERSATION
# ============================================================

def load_conversation(conversation_id):

    conn = get_db_connection()

    history_df = pd.read_sql_query(
        """
        SELECT
            id,
            question,
            answer,
            similarity_score,
            matched_question,
            matched,
            timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        conn,
        params=(conversation_id,)
    )

    conn.close()

    return history_df


# ============================================================
# DELETE ONE CONVERSATION
# ============================================================

def delete_conversation(conversation_id):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM messages WHERE conversation_id = ?",
        (conversation_id,)
    )

    cursor.execute(
        "DELETE FROM conversations WHERE id = ?",
        (conversation_id,)
    )

    conn.commit()
    conn.close()


# ============================================================
# DELETE ALL CONVERSATIONS
# ============================================================

def delete_all_conversations():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM messages"
    )

    cursor.execute(
        "DELETE FROM conversations"
    )

    conn.commit()
    conn.close()


# ============================================================
# SAVE CHAT MESSAGE
# ============================================================

def save_chat(
    conversation_id,
    question,
    answer,
    similarity_score,
    matched_question,
    matched
):

    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO messages
        (
            conversation_id,
            question,
            answer,
            similarity_score,
            matched_question,
            matched,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            question,
            answer,
            similarity_score,
            matched_question,
            int(matched),
            now
        )
    )

    # First question becomes chat title
    current_title = cursor.execute(
        """
        SELECT title
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,)
    ).fetchone()

    if current_title and current_title[0] == "New Chat":

        title = question.strip().replace(
            "\n",
            " "
        )

        if len(title) > 45:
            title = title[:45].rstrip() + "..."

        if not title:
            title = "New Chat"

        cursor.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                now,
                conversation_id
            )
        )

    else:

        cursor.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                conversation_id
            )
        )

    conn.commit()
    conn.close()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "knowledge_datascience.csv"
    )

    df = df[
        ["question", "answer"]
    ].copy()

    df["question"] = (
        df["question"]
        .astype(str)
        .str.strip()
    )

    df["answer"] = (
        df["answer"]
        .astype(str)
        .str.strip()
    )

    df["question"] = (
        df["question"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )

    df["answer"] = (
        df["answer"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )

    df = df.drop_duplicates(
        subset=[
            "question",
            "answer"
        ]
    ).reset_index(drop=True)

    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ].reset_index(drop=True)

    return df


# ============================================================
# LOAD MODEL + EMBEDDINGS
# ============================================================

@st.cache_resource
def load_model_and_embeddings(df):

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    question_embeddings = model.encode(
        df["question"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return model, question_embeddings


# ============================================================
# LOAD DATA + MODEL
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

    st.error(str(e))

    st.stop()


# ============================================================
# CONTEXT MEMORY
# ============================================================

CONTEXT_ENABLED = True


# ============================================================
# TOPIC KEYWORDS
# ============================================================

TOPIC_KEYWORDS = {

    "Python": ["python", "py", "python programming"],
    "SQL": ["sql", "structured query language", "database query"],
    "Machine Learning": ["machine learning", "ml", "machine-learning"],
    "Deep Learning": ["deep learning", "dl", "deep-learning"],
    "Artificial Intelligence": ["artificial intelligence", "ai", "artificial-intelligence"],
    "Data Science": ["data science", "data scientist", "data-science"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv"],
    "Generative AI": ["generative ai", "gen ai", "generative artificial intelligence"],
    "Statistics": ["statistics", "statistical"],
    "Transformer": ["transformer", "transformers", "transformer model"],
    "YOLO": ["yolo", "you only look once"],
    "U-Net": ["u-net", "unet", "u net"],
    "OpenCV": ["opencv", "open cv"]
}


# ============================================================
# FOLLOW-UP INTENTS
# ============================================================

FOLLOW_UP_INTENT_KEYWORDS = {

    "formula": [
        "formula", "formulas", "equation", "equations",
        "calculation", "calculate"
    ],

    "algorithm": [
        "algorithm", "algorithms"
    ],

    "type": [
        "type", "types", "kind", "kinds", "data type", "data types"
    ],

    "example": [
        "example", "examples", "sample", "samples"
    ],

    "application": [
        "application", "applications", "use", "uses", "usage"
    ],

    "advantage": [
        "advantage", "advantages", "benefit", "benefits"
    ],

    "disadvantage": [
        "disadvantage", "disadvantages", "limitation", "limitations"
    ],

    "feature": [
        "feature", "features"
    ],

    "function": [
        "function", "functions"
    ],

    "method": [
        "method", "methods"
    ],

    "difference": [
        "difference", "differences", "differentiate", "compare", "comparison"
    ],

    "working": [
        "working", "works", "work"
    ],

    "architecture": [
        "architecture", "structure"
    ],

    "process": [
        "process", "steps", "step"
    ],

    "definition": [
        "meaning", "definition", "define", "explain"
    ]
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DETECT KNOWN TOPIC
# ============================================================

def detect_explicit_topic(question):

    question_lower = normalize_text(question)

    matches = []

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            keyword = normalize_text(keyword)

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                question_lower
            ):

                matches.append(
                    (len(keyword), topic)
                )

    if matches:
        matches.sort(reverse=True)
        return matches[0][1]

    return None


# ============================================================
# EXTRACT GENERIC SUBJECT FROM PREVIOUS QUESTION
# ============================================================
# IMPORTANT:
# This is the part that fixes questions such as:
#
# What is Precision?
# What is its formula?
#
# Precision is NOT in TOPIC_KEYWORDS, so we cannot depend
# only on TOPIC_KEYWORDS anymore.
# ============================================================

def extract_subject_from_question(question):

    if not question:
        return ""

    q = normalize_text(question)

    patterns = [

        r"^what is (?:an |a |the )?(.+?)(?:\?)?$",

        r"^what are (?:an |a |the )?(.+?)(?:\?)?$",

        r"^define (?:an |a |the )?(.+?)(?:\?)?$",

        r"^explain (?:an |a |the )?(.+?)(?:\?)?$",

        r"^tell me about (?:an |a |the )?(.+?)(?:\?)?$",

        r"^what does (.+?) mean(?:\?)?$"
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            q
        )

        if match:

            subject = match.group(1).strip()

            # Do not treat pronouns as subjects.
            if subject in {
                "it",
                "its",
                "this",
                "that",
                "they",
                "them",
                "their"
            }:
                return ""

            return subject

    return ""


# ============================================================
# GET CURRENT ACTIVE CONTEXT
# ============================================================

def get_active_context():

    messages = st.session_state.get(
        "messages",
        []
    )

    if not messages:

        return {
            "active_topic": None,
            "previous_question": "",
            "previous_answer": "",
            "matched_question": ""
        }

    # Latest assistant answer + the user question before it.
    for i in range(
        len(messages) - 1,
        -1,
        -1
    ):

        if messages[i]["role"] != "assistant":
            continue

        previous_answer = messages[i].get(
            "content",
            ""
        )

        matched_question = messages[i].get(
            "matched_question",
            ""
        )

        previous_question = ""

        for j in range(
            i - 1,
            -1,
            -1
        ):

            if messages[j]["role"] == "user":

                previous_question = messages[j].get(
                    "content",
                    ""
                )

                break

        # First try known broad topic.
        active_topic = detect_explicit_topic(
            matched_question
        )

        if not active_topic:
            active_topic = detect_explicit_topic(
                previous_question
            )

        # CRITICAL FIX:
        # If the subject is not one of the known topics,
        # keep the actual previous subject.
        if not active_topic:

            active_topic = extract_subject_from_question(
                matched_question
            )

        if not active_topic:

            active_topic = extract_subject_from_question(
                previous_question
            )

        # Last fallback: matched question itself.
        if not active_topic and matched_question:

            active_topic = matched_question

        return {
            "active_topic": active_topic,
            "previous_question": previous_question,
            "previous_answer": previous_answer,
            "matched_question": matched_question
        }

    return {
        "active_topic": None,
        "previous_question": "",
        "previous_answer": "",
        "matched_question": ""
    }


# ============================================================
# DETECT FOLLOW-UP INTENT
# ============================================================

def detect_follow_up_intent(question):

    q = normalize_text(
        question
    )

    for intent, keywords in FOLLOW_UP_INTENT_KEYWORDS.items():

        for keyword in sorted(
            keywords,
            key=len,
            reverse=True
        ):

            keyword = normalize_text(keyword)

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                q
            ):
                return intent

    return "general"


# ============================================================
# CHECK FOLLOW-UP QUESTION
# ============================================================

def is_follow_up_question(
    current_question,
    context
):

    current = normalize_text(
        current_question
    )

    if not current:
        return False

    active_topic = context.get(
        "active_topic"
    )

    if not active_topic:
        return False

    # Explicit known topic means a new topic.
    explicit_topic = detect_explicit_topic(
        current
    )

    if explicit_topic:
        return False

    # Direct references are definitely follow-ups.
    reference_words = {
        "it",
        "its",
        "it's",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "above",
        "previous",
        "earlier",
        "same"
    }

    words = current.split()

    if any(
        word in reference_words
        for word in words
    ):
        return True

    # Intent-only questions are follow-ups.
    intent = detect_follow_up_intent(
        current
    )

    if intent != "general":

        # Example:
        # What is the formula?
        # What is the algorithm?
        # Give an example.
        # How does it work?
        if len(words) <= 10:
            return True

    # General short conversational questions.
    follow_up_starts = [
        "what is",
        "what are",
        "why",
        "how",
        "which",
        "where",
        "when",
        "can you",
        "could you",
        "tell me more",
        "explain more",
        "explain further",
        "describe"
    ]

    if len(words) <= 8 and any(
        current.startswith(start)
        for start in follow_up_starts
    ):

        # If it is a generic "what is X?" question,
        # determine whether X is probably a new subject.
        subject = extract_subject_from_question(
            current
        )

        if subject:

            previous_subject = (
                extract_subject_from_question(
                    context.get(
                        "previous_question",
                        ""
                    )
                )
            )

            # If the user says "what is the formula/algorithm",
            # subject is not treated as a new concept.
            if detect_follow_up_intent(current) != "general":
                return True

            # A different concrete subject is a new topic.
            if previous_subject and subject != previous_subject:
                return False

        return True

    return False


# ============================================================
# GET TOPIC CANDIDATES
# ============================================================

def get_topic_candidates(
    active_topic
):

    if not active_topic:
        return []

    topic = normalize_text(
        active_topic
    )

    # Known topic.
    topic_keywords = TOPIC_KEYWORDS.get(
        active_topic,
        []
    )

    if topic_keywords:

        search_terms = [
            normalize_text(x)
            for x in topic_keywords
        ]

    else:

        # Generic subject such as "precision".
        search_terms = [
            topic
        ]

    candidate_indices = []

    for index, question in enumerate(
        df["question"]
    ):

        q = normalize_text(
            question
        )

        if any(
            re.search(
                r"\b" + re.escape(term) + r"\b",
                q
            )
            for term in search_terms
            if term
        ):

            candidate_indices.append(
                index
            )

    return candidate_indices


# ============================================================
# FIND BEST SEMANTIC MATCH
# ============================================================

def find_best_match(
    query,
    candidate_indices=None
):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    if candidate_indices is None:

        candidate_indices = np.arange(
            len(df)
        )

    candidate_indices = np.array(
        candidate_indices,
        dtype=int
    )

    if len(candidate_indices) == 0:
        return None

    candidate_embeddings = (
        question_embeddings[
            candidate_indices
        ]
    )

    similarity_scores = cosine_similarity(
        [query_embedding],
        candidate_embeddings
    )[0]

    best_position = int(
        np.argmax(
            similarity_scores
        )
    )

    best_index = int(
        candidate_indices[
            best_position
        ]
    )

    return {
        "index": best_index,
        "score": float(
            similarity_scores[
                best_position
            ]
        ),
        "question": df.iloc[
            best_index
        ]["question"],
        "answer": df.iloc[
            best_index
        ]["answer"]
    }


# ============================================================
# FIND CONTEXTUAL MATCH
# ============================================================
# This function does TWO searches:
#
# 1. Context search:
#    previous topic + current follow-up
#
# 2. Direct search:
#    current question alone
#
# Then it uses lexical intent matching to prefer a dataset
# question about "formula", "algorithm", "example", etc.
#
# This prevents:
# "What is Precision?"
# "What is its formula?"
#
# from returning the original definition of Precision.
# ============================================================

def find_contextual_match(
    active_topic,
    previous_question,
    previous_answer,
    current_question
):

    intent = detect_follow_up_intent(
        current_question
    )

    # Keep the previous question because it contains
    # the actual concept even when that concept is not
    # in TOPIC_KEYWORDS.
    context_query_parts = [
        str(active_topic),
        previous_question,
        current_question
    ]

    if intent != "general":
        context_query_parts.append(
            intent
        )

    context_query = " ".join(
        x for x in context_query_parts
        if x
    ).strip()

    # Search active-topic candidates first.
    candidate_indices = get_topic_candidates(
        active_topic
    )

    # Get more than one candidate so we can rerank them.
    if candidate_indices:

        query_embedding = model.encode(
            context_query,
            normalize_embeddings=True
        )

        candidate_embeddings = (
            question_embeddings[
                np.array(
                    candidate_indices,
                    dtype=int
                )
            ]
        )

        scores = cosine_similarity(
            [query_embedding],
            candidate_embeddings
        )[0]

        # Top semantic candidates.
        top_positions = np.argsort(
            scores
        )[::-1][:20]

        best = None
        best_rank_score = -999.0

        current_words = set(
            normalize_text(
                current_question
            ).split()
        )

        intent_words = set(
            normalize_text(
                " ".join(
                    FOLLOW_UP_INTENT_KEYWORDS.get(
                        intent,
                        []
                    )
                )
            ).split()
        )

        for position in top_positions:

            index = candidate_indices[
                int(position)
            ]

            question_text = str(
                df.iloc[index]["question"]
            )

            normalized_question = normalize_text(
                question_text
            )

            question_words = set(
                normalized_question.split()
            )

            semantic_score = float(
                scores[int(position)]
            )

            # Count overlap with the requested intent.
            intent_overlap = len(
                question_words.intersection(
                    intent_words
                )
            )

            # Direct overlap with words in the current
            # question also helps.
            current_overlap = len(
                question_words.intersection(
                    current_words
                )
            )

            # Small reranking boost for the requested intent.
            rank_score = (
                semantic_score
                + min(intent_overlap, 3) * 0.025
                + min(current_overlap, 3) * 0.008
            )

            if rank_score > best_rank_score:

                best_rank_score = rank_score

                best = {
                    "index": int(index),
                    "score": semantic_score,
                    "question": question_text,
                    "answer": df.iloc[
                        index
                    ]["answer"]
                }

        if best is not None:

            best["search_query"] = (
                context_query
            )

            best["intent"] = intent

            return best

    # Final fallback: complete dataset.
    direct = find_best_match(
        context_query
    )

    if direct is not None:

        direct["search_query"] = (
            context_query
        )

        direct["intent"] = intent

    return direct


# ============================================================
# BUILD CONTEXT QUERY
# ============================================================

def build_context_query(
    current_question
):

    current_question = (
        current_question.strip()
    )

    explicit_topic = detect_explicit_topic(
        current_question
    )

    # Known explicit topic = new topic.
    if explicit_topic:

        return {
            "used_context": False,
            "active_topic": explicit_topic,
            "query": current_question,
            "intent": "new topic",
            "context": []
        }

    context = get_active_context()

    active_topic = context[
        "active_topic"
    ]

    if not active_topic:

        return {
            "used_context": False,
            "active_topic": None,
            "query": current_question,
            "intent": "new topic",
            "context": []
        }

    if not is_follow_up_question(
        current_question,
        context
    ):

        return {
            "used_context": False,
            "active_topic": None,
            "query": current_question,
            "intent": "new topic",
            "context": []
        }

    intent = detect_follow_up_intent(
        current_question
    )

    search_query = " ".join(
        x for x in [
            str(active_topic),
            context["previous_question"],
            current_question,
            intent if intent != "general" else ""
        ]
        if x
    )

    return {
        "used_context": True,
        "active_topic": active_topic,
        "query": search_query,
        "intent": intent,
        "context": [active_topic],
        "previous_question":
            context["previous_question"],
        "previous_answer":
            context["previous_answer"]
    }


# ============================================================
# GET ANSWER
# ============================================================

def get_answer(
    user_question,
    threshold=0.55
):

    user_question = (
        user_question.strip()
    )

    if not user_question:

        return {
            "answer": "Please enter a question.",
            "matched_question": None,
            "score": 0.0,
            "matched": False,
            "context_used": False,
            "context_questions": [],
            "search_query": "",
            "active_topic": None,
            "intent": "none"
        }

    context_result = build_context_query(
        user_question
    )

    context_used = context_result[
        "used_context"
    ]

    active_topic = context_result[
        "active_topic"
    ]

    # ========================================================
    # FOLLOW-UP
    # ========================================================

    if context_used:

        match = find_contextual_match(
            active_topic,
            context_result.get(
                "previous_question",
                ""
            ),
            context_result.get(
                "previous_answer",
                ""
            ),
            user_question
        )

        if match is None:

            return {
                "answer":
                    "Sorry, I do not have information related to this question.",
                "matched_question": None,
                "score": 0.0,
                "matched": False,
                "context_used": True,
                "context_questions": [active_topic],
                "search_query":
                    context_result["query"],
                "active_topic": active_topic,
                "intent":
                    context_result["intent"]
            }

        best_score = match["score"]
        best_question = match["question"]
        best_answer = match["answer"]

        search_query = match.get(
            "search_query",
            context_result["query"]
        )

        intent = match.get(
            "intent",
            context_result["intent"]
        )

        context_questions = [
            active_topic
        ]

    # ========================================================
    # NEW / DIRECT QUESTION
    # ========================================================

    else:

        match = find_best_match(
            user_question
        )

        if match is None:

            return {
                "answer":
                    "Sorry, I do not have information related to this question.",
                "matched_question": None,
                "score": 0.0,
                "matched": False,
                "context_used": False,
                "context_questions": [],
                "search_query": user_question,
                "active_topic": active_topic,
                "intent": "new topic"
            }

        best_score = match["score"]
        best_question = match["question"]
        best_answer = match["answer"]

        search_query = user_question
        intent = "new topic"
        context_questions = []

    # ========================================================
    # THRESHOLD
    # ========================================================

    if best_score >= threshold:

        return {
            "answer": best_answer,
            "matched_question": best_question,
            "score": best_score,
            "matched": True,
            "context_used": context_used,
            "context_questions": context_questions,
            "search_query": search_query,
            "active_topic": active_topic,
            "intent": intent
        }

    # ========================================================
    # FALLBACK
    # ========================================================

    return {
        "answer":
            "Sorry, I do not have information related to this question.",
        "matched_question": best_question,
        "score": best_score,
        "matched": False,
        "context_used": context_used,
        "context_questions": context_questions,
        "search_query": search_query,
        "active_topic": active_topic,
        "intent": intent
    }


# ============================================================
# SESSION STATE
# ============================================================

if "conversation_id" not in st.session_state:

    st.session_state.conversation_id = (
        create_new_conversation()
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# LOAD SELECTED CHAT
# ============================================================

def load_selected_conversation(
    conversation_id
):

    history_df = load_conversation(
        conversation_id
    )

    messages = []

    for _, row in history_df.iterrows():

        messages.append(
            {
                "role": "user",
                "content": row["question"]
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": row["answer"],
                "score": row["similarity_score"],
                "matched": bool(
                    row["matched"]
                ),
                "matched_question":
                    row["matched_question"]
            }
        )

    st.session_state.conversation_id = (
        conversation_id
    )

    st.session_state.messages = messages


# ============================================================
# START NEW CHAT
# ============================================================

def start_new_chat():

    st.session_state.conversation_id = (
        create_new_conversation()
    )

    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 EduBot AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI-powered educational assistant</div>',
    unsafe_allow_html=True
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


    # --------------------------------------------------------
    # AI MODEL
    # --------------------------------------------------------

    st.subheader("🧠 AI Model")

    st.caption(
        "Sentence Transformer: all-MiniLM-L6-v2"
    )

    st.caption(
        "Context Memory: Enabled"
    )

    st.caption(
        "Active Topic Tracking: Enabled"
    )

    # --------------------------------------------------------
    # SUGGESTED QUESTIONS
    # --------------------------------------------------------

    st.divider()

    st.subheader("💡 Suggested Questions")

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
            key=f"suggest_{question}",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                question
            )

            st.rerun()

    # --------------------------------------------------------
    # SIMILARITY THRESHOLD
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🎯 Similarity Threshold"
    )

    threshold = st.slider(
        "Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.55,
        step=0.05
    )

    # --------------------------------------------------------
    # CHAT HISTORY AT LAST
    # --------------------------------------------------------

    st.divider()

    st.subheader("📚 Chat History")

    conversations_df = get_conversations()

    if conversations_df.empty:

        st.caption(
            "No saved conversations yet."
        )

    else:

        for _, conversation in conversations_df.iterrows():

            conversation_id = int(
                conversation["id"]
            )

            title = str(
                conversation["title"]
            )

            row_col, delete_col = st.columns(
                [6, 1],
                gap="small"
            )

            # Clickable history
            with row_col:

                is_current = (
                    conversation_id
                    == st.session_state.conversation_id
                )

                if is_current:

                    button_label = (
                        f"🟢 {title}"
                    )

                else:

                    button_label = (
                        f"💬 {title}"
                    )

                if st.button(
                    button_label,
                    key=f"open_chat_{conversation_id}",
                    use_container_width=True
                ):

                    load_selected_conversation(
                        conversation_id
                    )

                    st.rerun()

            # Individual delete
            with delete_col:

                if st.button(
                    "🗑️",
                    key=f"delete_chat_{conversation_id}",
                    help="Delete this chat"
                ):

                    delete_conversation(
                        conversation_id
                    )

                    if (
                        conversation_id
                        == st.session_state.conversation_id
                    ):

                        start_new_chat()

                    st.rerun()

    # --------------------------------------------------------
    # DELETE ALL
    # --------------------------------------------------------

    if not conversations_df.empty:

        st.divider()

        if st.button(
            "🗑️ Delete Chat History",
            use_container_width=True
        ):

            delete_all_conversations()

            start_new_chat()

            st.rerun()

    # --------------------------------------------------------
    # HISTORY DATABASE SECTION (NEWLY ADDED)
    # --------------------------------------------------------
    st.divider()
    
    if st.button("🗄️ History Database", use_container_width=True):
        st.session_state.show_history_db = not st.session_state.get("show_history_db", False)

    if st.session_state.get("show_history_db", False):
        st.caption("### 🗄️ Saved Database Logs")
        
        conn = get_db_connection()
        logs_df = pd.read_sql_query(
            """
            SELECT 
                conversation_id AS ID,
                question AS Question,
                answer AS Answer,
                similarity_score AS Similarity,
                timestamp AS [Date & Time]
            FROM messages
            ORDER BY timestamp DESC
            """, 
            conn
        )
        conn.close()
        
        if logs_df.empty:
            st.info("No logs saved in database yet.")
        else:
            logs_df["Similarity"] = logs_df["Similarity"].apply(lambda x: f"{x * 100:.2%}" if isinstance(x, (int, float)) else "0.00%")
            
            st.dataframe(
                logs_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        """
        <div class="welcome-box">

        ### 🎓 Welcome to EduBot AI!

        Ask questions related to the educational knowledge base.

        **Example:**

        > Can you explain Machine Learning?

        I will find the most semantically similar question and return its corresponding answer.

        **Topics Include**

        <div>Artificial Intelligence</div>
        <div>Machine Learning</div>
        <div>Deep Learning</div>
        <div>Python</div>
        <div>NLP</div>
        <div>Computer Vision</div>
        <div>Generative AI</div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CURRENT CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

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
# CHAT INPUT
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

    user_question = (
        user_question.strip()
    )

    if not user_question:
        st.stop()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 EduBot is thinking..."
        ):

            result = get_answer(
                user_question,
                threshold
            )

        st.markdown(
            result["answer"]
        )

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

            if result["context_used"]:

                st.write(
                    "### 🧠 Context Memory"
                )

                st.write(
                    "EduBot detected this as a "
                    "follow-up question."
                )

                st.write(
                    f"**Active Topic:** "
                    f"{result['active_topic']}"
                )

                st.write(
                    f"**Follow-up Intent:** "
                    f"{result['intent']}"
                )

                st.write(
                    f"**Focused Search Query:** "
                    f"{result['search_query']}"
                )

            else:

                st.write(
                    "### 🆕 New Topic"
                )

                st.write(
                    "EduBot treated this as a "
                    "new topic."
                )

            if result["matched_question"]:

                st.write(
                    "**Best Matching Question:**"
                )

                st.write(
                    result["matched_question"]
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
    # SAVE TO SQLITE
    # --------------------------------------------------------

    save_chat(

        conversation_id=(
            st.session_state.conversation_id
        ),

        question=user_question,

        answer=result["answer"],

        similarity_score=result["score"],

        matched_question=(
            result["matched_question"]
        ),

        matched=result["matched"]
    )

    # --------------------------------------------------------
    # ADD ASSISTANT MESSAGE TO SESSION
    # --------------------------------------------------------

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": result["answer"],

            "score": result["score"],

            "matched": result["matched"],

            "matched_question":
                result["matched_question"]
        }
    )
