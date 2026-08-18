# Project Documentation — AI-Powered Educational Chatbot

## 1. Introduction

The AI-Powered Educational Chatbot is a retrieval-based educational question-answering system. It reads a CSV knowledge base containing questions and trusted answers, converts the stored questions and the user's query into semantic embeddings, and retrieves the most relevant answer.

## 2. Problem statement

Users should be able to ask educational questions in natural language. The system should retrieve an answer from the supplied knowledge base and should not invent an answer when relevant information is absent.

## 3. Objectives

- Build a CSV-driven question-answering system.
- Use semantic retrieval instead of exact keyword matching.
- Provide a fallback when relevance is insufficient.
- Provide chat history and suggested questions.
- Provide a clean interface suitable for future web integration.

## 4. Technology stack

- Python
- Pandas
- NumPy
- Sentence Transformers
- Streamlit
- PyTorch
- Pytest

## 5. Dataset

The starter dataset has 300 question-answer pairs covering:
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

Recommended final size: 250-300 curated records.

## 6. Methodology

### Step 1: Load data
Pandas reads the CSV and validates the required `question` and `answer` columns.

### Step 2: Clean data
Rows with missing questions or answers are removed and text is stripped.

### Step 3: Create embeddings
The `all-MiniLM-L6-v2` Sentence Transformer converts each stored question into a dense vector.

### Step 4: Encode user query
The same model converts the user's question into an embedding.

### Step 5: Similarity calculation
The embeddings are normalized. Their dot product is therefore equivalent to cosine similarity.

### Step 6: Retrieval
The highest-scoring knowledge-base question is selected.

### Step 7: Threshold
If the similarity is at least the configured threshold, its answer is returned. Otherwise, the chatbot returns the fallback response.

## 7. Why Sentence Transformers?

Exact string matching would fail when users phrase the same idea differently. Sentence embeddings allow semantically similar questions to be compared in vector space.

Example:

`What is Machine Learning?`

and

`Can you explain ML?`

may have similar embeddings even though the words are not identical.

## 8. User interface

The Streamlit interface provides:
- User and assistant message bubbles.
- Chat history during the current session.
- Suggested questions.
- Loading indicator.
- Retrieval confidence.
- Matched knowledge-base question.

## 9. Testing strategy

### Functional tests
Check exact questions, semantic variations, empty input, and fallback behavior.

### Dataset tests
Check required columns, missing values, duplicates, and row count.

### Retrieval tests
Prepare a test set containing:
- in-domain questions with expected records
- paraphrases of stored questions
- out-of-domain questions

### Threshold tuning
Try several thresholds, for example 0.50, 0.55, 0.60, and 0.65. Select a value that keeps relevant answers while rejecting unrelated questions. Do not select the threshold only because it gives a good result on one example.

## 10. Limitations

- The system can only answer questions represented by the knowledge base.
- A similarity score is not a proof that an answer is factually correct.
- The quality of retrieval depends on the quality and coverage of the CSV.
- The starter system does not generate new answers with an LLM.

## 11. Future scope

- Expand the knowledge base.
- Add persistent storage.
- Add REST APIs using FastAPI.
- Add a vector database for larger datasets.
- Add RAG with an LLM while restricting generation to retrieved evidence.
- Add multilingual support.
- Add monitoring and retrieval evaluation.

## 12. Conclusion

The project demonstrates a practical semantic-search chatbot using a CSV knowledge base, Sentence Transformers, similarity-based retrieval, thresholded fallback behavior, and a Streamlit interface.
