from pathlib import Path
import csv

# This project already contains a 200-row knowledge_base.csv.
# To expand it to 250-300 rows, add more records to the TOPICS dictionary
# using the same {"category", "question", "answer"} structure, then run:
#     python generate_dataset.py

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "knowledge_base.csv"

# Example extension records. Copy this pattern to add more rows.
EXTRA_ROWS = [
    {"category": "Artificial Intelligence", "question": "What is AI planning?", "answer": "AI planning is the process of finding a sequence of actions that can achieve a specified goal from an initial state."},
    {"category": "Machine Learning", "question": "What is ensemble learning?", "answer": "Ensemble learning combines predictions from multiple models to improve robustness or predictive performance."},
    {"category": "Deep Learning", "question": "What is an optimizer?", "answer": "An optimizer is an algorithm that updates model parameters using gradients or another optimization strategy to reduce the loss."},
    {"category": "Python", "question": "What is a generator in Python?", "answer": "A generator produces values lazily, usually with yield, so values can be generated one at a time instead of stored all at once."},
    {"category": "Data Science", "question": "What is data profiling?", "answer": "Data profiling examines structure, quality, distributions, missingness, and patterns in a dataset."},
    {"category": "Statistics", "question": "What is a z score?", "answer": "A z score indicates how many standard deviations an observation is from the mean."},
    {"category": "NLP", "question": "What is machine translation?", "answer": "Machine translation automatically converts text or speech from one language to another using computational models."},
    {"category": "Computer Vision", "question": "What is OCR?", "answer": "Optical Character Recognition (OCR) extracts machine-readable text from images or scanned documents."},
    {"category": "SQL", "question": "What is a CTE?", "answer": "A Common Table Expression (CTE) is a named temporary query result defined with WITH and used by a larger SQL statement."},
    {"category": "Generative AI", "question": "What is an AI guardrail?", "answer": "An AI guardrail is a rule, filter, validation step, or control used to reduce unsafe, irrelevant, or invalid model behavior."},
]

with DATA.open("r", newline="", encoding="utf-8") as f:
    existing = list(csv.DictReader(f))

seen = {(r["category"], r["question"].strip().lower()) for r in existing}
next_id = len(existing) + 1
for item in EXTRA_ROWS:
    key = (item["category"], item["question"].strip().lower())
    if key not in seen:
        existing.append({"id": next_id, **item})
        next_id += 1

with DATA.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "category", "question", "answer"])
    writer.writeheader()
    writer.writerows(existing)

print(f"Dataset ready: {len(existing)} rows -> {DATA}")
