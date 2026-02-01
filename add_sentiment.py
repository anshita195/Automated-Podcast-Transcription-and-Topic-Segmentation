#!/usr/bin/env python3
"""
add_sentiment.py

Adds sentiment score and label to each segment using VADER.
Input  : output_kws_summaries/json_updated/*.json
Output : output_kws_summaries/json_with_sentiment/*.json
"""

import os
import json
from glob import glob
from tqdm import tqdm

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# Ensure VADER is available
# -----------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = "output_kws_summaries/json_updated"
OUTPUT_DIR = "output_kws_summaries/json_with_sentiment"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Sentiment helper
# -----------------------------
def get_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return compound, label

# -----------------------------
# Main processing
# -----------------------------
json_files = sorted(glob(os.path.join(INPUT_DIR, "*.json")))
print(f"Found {len(json_files)} files")

for path in tqdm(json_files, desc="Adding sentiment"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    for seg in segments:
        text = seg.get("text", "")
        score, label = get_sentiment(text)

        seg["sentiment_score"] = score
        seg["sentiment_label"] = label

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Sentiment added successfully.")
print("Output directory:", OUTPUT_DIR)
