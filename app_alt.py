import os
import json
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from audio_preprocessing import process_file as preprocess_file
from transcription_generation import transcribe_episode
from keywords_and_summaries import (
    extract_keywords_tfidf,
    build_segment_title,
    normalize_text,
    summarize_t5,
)
from add_sentiment import get_sentiment


# =======================
# CONFIG
# =======================
INPUT_DIR = Path("audio_input")
CHUNKS_DIR = Path("audio_chunks")
TRANSCRIPTS_DIR = Path("transcripts")
SEGMENTS_DIR = Path("segments_runtime")

MODEL_NAME = "all-MiniLM-L6-v2"


# =======================
# SEGMENTATION HELPERS
# =======================
SENT_SPLIT_RE = r'(?<=[\.\?\!])\s+|\n+'


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_t5_model():
    """
    Load the local T5 model once and reuse it.
    Expects the same local directory used in keywords_and_summaries.py.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    local_dir = "models/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)
    model.eval()
    return model, tokenizer


def split_into_sentences(text: str):
    import re

    if not isinstance(text, str) or text.strip() == "":
        return []
    pieces = [s.strip() for s in re.split(SENT_SPLIT_RE, text) if s and s.strip()]
    return pieces


def word_count(text: str) -> int:
    return len(text.split())


def compute_text_similarities(texts, model):
    """Compute cosine similarity between consecutive text units using embeddings."""
    if len(texts) < 2:
        return np.array([], dtype=float)

    embeddings = model.encode(texts, show_progress_bar=False)
    sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        sims.append(float(sim))

    return np.array(sims, dtype=float)


def detect_boundaries_from_sims(sims, auto_k: float = 1.0):
    boundaries = [0]
    if sims.size == 0:
        return boundaries

    sims_mean = float(np.mean(sims))
    sims_std = float(np.std(sims))
    threshold = max(0.0, sims_mean - auto_k * sims_std)

    for i, s in enumerate(sims):
        if s < threshold:
            boundaries.append(i + 1)
    return boundaries


def build_segments_from_units(units, boundaries):
    """
    Build topic segments from low-level Whisper units that already have timestamps.
    Each unit is expected to be a dict with at least: start, end, text.
    """
    segments = []

    for seg_id, start_idx in enumerate(boundaries):
        if seg_id + 1 < len(boundaries):
            end_idx = boundaries[seg_id + 1] - 1
        else:
            end_idx = len(units) - 1

        unit_slice = units[start_idx:end_idx + 1]
        seg_text = " ".join(u.get("text", "") for u in unit_slice).strip()

        if not seg_text:
            continue

        segments.append({
            "segment_id": seg_id,
            "start_unit": start_idx,
            "end_unit": end_idx,
            "start_time": float(unit_slice[0]["start"]),
            "end_time": float(unit_slice[-1]["end"]),
            "text": seg_text,
            "num_words": word_count(seg_text)
        })

    return segments


def enrich_segments(segments):
    """
    Add keywords, T5-based summary, title and sentiment.
    """
    # Try to load T5 once (cached). If it fails, we fall back to a simple heuristic.
    t5_model = None
    t5_tokenizer = None
    try:
        t5_model, t5_tokenizer = load_t5_model()
    except Exception:
        t5_model = None
        t5_tokenizer = None

    for seg in segments:
        raw_text = normalize_text(seg.get("text", ""))

        # Keywords
        kws = extract_keywords_tfidf(raw_text, top_k=10)
        seg["keywords"] = kws

        # Summary: prefer T5, fall back to first 1–2 sentences
        summary = ""
        if t5_model is not None and t5_tokenizer is not None:
            try:
                summary = summarize_t5(t5_model, t5_tokenizer, raw_text)
            except Exception:
                summary = ""

        if not summary:
            sents = split_into_sentences(raw_text)
            summary = " ".join(sents[:2]) if sents else raw_text[:200]

        seg["summary"] = summary

        # Title: derive a short title from the summary; fall back to keyword-based title
        title = build_title_from_summary(summary, max_chars=60)
        if not title:
            title = build_segment_title(kws, summary, max_words=3)

        seg["title"] = title

        # Sentiment from full segment text
        score, label = get_sentiment(raw_text)
        seg["sentiment_score"] = score
        seg["sentiment_label"] = label

    return segments


# =======================
# UI HELPERS
# =======================

def render_timeline(segments, total_duration: float):
    if not segments:
        st.info("Timeline not available for this audio.")
        return

    rows = []
    for seg in segments:
        rows.append({
            "segment": f"S{seg['segment_id']}",
            "start": float(seg.get("start_time", 0.0)),
            "end": float(seg.get("end_time", 0.0)),
            "sentiment": seg.get("sentiment_label", "Neutral"),
            "title": seg.get("title", ""),
            "start_label": format_time(seg.get("start_time", 0.0)),
            "end_label": format_time(seg.get("end_time", 0.0)),
        })

    df = pd.DataFrame(rows)

    timeline = (
        alt.Chart(df)
        .mark_bar(height=30)
        .encode(
            x=alt.X(
                "start:Q",
                axis=alt.Axis(
                    title="Time (mm:ss)",
                    labelExpr=(
                        "floor(datum.value/60) + ':' + "
                        "(datum.value % 60 < 10 ? '0' : '') + (datum.value % 60)"
                    ),
                ),
            ),
            x2="end:Q",
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#4CAF50", "#FFC107", "#F44336"],
                ),
            ),
            tooltip=[
                "segment",
                "sentiment",
                "title",
                alt.Tooltip("start_label:N", title="Start"),
                alt.Tooltip("end_label:N", title="End"),
            ],
        )
        .properties(height=90)
    )

    st.altair_chart(timeline, use_container_width=True)


def format_time(seconds: float) -> str:
    """Convert seconds to mm:ss format."""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m:02d}:{s:02d}"


# =======================
# KEYWORD CLOUD HELPERS
# =======================
DISPLAY_STOPWORDS = {
    "yeah", "oh", "okay", "right", "know", "thing", "et", "cetera"
}


def clean_keywords_for_display(keywords):
    return [kw for kw in keywords if kw.lower() not in DISPLAY_STOPWORDS]


def render_keyword_cloud(keywords):
    from wordcloud import WordCloud

    if not keywords:
        st.info("No keywords available.")
        return

    freq = {kw: 1 for kw in keywords}
    wc = WordCloud(
        width=600,
        height=300,
        background_color="white",
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def build_title_from_summary(summary: str, max_chars: int = 60) -> str:
    """
    Build a short, human-readable title from the T5 summary.
    - Take the first sentence of the summary
    - Truncate to max_chars without cutting the last word
    - Ensure the first character is capitalized
    """
    import re

    if not isinstance(summary, str):
        return ""

    s = summary.strip().replace("\n", " ")
    if not s:
        return ""

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", s)
    first = sentences[0].strip() if sentences else s
    if not first:
        first = s

    # Truncate cleanly
    if len(first) > max_chars:
        cut = first[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        first = cut

    # Capitalize first character
    if first:
        first = first[0].upper() + first[1:]

    return first


# =======================
# CACHED SEGMENTATION
# =======================

@st.cache_data
def build_segments_from_json(json_transcript_path_str: str):
    with open(json_transcript_path_str, "r", encoding="utf-8") as f:
        data = json.load(f)

    units = data.get("segments", [])
    if not units:
        return [], 0.0

    units = sorted(units, key=lambda u: u.get("start", 0.0))
    total_duration = float(units[-1].get("end", 0.0))

    texts = [u.get("text", "") for u in units]

    model = load_embedding_model()
    sims = compute_text_similarities(texts, model)
    boundaries = detect_boundaries_from_sims(sims, auto_k=1.0)
    segments = build_segments_from_units(units, boundaries)
    segments = enrich_segments(segments)

    return segments, total_duration


def main():
    st.set_page_config(page_title="Podcast Navigator (Alt)", layout="wide")
    st.title("🎧 Podcast Pipeline Demo (Alt UI)")
    st.caption("Upload an audio file → preprocess → transcribe → segment → keywords, titles, sentiment → navigate by segments.")

    uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    if not uploaded:
        st.info("Upload a podcast/audio file to start.")
        return

    # Save to audio_input
    INPUT_DIR.mkdir(exist_ok=True)
    file_path = INPUT_DIR / uploaded.name
    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    episode_name = file_path.stem

    # Prefer existing JSON transcript if it already exists (avoid recomputation)
    json_transcript_path = TRANSCRIPTS_DIR / f"{episode_name}.json"

    if not json_transcript_path.exists():
        # Run full pipeline only if transcript does not exist yet
        with st.spinner("Preprocessing audio (noise reduction, loudness, silence trimming, chunking)..."):
            preprocess_file(file_path)

        episode_chunk_dir = CHUNKS_DIR / episode_name

        with st.spinner("Running Whisper transcription on processed chunks..."):
            transcribe_episode(episode_chunk_dir)

    if not json_transcript_path.exists():
        st.error("Transcript JSON was not created.")
        return

    # Ensure segments directory exists
    SEGMENTS_DIR.mkdir(exist_ok=True)
    episode_seg_dir = SEGMENTS_DIR / episode_name
    episode_seg_dir.mkdir(exist_ok=True)
    segments_json_path = episode_seg_dir / "segments.json"

    # If we already have enriched segments saved, reuse them
    if segments_json_path.exists():
        with open(segments_json_path, "r", encoding="utf-8") as f:
            seg_payload = json.load(f)
        segments = seg_payload.get("segments", [])
        total_duration = float(
            seg_payload.get(
                "total_duration",
                segments[-1]["end_time"] if segments else 0.0,
            )
        )
    else:
        # Build segments (cached) from JSON with timestamps
        with st.spinner("Computing semantic similarities and segmenting transcript..."):
            segments, total_duration = build_segments_from_json(str(json_transcript_path))

        if not segments:
            st.error("Segmentation produced no segments.")
            return

        # Persist enriched segments for future reuse
        seg_payload = {
            "episode_name": episode_name,
            "source_transcript": str(json_transcript_path),
            "total_duration": total_duration,
            "n_segments": len(segments),
            "segments": segments,
        }
        with open(segments_json_path, "w", encoding="utf-8") as f:
            json.dump(seg_payload, f, ensure_ascii=False, indent=2)

    st.success(f"Ready with {len(segments)} segments for this episode.")

    # Timeline + segment selection
    st.markdown("### Segment Timeline")
    render_timeline(segments, total_duration or (segments[-1]["end_time"] if segments else 0.0))

    # Navigation + segment exploration (similar style to app.py)
    st.markdown("### Explore Segments")

    if "seg_index" not in st.session_state:
        st.session_state.seg_index = 0

    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        if st.button("⬅ Previous") and st.session_state.seg_index > 0:
            st.session_state.seg_index -= 1

    with c3:
        if st.button("Next ➡") and st.session_state.seg_index < len(segments) - 1:
            st.session_state.seg_index += 1

    labels = [
        f"S{seg['segment_id']}: {seg.get('title','Segment')[:70]}"
        for seg in segments
    ]

    st.session_state.seg_index = st.selectbox(
        "📌 Jump to Segment",
        range(len(labels)),
        index=st.session_state.seg_index,
        format_func=lambda i: labels[i],
    )

    seg = segments[st.session_state.seg_index]

    st.subheader(f"Segment {seg['segment_id']:02d} – {seg.get('title', '')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Start time", format_time(seg.get("start_time", 0.0)))
    col2.metric("End time", format_time(seg.get("end_time", 0.0)))
    duration = max(0.0, float(seg.get("end_time", 0.0)) - float(seg.get("start_time", 0.0)))
    col3.metric("Duration (sec)", f"{duration:.1f}")

    # Sentiment display
    score = seg.get("sentiment_score", 0.0)
    label = seg.get("sentiment_label", "Neutral")
    st.markdown(f"**Sentiment:** {label} (score: {score:.3f})")

    # Keywords + cloud
    st.markdown("### 🔑 Keywords")
    keywords = clean_keywords_for_display(seg.get("keywords", []))
    if keywords:
        st.write(", ".join(keywords))
        render_keyword_cloud(keywords)
    else:
        st.info("No meaningful keywords.")

    # Summary
    st.markdown("### 📝 Summary")
    st.write(seg.get("summary", ""))

    # Transcript text
    st.markdown("### 📜 Transcript")
    st.text_area(
        label="",
        value=seg.get("text", ""),
        height=350,
    )


if __name__ == "__main__":
    main()

