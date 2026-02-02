import os
import json
import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "output_kws_summaries/json_with_sentiment"

st.set_page_config(
    page_title="Podcast Topic Timeline",
    layout="wide"
)

st.title("🎧 Podcast Topic Timeline")
st.caption(
    "Visual timeline of podcast topics. "
    "Block width ∝ segment length (word count). "
    "Color indicates sentiment."
)

# -----------------------------
# LOAD EPISODE FILES
# -----------------------------
@st.cache_data
def load_episode_files(data_dir):
    return sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".json")]
    )

episode_files = load_episode_files(DATA_DIR)

if not episode_files:
    st.error("No episode JSON files found.")
    st.stop()

# -----------------------------
# LOAD TITLES FOR DROPDOWN
# -----------------------------
@st.cache_data
def load_episode_titles(data_dir, files):
    mapping = {}
    for fname in files:
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            title = data.get("title", fname)
            mapping[title] = fname
    return mapping

title_to_file = load_episode_titles(DATA_DIR, episode_files)

# -----------------------------
# EPISODE SELECTOR
# -----------------------------
selected_title = st.selectbox(
    "🎙️ Select Podcast Episode",
    options=list(title_to_file.keys())
)

episode_path = os.path.join(DATA_DIR, title_to_file[selected_title])
with open(episode_path, "r", encoding="utf-8") as f:
    episode_data = json.load(f)

segments = episode_data.get("segments", [])

if not segments:
    st.warning("No segments found for this episode.")
    st.stop()

st.markdown(f"### {episode_data.get('title', 'Unknown Episode')}")
st.markdown("---")

# ============================================================
# 🧭 VISUAL TIMELINE (CORE WEEK 5 FEATURE)
# ============================================================

timeline_rows = []
cursor = 0

for seg in segments:
    timeline_rows.append({
        "segment_id": f"S{seg['segment_id']}",
        "start": cursor,
        "end": cursor + seg["num_words"],
        "sentiment": seg.get("sentiment_label", "Neutral"),
        "summary": seg.get("summary", "")
    })
    cursor += seg["num_words"]

df_timeline = pd.DataFrame(timeline_rows)

color_scale = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#4CAF50", "#FFC107", "#F44336"]
)

timeline_chart = (
    alt.Chart(df_timeline)
    .mark_bar(height=30)
    .encode(
        x=alt.X(
            "start:Q",
            title="Podcast Progress (word-based)",
            axis=alt.Axis(grid=False)
        ),
        x2="end:Q",
        y=alt.value(0),
        color=alt.Color(
            "sentiment:N",
            scale=color_scale,
            legend=alt.Legend(title="Sentiment")
        ),
        tooltip=[
            alt.Tooltip("segment_id:N", title="Segment"),
            alt.Tooltip("sentiment:N", title="Sentiment"),
            alt.Tooltip("summary:N", title="Summary")
        ],
    )
    .properties(height=80)
)

st.altair_chart(timeline_chart, use_container_width=True)
st.markdown("---")

# ============================================================
# SEGMENT SELECTION (INTERACTION)
# ============================================================

def build_segment_label(seg):
    summary = seg.get("summary", "").strip()
    if summary:
        return f"S{seg['segment_id']}: {summary[:70]}..."
    return f"S{seg['segment_id']}"

labels = [build_segment_label(s) for s in segments]

selected_index = st.selectbox(
    "📌 Select Topic Segment",
    options=list(range(len(labels))),
    format_func=lambda i: labels[i]
)

selected_segment = segments[selected_index]

# ============================================================
# SEGMENT DETAILS DISPLAY
# ============================================================

st.markdown("---")
st.subheader(f"📄 Segment {selected_segment['segment_id']}")

col1, col2, col3 = st.columns(3)
col1.metric("Start sentence", selected_segment["start_sentence"])
col2.metric("End sentence", selected_segment["end_sentence"])
col3.metric("Word count", selected_segment["num_words"])

# -----------------------------
# SUMMARY (POLISHED DISPLAY)
# -----------------------------
st.markdown("### 📝 Summary")
st.write(selected_segment.get("summary", "No summary available."))

# -----------------------------
# KEYWORDS + WORD CLOUD
# -----------------------------
st.markdown("### 🔑 Keywords")

keywords = selected_segment.get("keywords", [])
if keywords:
    st.write(", ".join(keywords))

    freq = {kw: 1 for kw in keywords}
    wc = WordCloud(
        width=600,
        height=300,
        background_color="white"
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No keywords available.")

# -----------------------------
# SENTIMENT
# -----------------------------
st.markdown("### 😊 Sentiment")

sentiment_label = selected_segment.get("sentiment_label", "Unknown")
sentiment_score = selected_segment.get("sentiment_score", 0.0)

color_map = {
    "Positive": "green",
    "Neutral": "orange",
    "Negative": "red"
}

st.markdown(
    f"<span style='color:{color_map.get(sentiment_label, 'black')};"
    f"font-weight:bold'>"
    f"{sentiment_label}</span> "
    f"(score: {sentiment_score:.2f})",
    unsafe_allow_html=True
)

# -----------------------------
# TRANSCRIPT
# -----------------------------
st.markdown("### 📜 Transcript Text")
st.text_area(
    "",
    selected_segment["text"],
    height=400
)

st.caption(
    "Use the timeline for overview and the dropdown for precise navigation."
)
