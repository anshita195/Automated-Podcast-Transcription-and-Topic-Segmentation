import os
import json
from pydub import AudioSegment, effects
import whisper

# -----------------------------
# CONFIG
# -----------------------------
INPUT_AUDIO = "audio_input/example.wav"
CHUNK_DIR = "audio_chunks"
PROCESSED_DIR = "audio_processed"
TRANSCRIPT_DIR = "transcripts"

CHUNK_LENGTH_MS = 25 * 1000  # 25 seconds
WHISPER_MODEL = "base"       # base is enough for Week 2

# -----------------------------
# CREATE DIRECTORIES
# -----------------------------
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# -----------------------------
# LOAD + PREPROCESS AUDIO
# -----------------------------
print("Loading audio...")
audio = AudioSegment.from_file(INPUT_AUDIO)

# Convert format
audio = audio.set_frame_rate(16000).set_channels(1)

# Normalize loudness
audio = effects.normalize(audio)

processed_path = os.path.join(PROCESSED_DIR, "example_clean.wav")
audio.export(processed_path, format="wav")
print(f"Saved cleaned audio → {processed_path}")

# -----------------------------
# CHUNK AUDIO
# -----------------------------
print("Chunking audio...")
chunks = []
for i in range(0, len(audio), CHUNK_LENGTH_MS):
    chunk = audio[i:i + CHUNK_LENGTH_MS]
    chunk_path = os.path.join(
        CHUNK_DIR, f"example_chunk_{i//CHUNK_LENGTH_MS:03d}.wav"
    )
    chunk.export(chunk_path, format="wav")

    chunks.append({
        "chunk_id": i // CHUNK_LENGTH_MS,
        "start_sec": i / 1000,
        "end_sec": min((i + CHUNK_LENGTH_MS) / 1000, len(audio) / 1000),
        "path": chunk_path
    })

print(f"Created {len(chunks)} chunks")

# -----------------------------
# LOAD WHISPER
# -----------------------------
print("Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)

# -----------------------------
# TRANSCRIBE CHUNKS
# -----------------------------
full_text = []
chunk_outputs = []

print("Transcribing...")
for c in chunks:
    result = model.transcribe(c["path"], fp16=False)
    text = result["text"].strip()

    full_text.append(text)
    chunk_outputs.append({
        "chunk_id": c["chunk_id"],
        "start_sec": c["start_sec"],
        "end_sec": c["end_sec"],
        "text": text
    })

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
txt_path = os.path.join(TRANSCRIPT_DIR, "example.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(full_text))

json_path = os.path.join(TRANSCRIPT_DIR, "example_chunks.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(chunk_outputs, f, indent=2)

print("Transcription complete!")
print(f"Saved transcript → {txt_path}")
print(f"Saved chunk metadata → {json_path}")
