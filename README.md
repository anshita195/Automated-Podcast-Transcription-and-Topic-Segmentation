# Automated Podcast Transcription and Topic Segmentation

This repository contains the initial milestone work for a project focused on building a complete pipeline for long-form podcast transcription and topic segmentation.

The primary goal of this phase is to prepare a high-quality, large-scale dataset consisting of aligned podcast audio and transcripts, suitable for downstream tasks such as topic boundary detection, summarization, and semantic segmentation.

---

## Dataset

The project uses the **Lex Fridman Podcast** dataset from Hugging Face. 
https://huggingface.co/datasets/Whispering-GPT/lex-fridman-podcast

This dataset was selected because:

- Episodes are long-form (typically 1–3 hours)
- Conversations span diverse technical and philosophical topics
- High-quality transcripts are available
- The format is well-suited for topic segmentation research

---

## Audio Files

Due to the large size of the processed audio files, they are not hosted directly on GitHub.

They can be downloaded from the following Google Drive folder:

**Google Drive:**  
https://drive.google.com/drive/folders/11WZNY_P54YauvS8zX-9YsdDo9A6JBctt?usp=sharing

The folder contains fully preprocessed podcast episodes in standardized WAV format.

---

## Repository Structure

### Files

#### `train-00000-of-00001-25f40520d4548308.parquet`

Original dataset in Parquet format.

#### `parquet_to_csv.py`

Converts the Parquet file into a CSV file (`lex_fridman_full.csv`) without cleaning.

#### `lex_fridman_full.csv`

Raw CSV conversion of the Parquet dataset.

#### `prepare_dataset_and_fetch_audio.py`

Main dataset preparation script. This script:

- Extracts relevant columns
- Adds row identifiers
- Creates a cleaned CSV
- Downloads episode audio
- Converts audio to WAV
- Resamples to 16 kHz mono
- Applies consistent naming
- Generates a download manifest

#### `lex_fridman_cleaned.csv`

Cleaned CSV containing:

- `row_id`
- `video_id`
- `title`
- `transcript`

#### `preprocessing.py`

Applies signal-level preprocessing:

- Noise reduction
- Loudness normalization
- Silence trimming

---

## Audio Format Standardization

All podcast episodes are standardized to the following format:

- WAV
- 16,000 Hz sample rate
- Mono channel

This format is commonly used in speech processing and NLP pipelines.

---

## Naming Convention

Each audio file follows this format:

podcast<row_id>_<video_id>.resampled.wav

makefile
Copy code

**Example:**

podcast036_HYsLTNXMl1Q.resampled.wav

yaml
Copy code

This ensures traceability between CSV entries and audio files.

---

## Dataset Size

Each Lex Fridman episode is typically 1–2 hours long. Even without using all audio files, the dataset already contains:

- Over 70 hours of audio even if we use only 45 podcasts
- More than 10 GB of processed data

This is sufficient for initial modeling, experimentation, and evaluation.

---

## Preprocessing Steps

Each audio file undergoes the following preprocessing:

- Noise reduction
- Loudness normalization
- Silence trimming
- Resampling to 16 kHz
- Conversion to mono

This ensures uniformity and improves data quality for downstream tasks.

---

## Current Status

The following components have been completed:

- Dataset selection
- Parquet to CSV conversion
- Dataset cleaning
- Audio downloading
- Audio standardization
- Audio preprocessing
- Naming standardization
- Manifest generation
- Reproducible pipeline setup

---

## Planned Next Steps

Future phases of the project will include:

- Automatic topic boundary detection
- Segment-level transcript alignment
- Embedding generation
- Clustering-based segmentation
- Summarization
- Visualization and navigation interface

---

## Reproducibility

The entire dataset can be regenerated from scratch using the provided scripts:

1. Convert Parquet to CSV
2. Clean metadata
3. Download episode audio
4. Standardize format (WAV, 16 kHz, mono)
5. Apply preprocessing

This ensures full reproducibility of the dataset pipeline.

---

## Notes

Due to the large size of the audio files, they are not stored directly in this GitHub repository.

All processed episodes can be downloaded from the Google Drive link provided above. The dataset can also be regenerated locally using the provided scripts.
