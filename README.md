# Podcast Summarizer

An AI-powered podcast summarizer that transcribes audio using **OpenAI Whisper** and generates summaries, key points, and highlights using **Facebook BART**.

## Features

- **Audio Upload** — Drag & drop or browse MP3, WAV, M4A, OGG, FLAC, or WEBM files
- **YouTube / Spotify URLs** — Paste a podcast link and the app downloads + processes it
- **AI Transcription** — Uses OpenAI Whisper (base model) for accurate speech-to-text
- **Smart Summarization** — Uses Facebook BART-large-CNN for extractive summarization
- **Key Points Extraction** — Identifies the most important discussion points
- **Highlights** — Surfaces notable quotes and statements
- **Full Transcript** — Expandable section with the complete transcription
- **Copy Results** — One-click copy of all results to clipboard

## Tech Stack

| Component      | Technology                  |
| -------------- | --------------------------- |
| Frontend       | React, Axios, Lucide Icons  |
| Backend        | FastAPI, Uvicorn             |
| Transcription  | OpenAI Whisper (base model) |
| Summarization  | Facebook BART-large-CNN     |
| Audio Download | yt-dlp (YouTube/Spotify)    |

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** — Required by Whisper and yt-dlp for audio processing
  - Windows: `choco install ffmpeg` or download from https://ffmpeg.org
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Setup & Installation

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **Note:** The first request will download the Whisper and BART models (~1.5 GB total). Subsequent requests will be faster.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm start
```

The app opens at **http://localhost:3000**.

## Usage

1. Open **http://localhost:3000** in your browser
2. Choose one of:
   - **Upload File** tab: Drag & drop or browse for an audio file
   - **Paste URL** tab: Enter a YouTube or Spotify podcast URL
3. Click **Summarize** and wait for processing
4. View the **Summary**, **Key Points**, and **Highlights** tabs
5. Expand **Full Transcript** to see the complete text
6. Click **Copy** to copy all results to clipboard

## API Endpoints

| Method | Endpoint             | Description                       |
| ------ | -------------------- | --------------------------------- |
| POST   | `/api/upload`        | Upload an audio file              |
| POST   | `/api/url`           | Submit a YouTube/Spotify URL      |
| GET    | `/api/status/{id}`   | Check processing job status       |

## Project Structure

```
podcast_summarizer/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app & routes
│   │   ├── transcriber.py   # Whisper transcription
│   │   ├── summarizer.py    # BART summarization
│   │   └── downloader.py    # yt-dlp audio download
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.js
│   │   │   ├── UploadSection.js
│   │   │   ├── StatusBar.js
│   │   │   └── ResultsPanel.js
│   │   ├── api.js
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.css
│   ├── .env
│   └── package.json
└── README.md
```

## Notes

- Processing time depends on podcast length (typically 1-5 min for a 30-min podcast)
- The Whisper `base` model balances speed and accuracy. For better accuracy, change to `small` or `medium` in `transcriber.py`
- GPU acceleration is supported — change `device=-1` to `device=0` in `summarizer.py` and remove `fp16=False` in `transcriber.py`
