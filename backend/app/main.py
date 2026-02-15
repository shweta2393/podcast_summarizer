import os
import uuid
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.transcriber import transcribe_audio
from app.summarizer import generate_summary
from app.downloader import download_audio_from_url

app = FastAPI(title="Podcast Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "podcast_summarizer"
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job store
jobs: dict = {}


class URLRequest(BaseModel):
    url: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    result: dict | None = None
    error: str | None = None


def cleanup_file(path: str):
    """Remove temporary file after processing."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def process_audio(job_id: str, audio_path: str):
    """Background task: transcribe and summarize audio."""
    try:
        jobs[job_id]["status"] = "transcribing"

        # Step 1: Transcribe
        transcript = transcribe_audio(audio_path)

        jobs[job_id]["status"] = "summarizing"

        # Step 2: Summarize
        summary_result = generate_summary(transcript)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "transcript": transcript,
            **summary_result,
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        cleanup_file(audio_path)


@app.get("/")
def root():
    return {"message": "Podcast Summarizer API is running"}


@app.post("/api/upload", response_model=JobStatus)
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload an audio file (MP3/WAV) for summarization."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: MP3, WAV, M4A, OGG, FLAC, WEBM.",
        )

    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    jobs[job_id] = {"status": "processing", "result": None, "error": None}
    background_tasks.add_task(process_audio, job_id, str(file_path))

    return JobStatus(job_id=job_id, status="processing")


@app.post("/api/url", response_model=JobStatus)
async def process_url(request: URLRequest, background_tasks: BackgroundTasks):
    """Process a YouTube or Spotify URL."""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "downloading", "result": None, "error": None}

    async def download_and_process():
        try:
            audio_path = download_audio_from_url(url, str(UPLOAD_DIR), job_id)
            process_audio(job_id, audio_path)
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)

    background_tasks.add_task(download_and_process)

    return JobStatus(job_id=job_id, status="downloading")


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Check the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )
