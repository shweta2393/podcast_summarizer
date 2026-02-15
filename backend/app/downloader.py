"""Download audio from YouTube/Spotify URLs using yt-dlp."""

import os
import re
import yt_dlp


def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube link."""
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch',
        r'(https?://)?(www\.)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/',
        r'(https?://)?music\.youtube\.com/',
    ]
    return any(re.match(p, url) for p in patterns)


def _is_spotify_url(url: str) -> bool:
    """Check if URL is a Spotify link."""
    return bool(re.match(r'(https?://)?open\.spotify\.com/', url))


def download_audio_from_url(url: str, output_dir: str, job_id: str) -> str:
    """
    Download audio from a YouTube or Spotify URL.

    Args:
        url: The video/podcast URL.
        output_dir: Directory to save the downloaded audio.
        job_id: Unique identifier for the job.

    Returns:
        Path to the downloaded audio file.
    """
    if not (_is_youtube_url(url) or _is_spotify_url(url)):
        # Try anyway – yt-dlp supports many sites
        print(f"URL may not be YouTube/Spotify, attempting download anyway: {url}")

    output_path = os.path.join(output_dir, f"{job_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'noplaylist': True,
    }

    print(f"Downloading audio from: {url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"Failed to download audio from URL: {e}")

    # Find the downloaded file
    expected_mp3 = os.path.join(output_dir, f"{job_id}.mp3")
    if os.path.exists(expected_mp3):
        print(f"Download complete: {expected_mp3}")
        return expected_mp3

    # Look for any file with the job_id
    for f in os.listdir(output_dir):
        if f.startswith(job_id):
            path = os.path.join(output_dir, f)
            print(f"Download complete: {path}")
            return path

    raise RuntimeError("Download completed but output file not found.")
