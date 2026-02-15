"""Audio transcription using OpenAI Whisper."""

import whisper

# Load model once at module level
# "small" gives significantly better accuracy for music/lyrics than "base"
# Options: tiny, base, small, medium, large
_model = None


def _get_model():
    global _model
    if _model is None:
        print("Loading Whisper model (small)...")
        _model = whisper.load_model("small")
        print("Whisper model loaded.")
    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Full transcript text.
    """
    model = _get_model()
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        fp16=False,
        condition_on_previous_text=True,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    transcript = result["text"].strip()
    print(f"Transcription complete. Length: {len(transcript)} chars")
    return transcript
