"""Voice transcription client using faster-whisper."""
from faster_whisper import WhisperModel


# Global model variable for lazy loading
_model = None


def get_model():
    """Lazy-load the Whisper model."""
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file to text using faster-whisper.

    Args:
        file_path: Path to the audio file (WAV, MP3, etc.)

    Returns:
        Transcribed text with detected language information
    """
    model = get_model()
    segments, info = model.transcribe(file_path, beam_size=5)

    # Collect all segments into full text
    transcription = " ".join([segment.text for segment in segments])

    # Format response with language detection
    result = f"[Detected language: {info.language} (probability: {info.language_probability:.2f})]\n{transcription.strip()}"

    return result
