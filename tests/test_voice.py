"""Test script for the voice transcription client."""
import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.voice_client import transcribe_audio


def main():
    audio_file = "data/samples/test_clip.wav"

    # Check if the audio file exists
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("\nPlease add a test audio clip to data/samples/test_clip.wav before running this test.")
        return

    print("Testing voice transcription...")
    print("-" * 50)
    print(f"Audio file: {audio_file}\n")

    result = transcribe_audio(audio_file)
    print(f"Transcription:\n{result}")
    print("-" * 50)


if __name__ == "__main__":
    main()
