#!/usr/bin/env python3
"""
Pi-Guy Listen Module - Speech-to-Text with Whisper
Usage:
    python listen.py              # Listen and transcribe once
    python listen.py --continuous # Keep listening
    python listen.py --duration 5 # Record for 5 seconds
"""

import os
import sys
import argparse
import subprocess
import tempfile
import requests

# Face dashboard API
FACE_API = "http://localhost:5000/api"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DEVICE = "plughw:2,0"  # Blue Yeti


def notify_face(action, **kwargs):
    """Send command to face dashboard"""
    try:
        if action == "mood":
            requests.get(f"{FACE_API}/mood/{kwargs.get('mood', 'neutral')}", timeout=1)
        elif action == "blink":
            requests.get(f"{FACE_API}/blink", timeout=1)
    except requests.exceptions.RequestException:
        pass


def record_audio(duration=5, filename=None):
    """Record audio from microphone using arecord"""
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix='.wav')
        os.close(fd)

    print(f"Recording for {duration} seconds... Speak now!")
    notify_face("mood", mood="thinking")

    cmd = [
        'arecord',
        '-D', DEVICE,
        '-d', str(duration),
        '-f', 'S16_LE',
        '-r', str(SAMPLE_RATE),
        '-c', str(CHANNELS),
        filename
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("Recording complete.")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"Recording failed: {e}")
        return None
    finally:
        notify_face("mood", mood="neutral")


def transcribe_whisper_local(audio_file, model_size="tiny"):
    """Transcribe using local Whisper model"""
    try:
        import whisper

        print(f"Loading Whisper {model_size} model...")
        notify_face("mood", mood="thinking")

        model = whisper.load_model(model_size)

        print("Transcribing...")
        result = model.transcribe(audio_file)

        notify_face("mood", mood="neutral")
        notify_face("blink")

        return result["text"].strip()
    except ImportError:
        print("Whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def transcribe_whisper_api(audio_file, api_key=None):
    """Transcribe using OpenAI Whisper API (faster, requires API key)"""
    api_key = api_key or os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("No OpenAI API key found. Set OPENAI_API_KEY or use local mode.")
        return None

    print("Transcribing via OpenAI API...")
    notify_face("mood", mood="thinking")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        with open(audio_file, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )

        notify_face("mood", mood="neutral")
        notify_face("blink")

        return result.strip()
    except Exception as e:
        print(f"API transcription error: {e}")
        return None


def listen(duration=5, use_api=False, model_size="tiny"):
    """Record and transcribe speech - audio is deleted after transcription to save space"""
    # Record audio
    audio_file = record_audio(duration)

    if not audio_file:
        return None

    try:
        # Transcribe
        if use_api:
            text = transcribe_whisper_api(audio_file)
        else:
            text = transcribe_whisper_local(audio_file, model_size)

        return text
    finally:
        # Always clean up temp file to save disk space
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)


def main():
    parser = argparse.ArgumentParser(description="Pi-Guy Speech-to-Text")
    parser.add_argument("--duration", "-d", type=int, default=5,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--continuous", "-c", action="store_true",
                        help="Continuous listening mode")
    parser.add_argument("--api", action="store_true",
                        help="Use OpenAI Whisper API instead of local model")
    parser.add_argument("--model", "-m", default="tiny",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: tiny)")

    args = parser.parse_args()

    if args.continuous:
        print("Continuous listening mode. Press Ctrl+C to stop.")
        while True:
            try:
                text = listen(args.duration, args.api, args.model)
                if text:
                    print(f"\nYou said: {text}\n")
            except KeyboardInterrupt:
                print("\nStopped listening.")
                break
    else:
        text = listen(args.duration, args.api, args.model)
        if text:
            print(f"\nYou said: {text}")
        else:
            print("No speech detected or transcription failed.")


if __name__ == "__main__":
    main()
