#!/usr/bin/env python3
"""
Pi-Guy Speech Module - ElevenLabs TTS with face animation sync
Usage:
    python speak.py "Hello, I am Pi-Guy!"
    python speak.py --voice "George" "Testing different voice"
    python speak.py --list-voices
"""

import os
import sys
import argparse
import requests
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play, stream

# API Key (set via environment variable or directly here)
API_KEY = os.environ.get('ELEVENLABS_API_KEY', '2a810134654d575673cb11ce019168644b8c4154d4f2398a7b1f740234afafba')

# Face dashboard API
FACE_API = "http://localhost:5000/api"

# Default voice settings
# Note: Custom voice "eZm9vdjYgL9PZKtf7XMM" was hitting API limits, using George as fallback
DEFAULT_VOICE = "George"  # Can override with --voice flag
DEFAULT_MODEL = "eleven_turbo_v2_5"  # Fast model, good for real-time

# Voice ID cache
VOICE_IDS = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "drew": "29vD33N1CtxCmqQRPOHJ",
    "clyde": "2EiwWnXFnvU5JabPnv8n",
    "paul": "5Q0t7uMcjvnagumLfvZi",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "dave": "CYw3kZ02Hs0563khs1Fj",
    "fin": "D38z5RcWu1voky8WS1ja",
    "sarah": "EXAVITQu4vr4xnSDxMaL",
    "antoni": "ErXwobaYiN019PkySvjV",
    "thomas": "GBv7mTt0atIp3Br8iCZE",
    "charlie": "IKne3meq5aSn9XLyUdCD",
    "george": "JBFqnCBsd6RMkjVDRZzb",
    "emily": "LcfcDJNUP1GQjkzn1xUU",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "callum": "N2lVS1w4EtoT3dr4eOWO",
    "patrick": "ODq5zmih8GrVes37Dizd",
    "harry": "SOYHLrjzK2X1ezoPC6cr",
    "liam": "TX3LPaxmHKxFdv7VOQHJ",
    "dorothy": "ThT5KcBeYPX3keUQqHPh",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "charlotte": "XB0fDUnXU5powFXDhCwa",
    "matilda": "XrExE9yKIg1WjnnlVkGX",
    "matthew": "Yko7PKs6WkxO6YG9LtNq",
    "james": "ZQe5CZNOzWyzPSCn5a3c",
    "joseph": "Zlb1dXrM653N07WRdFW3",
    "jeremy": "bVMeCyTHy58xNoL34h3p",
    "michael": "flq6f7yk4E4fJM5XTYuZ",
    "ethan": "g5CIjZEefAph4nQFvHAz",
    "gigi": "jBpfuIE2acCO8z3wKNLl",
    "freya": "jsCqWAovK2LkecY7zXl4",
    "grace": "oWAxZDx7w5VEj9dCyTzz",
    "daniel": "onwK4e9ZLuTAKqWW03F9",
    "serena": "pMsXgVXv3BLzUgSXRplE",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "nicole": "piTKgcLEGmPE4e6mEKli",
    "jessie": "t0jbNlBVZ17f02VDIeMI",
    "ryan": "wViXBPUzp2ZZixB1xQuM",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
    "glinda": "z9fAnlkpzviPz146aGWa",
    "giovanni": "zcAOhNBS3c14rBihAFp1",
    "mimi": "zrHiDhphv9ZnVXBqCLjz",
}


def notify_face(action, **kwargs):
    """Send command to face dashboard"""
    try:
        if action == "talk_start":
            requests.get(f"{FACE_API}/talk/start", timeout=1)
        elif action == "talk_stop":
            requests.get(f"{FACE_API}/talk/stop", timeout=1)
        elif action == "mood":
            requests.get(f"{FACE_API}/mood/{kwargs.get('mood', 'neutral')}", timeout=1)
        elif action == "blink":
            requests.get(f"{FACE_API}/blink", timeout=1)
    except requests.exceptions.RequestException:
        pass  # Face might not be running, that's okay


def get_voice_id(voice_name):
    """Get voice ID from name"""
    name_lower = voice_name.lower()
    if name_lower in VOICE_IDS:
        return VOICE_IDS[name_lower]
    # If it looks like an ID already, return it
    if len(voice_name) == 20:
        return voice_name
    # Default to George
    return VOICE_IDS["george"]


def list_voices():
    """List available voices"""
    print("\nAvailable voices:")
    print("-" * 40)
    for name in sorted(VOICE_IDS.keys()):
        print(f"  {name.capitalize()}")
    print("-" * 40)
    print(f"\nDefault: {DEFAULT_VOICE}")


def speak(text, voice=DEFAULT_VOICE, model=DEFAULT_MODEL, mood=None, streaming=True):
    """
    Speak text using ElevenLabs TTS with face animation

    Args:
        text: Text to speak
        voice: Voice name or ID
        model: Model ID to use
        mood: Optional mood to set before speaking
        streaming: Use streaming for lower latency
    """
    client = ElevenLabs(api_key=API_KEY)
    voice_id = get_voice_id(voice)

    # Set mood if specified
    if mood:
        notify_face("mood", mood=mood)

    # Blink before speaking (natural behavior)
    notify_face("blink")

    print(f"Speaking: {text[:50]}..." if len(text) > 50 else f"Speaking: {text}")

    # Start face talking animation
    notify_face("talk_start")

    try:
        if streaming:
            # Streaming mode - lower latency
            audio_stream = client.text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id=model,
                output_format="mp3_22050_32",
            )
            stream(audio_stream)
        else:
            # Non-streaming mode
            audio = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model,
                output_format="mp3_22050_32",
            )
            play(audio)
    finally:
        # Stop face talking animation
        notify_face("talk_stop")

    # Blink after speaking
    notify_face("blink")


def main():
    parser = argparse.ArgumentParser(description="Pi-Guy TTS with ElevenLabs")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--voice", "-v", default=DEFAULT_VOICE, help=f"Voice name (default: {DEFAULT_VOICE})")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--mood", help="Set face mood before speaking (happy, sad, angry, etc.)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (higher latency)")
    parser.add_argument("--list-voices", "-l", action="store_true", help="List available voices")

    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if not args.text:
        parser.print_help()
        print("\nExample:")
        print('  python speak.py "Hello, I am Pi-Guy!"')
        return

    speak(
        text=args.text,
        voice=args.voice,
        model=args.model,
        mood=args.mood,
        streaming=not args.no_stream
    )


if __name__ == "__main__":
    main()
