#!/usr/bin/env python3
"""
Pi-Guy Speech Module - Local Dia2 TTS with face animation sync
Usage:
    python speak.py "Hello, I am Pi-Guy!"
    python speak.py --speaker S2 "Testing different speaker"
"""

import argparse
import os
import subprocess
import tempfile

import requests

# Face dashboard API
FACE_API = "http://localhost:5000/api"


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


def get_dia2_model():
    try:
        from dia2 import Dia2
    except ImportError as exc:
        raise RuntimeError("dia2 is not installed. Run: pip install dia2") from exc

    repo = os.environ.get("DIA2_REPO", "nari-labs/Dia2-2B")
    device = os.environ.get("DIA2_DEVICE")
    dtype = os.environ.get("DIA2_DTYPE")

    if not device:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if not dtype:
        dtype = "bfloat16" if device == "cuda" else "float32"

    return Dia2.from_repo(repo, device=device, dtype=dtype)


def speak(text, speaker="S1", cfg_scale=2.0, temperature=0.8, top_k=50, use_cuda_graph=False, output=None, play_audio=True, mood=None):
    """
    Speak text using Dia2 TTS with face animation.

    Args:
        text: Text to speak
        speaker: Speaker tag (S1 or S2)
        cfg_scale: Classifier-free guidance scale
        temperature: Sampling temperature
        top_k: Sampling top-k
        use_cuda_graph: Enable CUDA graph if supported
        output: Optional path to write WAV output
        play_audio: Play audio via aplay
        mood: Optional mood to set before speaking
    """
    if "[S1]" not in text and "[S2]" not in text:
        text = f"[{speaker}] {text}"

    if mood:
        notify_face("mood", mood=mood)

    notify_face("blink")
    notify_face("talk_start")

    try:
        from dia2 import GenerationConfig, SamplingConfig

        model = get_dia2_model()
        config = GenerationConfig(
            cfg_scale=cfg_scale,
            audio=SamplingConfig(temperature=temperature, top_k=top_k),
            use_cuda_graph=use_cuda_graph,
        )

        if output:
            output_path = output
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                output_path = wav_file.name

        try:
            model.generate(text, config=config, output_wav=output_path, verbose=False)
        finally:
            notify_face("talk_stop")

        if play_audio:
            subprocess.run(["aplay", output_path], check=False)

        notify_face("blink")

        return output_path
    finally:
        if not output and 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)


def main():
    parser = argparse.ArgumentParser(description="Pi-Guy TTS with Dia2")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--speaker", "-s", default="S1", help="Speaker tag (S1 or S2)")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG scale (default: 2.0)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50, help="Sampling top-k (default: 50)")
    parser.add_argument("--use-cuda-graph", action="store_true", help="Enable CUDA graph")
    parser.add_argument("--mood", help="Set face mood before speaking (happy, sad, angry, etc.)")
    parser.add_argument("--output", "-o", help="Write WAV output to file")
    parser.add_argument("--no-play", action="store_true", help="Skip playback (only generate WAV)")

    args = parser.parse_args()

    if not args.text:
        parser.print_help()
        print("\nExample:")
        print('  python speak.py "Hello, I am Pi-Guy!"')
        return

    speak(
        text=args.text,
        speaker=args.speaker,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cuda_graph=args.use_cuda_graph,
        output=args.output,
        play_audio=not args.no_play,
        mood=args.mood,
    )


if __name__ == "__main__":
    main()
