#!/usr/bin/env python3
"""
Pi-Guy Speech Module - Local pluggable TTS with face animation sync
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

MOOD_PROFILES = {
    'neutral': {'temperature': 0.75, 'cfg_scale': 2.0, 'speaking_rate': 1.0},
    'happy': {'temperature': 0.95, 'cfg_scale': 1.8, 'speaking_rate': 1.1},
    'sad': {'temperature': 0.55, 'cfg_scale': 2.3, 'speaking_rate': 0.9},
    'angry': {'temperature': 0.7, 'cfg_scale': 2.6, 'speaking_rate': 1.15},
    'thinking': {'temperature': 0.6, 'cfg_scale': 2.2, 'speaking_rate': 0.95},
    'surprised': {'temperature': 0.9, 'cfg_scale': 1.9, 'speaking_rate': 1.05},
}

DEFAULT_TTS_BACKEND = os.environ.get("TTS_BACKEND", "dia2")
_dia2_model = None
_xtts_model = None


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
    global _dia2_model
    if _dia2_model is not None:
        return _dia2_model
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

    _dia2_model = Dia2.from_repo(repo, device=device, dtype=dtype)
    return _dia2_model


def get_xtts_model():
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError("XTTS backend unavailable. Install with: pip install TTS") from exc

    xtts_name = os.environ.get("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
    _xtts_model = TTS(model_name=xtts_name)
    return _xtts_model


def _generate_with_dia2(text, output_path, cfg_scale, temperature, top_k, use_cuda_graph, **_kwargs):
    from dia2 import GenerationConfig, SamplingConfig

    model = get_dia2_model()
    config = GenerationConfig(
        cfg_scale=cfg_scale,
        audio=SamplingConfig(temperature=temperature, top_k=top_k),
        use_cuda_graph=use_cuda_graph,
    )
    model.generate(text, config=config, output_wav=output_path, verbose=False)


def _generate_with_xtts(text, output_path, speaking_rate=1.0, **_kwargs):
    model = get_xtts_model()
    language = os.environ.get("XTTS_LANGUAGE", "en")
    speaker_wav = os.environ.get("XTTS_SPEAKER_WAV")
    try:
        model.tts_to_file(
            text=text,
            file_path=output_path,
            language=language,
            speaker_wav=speaker_wav,
            speed=speaking_rate,
        )
    except TypeError:
        model.tts_to_file(text=text, file_path=output_path, language=language, speaker_wav=speaker_wav)


def _generate_with_piper(text, output_path, speaking_rate=1.0, **_kwargs):
    model_path = os.environ.get("PIPER_MODEL")
    if not model_path:
        raise RuntimeError("Piper backend requires PIPER_MODEL to point to a voice model .onnx file")
    length_scale = 1.0 / max(0.5, min(2.0, speaking_rate))
    command = [
        "piper",
        "--model",
        model_path,
        "--output_file",
        output_path,
        "--length_scale",
        str(length_scale),
    ]
    subprocess.run(command, input=text.encode('utf-8'), check=True)


TTS_BACKENDS = {
    'dia2': _generate_with_dia2,
    'xtts': _generate_with_xtts,
    'piper': _generate_with_piper,
}


def speak(text, speaker="S1", cfg_scale=None, temperature=None, top_k=50, use_cuda_graph=False, output=None, play_audio=True, mood=None, backend=None, speaking_rate=None):
    """
    Speak text using a configurable TTS backend with face animation.

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

    mood_name = (mood or 'neutral').lower()
    mood_profile = MOOD_PROFILES.get(mood_name, MOOD_PROFILES['neutral'])

    if cfg_scale is None:
        cfg_scale = mood_profile['cfg_scale']
    if temperature is None:
        temperature = mood_profile['temperature']
    if speaking_rate is None:
        speaking_rate = mood_profile['speaking_rate']

    backend_name = (backend or DEFAULT_TTS_BACKEND).lower()
    generator = TTS_BACKENDS.get(backend_name)
    if generator is None:
        raise RuntimeError(f"Unsupported backend: {backend_name}")

    print(f"Applying mood profile '{mood_name}' with backend '{backend_name}': {mood_profile}")

    if output:
        output_path = output
    else:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            output_path = wav_file.name

    try:
        generator(
            text=text,
            output_path=output_path,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            use_cuda_graph=use_cuda_graph,
            speaking_rate=speaking_rate,
        )

        notify_face("mood", mood=mood_name)
        notify_face("blink")
        notify_face("talk_start")

        if play_audio:
            subprocess.run(["aplay", output_path], check=False)

        notify_face("talk_stop")
        notify_face("mood", mood='neutral')

        notify_face("blink")

        return output_path
    finally:
        if not output and 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)


def main():
    parser = argparse.ArgumentParser(description="Pi-Guy TTS with pluggable backends")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--speaker", "-s", default="S1", help="Speaker tag (S1 or S2)")
    parser.add_argument("--cfg-scale", type=float, help="CFG scale (defaults from mood profile)")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (defaults from mood profile)")
    parser.add_argument("--top-k", type=int, default=50, help="Sampling top-k (default: 50)")
    parser.add_argument("--use-cuda-graph", action="store_true", help="Enable CUDA graph")
    parser.add_argument("--mood", help="Emotion profile to apply (happy, sad, angry, thinking, etc.)")
    parser.add_argument("--speaking-rate", type=float, help="Speech speed multiplier (defaults from mood profile)")
    parser.add_argument("--backend", choices=sorted(TTS_BACKENDS.keys()), help="TTS backend override")
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
        backend=args.backend,
        speaking_rate=args.speaking_rate,
    )


if __name__ == "__main__":
    main()
