#!/usr/bin/env python3
"""
Pi-Guy Dashboard - Sci-Fi System Monitor
A futuristic dashboard for Raspberry Pi 5
"""

import base64
import logging
import uuid
import json
import math
import os
import psutil
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from schemas.realtime import build_realtime_turn_contract, parse_speech_directives

PIGUY_ENV = os.environ.get('PIGUY_ENV', 'dev').strip().lower()
if PIGUY_ENV not in {'dev', 'prod'}:
    raise RuntimeError("Invalid PIGUY_ENV. Use 'dev' or 'prod'.")

IS_PROD = PIGUY_ENV == 'prod'
SECRET_KEY = os.environ.get('SECRET_KEY')
if IS_PROD and not SECRET_KEY:
    raise RuntimeError('SECRET_KEY must be set when PIGUY_ENV=prod')
if not SECRET_KEY:
    SECRET_KEY = 'pi-guy-dev-only-secret-key'

SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get(
    'PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS',
    'http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,null' if not IS_PROD else '',
)
if not SOCKETIO_CORS_ALLOWED_ORIGINS:
    if IS_PROD:
        raise RuntimeError(
            'PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS is required when PIGUY_ENV=prod'
        )
    SOCKETIO_CORS_ALLOWED_ORIGINS = 'http://localhost:5000,http://127.0.0.1:5000'
SOCKETIO_CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in SOCKETIO_CORS_ALLOWED_ORIGINS.split(',')
    if origin.strip()
]

API_CORS_ALLOWED_ORIGINS = os.environ.get(
    'PIGUY_API_CORS_ALLOWED_ORIGINS',
    'http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,null' if not IS_PROD else '',
)
if not API_CORS_ALLOWED_ORIGINS:
    if IS_PROD:
        raise RuntimeError(
            'PIGUY_API_CORS_ALLOWED_ORIGINS is required when PIGUY_ENV=prod'
        )
    API_CORS_ALLOWED_ORIGINS = 'http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,null'
API_CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in API_CORS_ALLOWED_ORIGINS.split(',')
    if origin.strip()
]

API_KEY = os.environ.get('PIGUY_API_KEY', '').strip()
if IS_PROD and not API_KEY:
    raise RuntimeError('PIGUY_API_KEY must be set when PIGUY_ENV=prod')

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins=SOCKETIO_CORS_ALLOWED_ORIGINS)
_dia2_model = None
_dia2_lock = threading.Lock()

MOOD_PROFILES = {
    'neutral': {'temperature': 0.75, 'cfg_scale': 2.0, 'speaking_rate': 1.0},
    'happy': {'temperature': 0.95, 'cfg_scale': 1.8, 'speaking_rate': 1.1},
    'sad': {'temperature': 0.55, 'cfg_scale': 2.3, 'speaking_rate': 0.9},
    'angry': {'temperature': 0.7, 'cfg_scale': 2.6, 'speaking_rate': 1.15},
    'thinking': {'temperature': 0.6, 'cfg_scale': 2.2, 'speaking_rate': 0.95},
    'surprised': {'temperature': 0.9, 'cfg_scale': 1.9, 'speaking_rate': 1.05},
}

DEFAULT_TTS_BACKEND = os.environ.get("TTS_BACKEND", "dia2")
_xtts_model = None

DEFAULT_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.1:8b")
DEFAULT_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_AUDIO_DEVICE = os.environ.get("PIGUY_AUDIO_DEVICE", "default")
MODEL_SETTINGS_PATH = os.environ.get("PIGUY_MODEL_SETTINGS_PATH", "model-settings.json")
MODEL_API_KEY_ENV_VAR = "PIGUY_MODEL_API_KEY"
ALLOW_CDN_FALLBACK = os.environ.get("ALLOW_CDN_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

PROVIDER_PRESETS = [
    {"id": "ollama-local", "label": "Ollama Localhost", "api_base": "http://localhost:11434", "api_style": "ollama"},
    {"id": "lmstudio-local", "label": "LM Studio Localhost", "api_base": "http://localhost:1234", "api_style": "openai"},
    {"id": "openwebui-local", "label": "Open WebUI Localhost", "api_base": "http://localhost:3000", "api_style": "openai"},
    {"id": "openai", "label": "OpenAI", "api_base": "https://api.openai.com", "api_style": "openai"},
    {"id": "anthropic", "label": "Anthropic", "api_base": "https://api.anthropic.com", "api_style": "openai"},
    {"id": "google-gemini", "label": "Google Gemini", "api_base": "https://generativelanguage.googleapis.com", "api_style": "openai"},
    {"id": "groq", "label": "Groq", "api_base": "https://api.groq.com", "api_style": "openai"},
    {"id": "mistral", "label": "Mistral", "api_base": "https://api.mistral.ai", "api_style": "openai"},
    {"id": "together", "label": "Together", "api_base": "https://api.together.xyz", "api_style": "openai"},
]

DEFAULT_MODEL_SETTINGS = {
    "provider": "ollama-local",
    "api_base": DEFAULT_OLLAMA_HOST,
    "api_style": "ollama",
    "api_key": "",
    "text_model": DEFAULT_TEXT_MODEL,
    "vision_model": DEFAULT_VISION_MODEL,
    "fallback": {
        "enabled": True,
        "text_model": "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        "diffusion_model": "/static/vendor/transformers.min.js",
        "audio_model": "/static/vendor/transformers.min.js",
        "cdn_js_libs": [
            "local:/static/vendor/socket.io.min.js",
            "local:/static/vendor/transformers.min.js",
            "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2",
            "https://unpkg.com/@xenova/transformers@2.17.2",
            "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0",
        ],
        "strategy": "local_first_offline_preferred",
        "notes": "Prefer local /static/vendor JS libraries. CDN entries are optional and only used when ALLOW_CDN_FALLBACK=true.",
    },
}
_model_settings = None
_model_settings_lock = threading.Lock()
_model_secrets = {}


def _get_model_api_key():
    runtime_secret = _model_secrets.get("api_key")
    if isinstance(runtime_secret, str) and runtime_secret.strip():
        return runtime_secret.strip()
    return os.environ.get(MODEL_API_KEY_ENV_VAR, "").strip()


def _serialize_model_settings_for_response(settings):
    response_settings = dict(settings)
    response_settings.pop("api_key", None)
    response_settings["api_key_configured"] = bool(_get_model_api_key())
    return response_settings


def _apply_model_settings_payload(current, payload):
    if not isinstance(payload, dict):
        return current

    for key in ['provider', 'api_base', 'api_style', 'text_model', 'vision_model']:
        if key in payload and isinstance(payload[key], str):
            current[key] = payload[key].strip()

    fallback = payload.get('fallback')
    if isinstance(fallback, dict):
        for key in ['text_model', 'diffusion_model', 'audio_model', 'notes', 'strategy']:
            value = fallback.get(key)
            if isinstance(value, str):
                current['fallback'][key] = value.strip()
        enabled = fallback.get('enabled')
        if isinstance(enabled, bool):
            current['fallback']['enabled'] = enabled
        cdn_js_libs = fallback.get('cdn_js_libs')
        if isinstance(cdn_js_libs, list):
            current['fallback']['cdn_js_libs'] = [
                entry.strip() for entry in cdn_js_libs if isinstance(entry, str) and entry.strip()
            ]

    return current


def _apply_model_secret_payload(payload):
    if not isinstance(payload, dict):
        return

    candidate = None
    secrets = payload.get('secrets')
    if isinstance(secrets, dict) and isinstance(secrets.get('api_key'), str):
        candidate = secrets.get('api_key', '')
    elif isinstance(payload.get('api_key'), str):
        candidate = payload.get('api_key', '')

    if candidate is not None:
        _model_secrets['api_key'] = candidate.strip()


def require_api_key():
    if not API_KEY:
        return None
    provided_key = request.headers.get('X-API-Key', '').strip()
    if provided_key != API_KEY:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    return None


@app.before_request
def enforce_api_authentication():
    if request.path.startswith('/api/') and request.method == 'OPTIONS':
        return ('', 204)
    if request.path.startswith('/api/'):
        unauthorized = require_api_key()
        if unauthorized:
            return unauthorized
    return None


@app.after_request
def attach_cors_headers(response):
    if not request.path.startswith('/api/'):
        return response

    origin = request.headers.get('Origin', '').strip()
    allowed_origin = None
    if '*' in API_CORS_ALLOWED_ORIGINS:
        allowed_origin = '*'
    elif origin and origin in API_CORS_ALLOWED_ORIGINS:
        allowed_origin = origin
    elif origin == 'null' and 'null' in API_CORS_ALLOWED_ORIGINS:
        allowed_origin = 'null'

    if allowed_origin:
        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Vary'] = 'Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,PATCH,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,X-API-Key'

    return response


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


def _wav_duration_seconds(path):
    try:
        with wave.open(path, 'rb') as wav_file:
            frame_rate = wav_file.getframerate() or 1
            frame_count = wav_file.getnframes()
            return frame_count / float(frame_rate)
    except Exception:
        return 2.5


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


def _schedule_face_reset(mood, delay_seconds):
    def _reset():
        try:
            socketio.emit('set_mood', {'mood': 'neutral'})
            socketio.emit('stop_talking')
        except Exception:
            app.logger.warning("Failed to reset face state after mood '%s' playback", mood)

    timer = threading.Timer(max(0.2, delay_seconds), _reset)
    timer.daemon = True
    timer.start()


def _openai_compatible_request(path, payload, api_base, api_key=""):
    data = json.dumps(payload).encode("utf-8")
    url = api_base.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_obj = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request_obj, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"Model API request failed: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Model API is not reachable. Check provider endpoint settings.") from exc


def _ollama_request(path, payload, api_base=None):
    data = json.dumps(payload).encode("utf-8")
    url = (api_base or DEFAULT_OLLAMA_HOST).rstrip("/") + path
    request_obj = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"Ollama request failed: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama is not reachable. Set OLLAMA_HOST if needed.") from exc


def get_model_settings():
    global _model_settings
    if _model_settings is not None:
        return _model_settings

    settings = dict(DEFAULT_MODEL_SETTINGS)
    settings["fallback"] = dict(DEFAULT_MODEL_SETTINGS["fallback"])

    if os.path.exists(MODEL_SETTINGS_PATH):
        try:
            with open(MODEL_SETTINGS_PATH, "r", encoding="utf-8") as file_obj:
                persisted = json.load(file_obj)
            if isinstance(persisted, dict):
                safe_persisted = dict(persisted)
                safe_persisted.pop("api_key", None)
                settings.update({k: v for k, v in safe_persisted.items() if k != "fallback"})
                if isinstance(safe_persisted.get("fallback"), dict):
                    settings["fallback"].update(safe_persisted["fallback"])
        except Exception:
            app.logger.warning("Unable to load model settings from %s", MODEL_SETTINGS_PATH)

    _model_settings = settings
    return _model_settings


def save_model_settings(settings):
    persisted = dict(settings)
    persisted.pop("api_key", None)
    with open(MODEL_SETTINGS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(persisted, file_obj, indent=2)


def _chat_completion(messages, model, settings=None):
    model_settings = settings or get_model_settings()
    api_style = model_settings.get("api_style", "ollama")
    api_base = model_settings.get("api_base", DEFAULT_OLLAMA_HOST)
    api_key = _get_model_api_key()

    try:
        if api_style == "openai":
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            response = _openai_compatible_request("/v1/chat/completions", payload, api_base=api_base, api_key=api_key)
            choices = response.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return message.get("content", "")

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        response = _ollama_request("/api/chat", payload, api_base=api_base)
        message = response.get("message", {})
        return message.get("content", "")
    except RuntimeError as exc:
        fallback = model_settings.get("fallback") or {}
        if fallback.get("enabled", True):
            return _fallback_chat_completion(messages, model_settings, exc)
        raise


def _fallback_chat_completion(messages, model_settings, error):
    fallback = model_settings.get("fallback") or {}
    user_turns = [m.get("content", "").strip() for m in messages if m.get("role") == "user"]
    latest_user = user_turns[-1] if user_turns else "I am ready for your next instruction."
    brief = re.sub(r"\s+", " ", latest_user)[:220]
    js_libs = fallback.get("cdn_js_libs") or []
    top_lib = js_libs[0] if js_libs else fallback.get("audio_model") or "local rule stack"

    return (
        "Primary model endpoint is unavailable, so I switched to fallback reasoning mode "
        f"for sanity checks. I understood your request as: '{brief}'. "
        "I can still coordinate avatar/dashboard actions, validate comprehension, and keep the "
        "orchestration loop alive while model services recover. "
        f"Fallback stack: {top_lib}. Error: {str(error)}"
    )



class RealtimeRAGOrchestrator:
    """Lightweight multi-tier realtime RAG + agent orchestration."""

    def __init__(
        self,
        session_ttl_seconds=1800,
        cleanup_interval_seconds=60,
        max_history_items=32,
        max_memory_notes=48,
        max_tool_events=48,
        max_user_text_chars=4000,
        max_reply_chars=4000,
        max_modality_json_chars=3000,
    ):
        self.sessions = {}
        self.lock = threading.Lock()
        self.session_ttl_seconds = max(60, int(session_ttl_seconds))
        self.cleanup_interval_seconds = max(1, int(cleanup_interval_seconds))
        self.max_history_items = max(4, int(max_history_items))
        self.max_memory_notes = max(1, int(max_memory_notes))
        self.max_tool_events = max(1, int(max_tool_events))
        self.max_user_text_chars = max(64, int(max_user_text_chars))
        self.max_reply_chars = max(128, int(max_reply_chars))
        self.max_modality_json_chars = max(256, int(max_modality_json_chars))
        self._last_cleanup_at = 0.0
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'cleanup_runs': 0,
            'expired_sessions_total': 0,
            'active_sessions': 0,
            'truncated_user_payloads': 0,
            'truncated_modality_payloads': 0,
            'truncated_reply_payloads': 0,
            'last_cleanup_evictions': 0,
        }

    def _append_bounded(self, values, item, max_len):
        values.append(item)
        overflow = len(values) - max_len
        if overflow > 0:
            del values[:overflow]

    def _record_active_sessions(self):
        self.metrics['active_sessions'] = len(self.sessions)

    def _cleanup_expired_sessions(self, now=None, force=False):
        now = now if now is not None else time.time()
        with self.lock:
            if not force and (now - self._last_cleanup_at) < self.cleanup_interval_seconds:
                return 0

            expired = []
            for session_id, session in self.sessions.items():
                last_active = session.get('last_active_at', session.get('created_at', now))
                if (now - last_active) > self.session_ttl_seconds:
                    expired.append(session_id)

            for session_id in expired:
                self.sessions.pop(session_id, None)

            self._last_cleanup_at = now
            self.metrics['cleanup_runs'] += 1
            self.metrics['expired_sessions_total'] += len(expired)
            self.metrics['last_cleanup_evictions'] = len(expired)
            self._record_active_sessions()

        if expired:
            self.logger.info(
                'Realtime session cleanup evicted %s inactive sessions (active=%s)',
                len(expired),
                self.metrics['active_sessions'],
            )
        return len(expired)

    def _sanitize_modality(self, modality):
        if not isinstance(modality, dict) or not modality:
            return {}

        normalized = {}
        for key, value in modality.items():
            if len(normalized) >= 12:
                break
            if not isinstance(key, str):
                continue

            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key[:64]] = value if not isinstance(value, str) else value[:320]
                continue

            if isinstance(value, list):
                compact = []
                for entry in value[:8]:
                    if isinstance(entry, (str, int, float, bool)) or entry is None:
                        compact.append(entry if not isinstance(entry, str) else entry[:180])
                normalized[key[:64]] = compact
                continue

            if isinstance(value, dict):
                compact = {}
                for inner_key, inner_value in list(value.items())[:8]:
                    if not isinstance(inner_key, str):
                        continue
                    if isinstance(inner_value, (str, int, float, bool)) or inner_value is None:
                        compact[inner_key[:64]] = inner_value if not isinstance(inner_value, str) else inner_value[:180]
                normalized[key[:64]] = compact

        encoded = json.dumps(normalized)
        if len(encoded) > self.max_modality_json_chars:
            self.metrics['truncated_modality_payloads'] += 1
            return {'warning': 'modality_payload_truncated'}
        return normalized

    def get_metrics(self):
        self._cleanup_expired_sessions()
        with self.lock:
            snapshot = dict(self.metrics)
            snapshot['active_sessions'] = len(self.sessions)
            snapshot['session_ttl_seconds'] = self.session_ttl_seconds
            snapshot['cleanup_interval_seconds'] = self.cleanup_interval_seconds
            return snapshot

    def start_session(self, profile='default'):
        self._cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                'profile': profile,
                'created_at': time.time(),
                'last_active_at': time.time(),
                'history': [],
                'memory_notes': [],
                'tool_events': [],
                'behavior_state': {
                    'warmth': 0.55,
                    'directness': 0.5,
                    'energy': 0.45,
                    'reflectiveness': 0.5,
                },
                'task_progress': 0,
            }
            self._record_active_sessions()
        return session_id

    def _get_session(self, session_id):
        self._cleanup_expired_sessions()
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None

            now = time.time()
            last_active = session.get('last_active_at', session.get('created_at', now))
            if (now - last_active) > self.session_ttl_seconds:
                self.sessions.pop(session_id, None)
                self.metrics['expired_sessions_total'] += 1
                self._record_active_sessions()
                self.logger.info('Realtime session %s expired on access.', session_id)
                return None
            return session

    def _resolve_affect(self, text):
        value = (text or '').lower()
        signal = {
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'certainty': 0.5,
        }

        affect_lexicon = [
            (['angry', 'mad', 'furious', 'rage', 'annoyed'], {'valence': -0.85, 'arousal': 0.85, 'dominance': 0.65, 'certainty': 0.65}),
            (['sad', 'sorry', 'down', 'error', 'fail', 'upset'], {'valence': -0.75, 'arousal': -0.35, 'dominance': -0.45, 'certainty': 0.6}),
            (['think', 'why', 'how', 'consider', 'analyze', 'maybe', 'perhaps'], {'valence': 0.05, 'arousal': 0.05, 'dominance': -0.1, 'certainty': -0.2}),
            (['wow', 'amazing', 'surprising', 'unbelievable'], {'valence': 0.3, 'arousal': 0.95, 'dominance': 0.1, 'certainty': 0.1}),
            (['great', 'good', 'love', 'happy', 'awesome', 'nice'], {'valence': 0.9, 'arousal': 0.5, 'dominance': 0.4, 'certainty': 0.45}),
            (['sure', 'definitely', 'certain', 'clear'], {'certainty': 0.5, 'dominance': 0.2}),
            (['unsure', 'uncertain', 'guess', 'possibly'], {'certainty': -0.55, 'dominance': -0.25}),
        ]

        match_count = 0
        for tokens, contribution in affect_lexicon:
            if any(token in value for token in tokens):
                match_count += 1
                for key, amount in contribution.items():
                    signal[key] += amount

        punctuation_intensity = min(1.0, value.count('!') * 0.18)
        if punctuation_intensity:
            signal['arousal'] += punctuation_intensity
            signal['certainty'] += punctuation_intensity * 0.2

        certainty_floor = 0.45 if match_count == 0 else 0.35
        affect_vector = {
            'valence': max(-1.0, min(1.0, signal['valence'])),
            'arousal': max(-1.0, min(1.0, signal['arousal'])),
            'dominance': max(-1.0, min(1.0, signal['dominance'])),
            'certainty': max(0.0, min(1.0, certainty_floor + signal['certainty'] * 0.4)),
        }

        if affect_vector['valence'] <= -0.45 and affect_vector['arousal'] > 0.35:
            primary_mood = 'angry'
        elif affect_vector['valence'] <= -0.35 and affect_vector['arousal'] <= 0.25:
            primary_mood = 'sad'
        elif affect_vector['arousal'] >= 0.7 and affect_vector['valence'] >= 0.1:
            primary_mood = 'surprised'
        elif affect_vector['valence'] >= 0.45:
            primary_mood = 'happy'
        elif affect_vector['certainty'] < 0.4 or ('why' in value or 'how' in value):
            primary_mood = 'thinking'
        else:
            primary_mood = 'neutral'

        return {
            'primary_mood': primary_mood,
            'affect_vector': affect_vector,
        }

    def _classify_mood(self, text):
        return self._resolve_affect(text)['primary_mood']

    def _smooth_state(self, previous, target, inertia=0.85):
        blended = {}
        for key, old_value in previous.items():
            next_value = target.get(key, old_value)
            blended[key] = round((inertia * old_value) + ((1.0 - inertia) * next_value), 3)
        return blended

    def _build_sequential_task_list(self, progress_index):
        task_steps = [
            'Perceptual layer: normalize language, tone, and multimodal cues into a shared observation object.',
            'Appraisal layer: map observations into continuous affect and social context dimensions.',
            'Cognitive layer: blend deliberative, social, and goal reasoning into candidate intents.',
            'Self/identity layer: update style posture with gradual state transitions (hysteresis).',
            'Action policy layer: mix conversational actions (ask, explain, reassure, challenge) by weighted policy.',
            'Expressive layer: render reply structure, cadence, and tone from the current behavior state.',
        ]
        active_index = max(0, min(progress_index, len(task_steps) - 1))
        task_list = []
        for index, step in enumerate(task_steps):
            status = 'pending'
            if index < active_index:
                status = 'completed'
            elif index == active_index:
                status = 'in_progress'
            task_list.append({'step': step, 'status': status})
        return task_list

    def _perceptual_layer(self, user_text, modality, tier_memory):
        text = (user_text or '').strip()
        modality = modality or {}
        tokens = text.split()
        observation = {
            'text': text,
            'token_count': len(tokens),
            'question_density': round(min(1.0, text.count('?') / 2.0), 3),
            'punctuation_energy': round(min(1.0, text.count('!') * 0.2), 3),
            'memory_salience': round(min(1.0, len(tier_memory.get('recent_context', [])) / 6.0), 3),
            'modality_present': sorted([key for key, value in modality.items() if value]),
        }
        observation['uncertainty'] = round(max(0.05, 1.0 - min(1.0, len(tokens) / 30.0)), 3)
        return observation

    def _appraisal_layer(self, observation, affect):
        affect_vector = affect['affect_vector']
        novelty = min(1.0, (observation['token_count'] / 24.0) + (observation['question_density'] * 0.2))
        appraisal = {
            'safety': round((affect_vector['valence'] + 1.0) / 2.0, 3),
            'urgency': round(max(observation['punctuation_energy'], (affect_vector['arousal'] + 1.0) / 2.0), 3),
            'novelty': round(novelty, 3),
            'social_closeness': round(0.45 + (0.35 * max(0.0, affect_vector['valence'])), 3),
            'certainty': round(affect_vector['certainty'], 3),
        }
        return appraisal

    def _cognitive_layer(self, user_text, appraisal, tier_tools):
        text = (user_text or '').lower()
        planning_weight = 0.65 if len(text.split()) > 24 else 0.35
        intent = 'supportive_explanation'
        if '?' in text:
            intent = 'clarifying_answer'
        if tier_tools:
            intent = 'tool_augmented_guidance'
        cognition = {
            'intent': intent,
            'deliberative_weight': round(min(1.0, planning_weight + appraisal['novelty'] * 0.2), 3),
            'social_weight': round(min(1.0, 0.35 + appraisal['social_closeness'] * 0.4), 3),
            'goal': 'maintain coherent digital-being behavior while advancing user request',
        }
        return cognition

    def _self_identity_layer(self, session, appraisal, cognition):
        previous_state = session.get('behavior_state', {})
        target_state = {
            'warmth': min(1.0, 0.35 + appraisal['social_closeness'] * 0.6),
            'directness': min(1.0, 0.25 + cognition['deliberative_weight'] * 0.7),
            'energy': min(1.0, 0.25 + appraisal['urgency'] * 0.65),
            'reflectiveness': min(1.0, 0.3 + appraisal['novelty'] * 0.55),
        }
        updated_state = self._smooth_state(previous_state, target_state, inertia=0.85)
        session['behavior_state'] = updated_state
        return {'previous': previous_state, 'target': target_state, 'current': updated_state}

    def _action_policy_layer(self, cognition, identity_state, tier_tools):
        current = identity_state['current']
        policy_mix = {
            'exploratory': round(min(1.0, 0.2 + current['reflectiveness'] * 0.5), 3),
            'supportive': round(min(1.0, 0.25 + current['warmth'] * 0.5), 3),
            'instructional': round(min(1.0, 0.2 + current['directness'] * 0.6), 3),
        }
        total = sum(policy_mix.values()) or 1.0
        normalized_mix = {key: round(value / total, 3) for key, value in policy_mix.items()}
        return {
            'intent': cognition['intent'],
            'policy_mix': normalized_mix,
            'tools_requested': tier_tools,
        }

    def _expressive_layer(self, mood, identity_state, action_policy):
        current = identity_state['current']
        return {
            'mood_hint': mood,
            'tone': {
                'warmth': current['warmth'],
                'directness': current['directness'],
                'energy': current['energy'],
            },
            'cadence': 'measured' if current['energy'] < 0.55 else 'animated',
            'policy_blend': action_policy['policy_mix'],
        }

    def _retrieve_memory(self, session, user_text):
        recent = session['history'][-6:]
        memory_lines = []
        for item in recent:
            memory_lines.append(f"{item['role']}: {item['content'][:180]}")
        modality_lines = []
        for item in session['history'][-8:]:
            modality = item.get('modality') or {}
            if modality.get('vision'):
                modality_lines.append(f"vision: {str(modality['vision'])[:140]}")
            if modality.get('audio'):
                modality_lines.append(f"audio: {str(modality['audio'])[:140]}")
        retrieval = {
            'recent_context': memory_lines,
            'memory_notes': session['memory_notes'][-6:],
            'modality_notes': modality_lines[-6:],
            'query': user_text,
        }
        return retrieval

    def _tool_router(self, user_text):
        text = (user_text or '').lower()
        tools = []
        if any(word in text for word in ['cpu', 'memory', 'disk', 'temp', 'system']):
            tools.append('system_stats')
        if any(word in text for word in ['see', 'vision', 'camera', 'image']):
            tools.append('vision_model')
        if any(word in text for word in ['speak', 'voice', 'say out loud']):
            tools.append('tts')
        return tools

    def _build_thought_events(self, user_text, reply, mood, tools):
        """Build lightweight internal-thought snippets for frontend cloud visualization."""
        user_text = (user_text or '').strip()
        reply = (reply or '').strip()
        tags = []
        if mood:
            tags.append(mood)
        if len(user_text.split()) > 16:
            tags.append('long-input')
        if '?' in user_text:
            tags.append('question')
        if tools:
            tags.append('tool-aware')

        confidence = 0.86
        if mood in ['sad', 'thinking']:
            confidence = 0.68
        if not reply:
            confidence = 0.42

        primary_intensity = min(1.0, 0.32 + (0.15 * len(tools)) + (0.2 if mood in ['surprised', 'angry'] else 0))
        followup_intensity = min(1.0, max(0.14, primary_intensity - 0.2))

        events = [
            {
                'text': f"Intent map: {' + '.join(tools) if tools else 'conversation-first'}",
                'emotion_tags': (tags + ['intent'])[:4],
                'emoji': '🧭',
                'lifetime_ms': 1900,
                'intensity': round(primary_intensity, 2),
            },
            {
                'text': f"Confidence check: {int(confidence * 100)}%",
                'emotion_tags': [mood or 'neutral', 'confidence'],
                'emoji': '📈' if confidence >= 0.75 else '🫧',
                'lifetime_ms': 1650,
                'intensity': round(followup_intensity, 2),
            },
        ]

        if user_text:
            events.append({
                'text': f"Focus: {user_text[:44]}{'…' if len(user_text) > 44 else ''}",
                'emotion_tags': [mood or 'neutral', 'intent', 'context'],
                'emoji': '💭',
                'lifetime_ms': 2100,
                'intensity': round(min(0.95, primary_intensity + 0.1), 2),
            })

        return events[:4]

    def run_turn(self, session_id, user_text, model=DEFAULT_TEXT_MODEL, modality=None):
        session = self._get_session(session_id)
        if session is None:
            raise RuntimeError('Invalid realtime session')
        user_text = str(user_text or '').strip()
        if len(user_text) > self.max_user_text_chars:
            user_text = user_text[:self.max_user_text_chars]
            self.metrics['truncated_user_payloads'] += 1
        modality = self._sanitize_modality(modality)

        tier_memory = self._retrieve_memory(session, user_text)
        tier_tools = self._tool_router(user_text)
        tier_skill_hints = [
            'conversation',
            'planner' if len(user_text.split()) > 20 else 'quick-response',
            'diagnostics' if 'system_stats' in tier_tools else 'creative',
        ]

        perception = self._perceptual_layer(user_text, modality, tier_memory)
        seed_affect = self._resolve_affect(user_text)
        appraisal = self._appraisal_layer(perception, seed_affect)
        cognition = self._cognitive_layer(user_text, appraisal, tier_tools)
        identity = self._self_identity_layer(session, appraisal, cognition)
        action_policy = self._action_policy_layer(cognition, identity, tier_tools)
        expression_plan = self._expressive_layer(seed_affect['primary_mood'], identity, action_policy)

        system_context = {
            'memory': tier_memory,
            'tools': tier_tools,
            'skills': tier_skill_hints,
            'digital_being': {
                'perception': perception,
                'appraisal': appraisal,
                'cognition': cognition,
                'identity': identity['current'],
                'action_policy': action_policy,
                'expression_plan': expression_plan,
            },
        }

        messages = [
            {
                'role': 'system',
                'content': (
                    'You are Pi-Guy, a realtime multimodal avatar. '
                    'Use concise but expressive language, keep emotional alignment with user tone, '
                    'and optionally mention how tool tiers could help.\n'
                    f'Realtime context: {json.dumps(system_context)}'
                ),
            },
        ]

        for item in session['history'][-8:]:
            messages.append({'role': item['role'], 'content': item['content']})

        messages.append({'role': 'user', 'content': user_text})
        if modality:
            messages.append({
                'role': 'system',
                'content': f"Multimodal companion context: {json.dumps(modality)}",
            })
        reply = _chat_completion(messages, model)
        if len(reply or '') > self.max_reply_chars:
            reply = (reply or '')[:self.max_reply_chars]
            self.metrics['truncated_reply_payloads'] += 1
        affect = self._resolve_affect(reply or user_text)
        mood = affect['primary_mood']

        with self.lock:
            self._append_bounded(
                session['history'],
                {'role': 'user', 'content': user_text, 'modality': modality},
                self.max_history_items,
            )
            self._append_bounded(
                session['history'],
                {'role': 'assistant', 'content': reply, 'modality': {'mood': mood, 'affect_vector': affect['affect_vector']}},
                self.max_history_items,
            )
            self._append_bounded(
                session['tool_events'],
                {'ts': time.time(), 'tools': tier_tools},
                self.max_tool_events,
            )
            if user_text and len(user_text) > 24:
                self._append_bounded(session['memory_notes'], user_text[:220], self.max_memory_notes)
            session['task_progress'] = min(5, session.get('task_progress', 0) + 1)
            session['last_active_at'] = time.time()

        payload = build_realtime_turn_contract(
            reply=reply,
            mood=mood,
            layers={
                'retrieval': tier_memory,
                'tool_router': tier_tools,
                'skill_hints': tier_skill_hints,
                'synthesis_model': model,
                'digital_being': {
                    'perception': perception,
                    'appraisal': appraisal,
                    'cognition': cognition,
                    'identity': identity,
                    'action_policy': action_policy,
                    'expression': expression_plan,
                },
                'sequential_task_list': self._build_sequential_task_list(session.get('task_progress', 0)),
            },
        )
        payload['thought_events'] = self._build_thought_events(user_text, reply, mood, tier_tools)
        payload['affect_vector'] = affect['affect_vector']
        return payload

    def state(self, session_id):
        session = self._get_session(session_id)
        if session is None:
            return None
        return {
            'profile': session['profile'],
            'turns': len(session['history']) // 2,
            'memory_notes': len(session['memory_notes']),
            'recent_tools': session['tool_events'][-5:],
        }


orchestrator = RealtimeRAGOrchestrator()
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json() or {}
    settings = get_model_settings()
    messages = data.get('messages', [])
    model = data.get('model', settings.get('text_model', DEFAULT_TEXT_MODEL))
    if not messages:
        return jsonify({'status': 'error', 'message': 'No messages provided'}), 400
    try:
        content = _chat_completion(messages, model, settings=settings)
        return jsonify({'status': 'ok', 'message': content, 'model': model})
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/vision', methods=['POST'])
def api_vision():
    data = request.get_json() or {}
    settings = get_model_settings()
    prompt = data.get('prompt', '').strip()
    image = data.get('image')
    model = data.get('model', settings.get('vision_model', DEFAULT_VISION_MODEL))
    if not prompt or not image:
        return jsonify({'status': 'error', 'message': 'Prompt and image required'}), 400
    try:
        messages = [{
            'role': 'user',
            'content': prompt,
            'images': [image],
        }]
        content = _chat_completion(messages, model, settings=settings)
        return jsonify({'status': 'ok', 'message': content, 'model': model})
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500



@app.route('/api/realtime/session/start', methods=['POST'])
def api_realtime_start():
    data = request.get_json() or {}
    profile = data.get('profile', 'default')
    session_id = orchestrator.start_session(profile=profile)
    return jsonify({'status': 'ok', 'session_id': session_id, 'profile': profile})


@app.route('/api/realtime/turn', methods=['POST'])
def api_realtime_turn():
    data = request.get_json() or {}
    settings = get_model_settings()
    session_id = data.get('session_id')
    user_text = (data.get('text') or '').strip()
    modality = data.get('modality') if isinstance(data.get('modality'), dict) else {}
    model = data.get('model', settings.get('text_model', DEFAULT_TEXT_MODEL))
    if not session_id or not user_text:
        return jsonify({'status': 'error', 'message': 'session_id and text required'}), 400
    try:
        result = orchestrator.run_turn(session_id, user_text, model=model, modality=modality)
        return jsonify({'status': 'ok', **result})
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/realtime/state')
def api_realtime_state():
    session_id = request.args.get('session_id', '').strip()
    if not session_id:
        return jsonify({'status': 'error', 'message': 'session_id required'}), 400
    snapshot = orchestrator.state(session_id)
    if snapshot is None:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 404
    return jsonify({'status': 'ok', 'state': snapshot})


@app.route('/api/realtime/metrics')
def api_realtime_metrics():
    return jsonify({'status': 'ok', 'metrics': orchestrator.get_metrics()})


@app.route('/api/settings/models', methods=['GET', 'POST'])
def api_model_settings():
    global _model_settings
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'settings': _serialize_model_settings_for_response(get_model_settings()),
            'provider_presets': PROVIDER_PRESETS,
        })

    data = request.get_json() or {}
    with _model_settings_lock:
        current = dict(get_model_settings())
        current['fallback'] = dict(current.get('fallback', {}))
        _apply_model_settings_payload(current, data)
        _apply_model_secret_payload(data)

        _model_settings = current
        save_model_settings(_model_settings)

    return jsonify({'status': 'ok', 'settings': _serialize_model_settings_for_response(_model_settings)})

def get_cpu_temp():
    """Get CPU temperature from thermal zone"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000
            return round(temp, 1)
    except:
        return 0

def get_system_stats():
    """Gather all system statistics"""
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Network stats
    net = psutil.net_io_counters()

    # CPU frequency
    try:
        cpu_freq = psutil.cpu_freq()
        freq_current = cpu_freq.current if cpu_freq else 0
    except:
        freq_current = 0

    return {
        'cpu': {
            'percent': cpu_percent,
            'temp': get_cpu_temp(),
            'freq': round(freq_current, 0),
            'cores': psutil.cpu_count()
        },
        'memory': {
            'percent': memory.percent,
            'used': round(memory.used / (1024**3), 2),
            'total': round(memory.total / (1024**3), 2)
        },
        'disk': {
            'percent': disk.percent,
            'used': round(disk.used / (1024**3), 1),
            'total': round(disk.total / (1024**3), 1)
        },
        'network': {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv
        }
    }


def _get_top_processes(limit=5):
    process_rows = []
    for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info', 'cmdline']):
        try:
            info = process.info
            memory_info = info.get('memory_info')
            process_rows.append({
                'pid': info.get('pid'),
                'name': info.get('name') or 'unknown',
                'cpu_percent': round(info.get('cpu_percent') or 0.0, 1),
                'memory_percent': round(info.get('memory_percent') or 0.0, 2),
                'rss_mb': round((memory_info.rss if memory_info else 0) / (1024**2), 2),
                'cmdline': ' '.join(info.get('cmdline') or []),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return sorted(
        process_rows,
        key=lambda row: (row['cpu_percent'], row['memory_percent']),
        reverse=True,
    )[: max(1, limit)]


def build_performance_snapshot(duration_seconds=8.0, interval_seconds=1.0, top_limit=5):
    duration_seconds = max(1.0, min(30.0, float(duration_seconds)))
    interval_seconds = max(0.2, min(5.0, float(interval_seconds)))
    top_limit = max(1, min(20, int(top_limit)))

    sample_count = max(1, int(math.ceil(duration_seconds / interval_seconds)))
    sampled_stats = []
    first_sample_ts = time.time()

    for index in range(sample_count):
        sampled_stats.append(get_system_stats())
        if index < sample_count - 1:
            time.sleep(interval_seconds)

    last_sample_ts = time.time()
    elapsed_seconds = max(last_sample_ts - first_sample_ts, 0.001)

    cpu_values = [sample['cpu']['percent'] for sample in sampled_stats]
    memory_values = [sample['memory']['percent'] for sample in sampled_stats]
    temp_values = [sample['cpu']['temp'] for sample in sampled_stats]

    network_start = sampled_stats[0]['network']
    network_end = sampled_stats[-1]['network']

    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except (AttributeError, OSError):
        load_1m, load_5m, load_15m = 0.0, 0.0, 0.0

    return {
        'window': {
            'duration_seconds': round(elapsed_seconds, 2),
            'interval_seconds': interval_seconds,
            'sample_count': sample_count,
        },
        'cpu': {
            'avg_percent': round(sum(cpu_values) / len(cpu_values), 2),
            'peak_percent': max(cpu_values),
            'avg_temp': round(sum(temp_values) / len(temp_values), 2),
            'peak_temp': max(temp_values),
            'load_avg': {
                '1m': round(load_1m, 2),
                '5m': round(load_5m, 2),
                '15m': round(load_15m, 2),
            },
        },
        'memory': {
            'avg_percent': round(sum(memory_values) / len(memory_values), 2),
            'peak_percent': max(memory_values),
            'latest': sampled_stats[-1]['memory'],
        },
        'disk': {
            'latest': sampled_stats[-1]['disk'],
        },
        'network': {
            'bytes_sent_delta': network_end['bytes_sent'] - network_start['bytes_sent'],
            'bytes_recv_delta': network_end['bytes_recv'] - network_start['bytes_recv'],
            'bytes_sent_per_sec': round((network_end['bytes_sent'] - network_start['bytes_sent']) / elapsed_seconds, 2),
            'bytes_recv_per_sec': round((network_end['bytes_recv'] - network_start['bytes_recv']) / elapsed_seconds, 2),
        },
        'top_processes': _get_top_processes(limit=top_limit),
        'samples': sampled_stats,
    }

# Background thread for pushing updates
def background_stats():
    """Push system stats every second"""
    while True:
        stats = get_system_stats()
        socketio.emit('stats_update', stats)
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html', allow_cdn_fallback=ALLOW_CDN_FALLBACK)


@app.route('/<path:path>')
def spa_fallback(path):
    if path.startswith('api/'):
        return jsonify({'status': 'error', 'message': 'Not found'}), 404
    return render_template('index.html', allow_cdn_fallback=ALLOW_CDN_FALLBACK)

@app.route('/face')
def face():
    response = app.make_response(render_template('face.html', allow_cdn_fallback=ALLOW_CDN_FALLBACK))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/stats')
def api_stats():
    return jsonify(get_system_stats())


@app.route('/api/stats/snapshot')
def api_stats_snapshot():
    try:
        duration_seconds = float(request.args.get('seconds', 8.0))
        interval_seconds = float(request.args.get('interval', 1.0))
        top_limit = int(request.args.get('top', 5))
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid query parameters'}), 400

    snapshot = build_performance_snapshot(
        duration_seconds=duration_seconds,
        interval_seconds=interval_seconds,
        top_limit=top_limit,
    )
    return jsonify({'status': 'ok', 'snapshot': snapshot})

@app.route('/api/mood/<mood>')
def set_mood(mood):
    """Set the face mood via HTTP API"""
    unauthorized = require_api_key()
    if unauthorized:
        return unauthorized

    valid_moods = ['neutral', 'happy', 'sad', 'angry', 'thinking', 'surprised']
    if mood in valid_moods:
        socketio.emit('set_mood', {'mood': mood})
        return jsonify({'status': 'ok', 'mood': mood})
    return jsonify({'status': 'error', 'message': 'Invalid mood'}), 400

@app.route('/api/blink')
def trigger_blink():
    """Trigger a blink via HTTP API"""
    unauthorized = require_api_key()
    if unauthorized:
        return unauthorized

    socketio.emit('blink')
    return jsonify({'status': 'ok'})

@app.route('/api/talk/<action>')
def control_talk(action):
    """Start or stop talking animation via HTTP API"""
    unauthorized = require_api_key()
    if unauthorized:
        return unauthorized

    if action == 'start':
        socketio.emit('start_talking')
        return jsonify({'status': 'ok', 'talking': True})
    elif action == 'stop':
        socketio.emit('stop_talking')
        return jsonify({'status': 'ok', 'talking': False})
    return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

@app.route('/api/look')
def look_at():
    """Make eyes look at a position (x, y from 0-1)"""
    unauthorized = require_api_key()
    if unauthorized:
        return unauthorized

    x = request.args.get('x', type=float)
    y = request.args.get('y', type=float)
    if x is not None and y is not None:
        socketio.emit('look_at', {'x': x, 'y': y})
        return jsonify({'status': 'ok', 'x': x, 'y': y})
    return jsonify({'status': 'error', 'message': 'Missing x or y'}), 400

@app.route('/api/speak', methods=['POST'])
def api_speak():
    """Generate TTS locally with Dia2 and send to face for playback with lip sync"""
    unauthorized = require_api_key()
    if unauthorized:
        return unauthorized

    data = request.get_json() or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    speaker = data.get('speaker', 'S1')
    if not re.search(r'\[S[12]\]', text):
        text = f"[{speaker}] {text}"

    directives = parse_speech_directives(data)
    mood = directives['emotion_state']['mood']
    mood_profile = MOOD_PROFILES.get(mood, MOOD_PROFILES['neutral'])
    backend_name = (data.get('backend') or DEFAULT_TTS_BACKEND).lower()
    backend = TTS_BACKENDS.get(backend_name)
    if backend is None:
        return jsonify({'status': 'error', 'message': f'Unsupported backend: {backend_name}'}), 400

    cfg_scale = float(data.get('cfg_scale', mood_profile['cfg_scale']))
    temperature = float(data.get('temperature', mood_profile['temperature']))
    top_k = int(data.get('top_k', 50))
    speaking_rate = float(data.get('speaking_rate', mood_profile['speaking_rate']))
    use_cuda_graph = bool(data.get('use_cuda_graph', False))

    app.logger.info(
        "TTS request backend=%s mood=%s profile=%s",
        backend_name,
        mood,
        mood_profile,
    )

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
        output_path = wav_file.name

    try:
        with _dia2_lock:
            backend(
                text=text,
                output_path=output_path,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k,
                use_cuda_graph=use_cuda_graph,
                speaking_rate=speaking_rate,
            )

        with open(output_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_duration = _wav_duration_seconds(output_path)

        socketio.emit('set_mood', {
            'mood': mood,
            'emotion_state': directives['emotion_state'],
            'thought_event': directives['thought_event'],
            'expression_directives': directives,
            'emoji_directive': directives['emoji_directive'],
        })
        socketio.emit('start_talking')
        socketio.emit('play_audio', {
            'base64': audio_base64,
            'mime': 'audio/wav'
        })
        _schedule_face_reset(mood, audio_duration + 0.2)

        return jsonify({
            'status': 'ok',
            'length': len(audio_bytes),
            'base64': audio_base64,
            'mime': 'audio/wav',
            'mood': mood,
            'emotion_state': directives['emotion_state'],
            'thought_event': directives['thought_event'],
            'expression_directives': directives,
            'emoji_directive': directives['emoji_directive'],
            'backend': backend_name,
            'profile': mood_profile,
        })

    except RuntimeError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'message': f'TTS backend process failed: {e}'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'TTS error: {e}'}), 500
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

@app.route('/api/listen')
def api_listen():
    """Record and transcribe speech"""
    unauthorized = require_api_key()
    if unauthorized:
        return unauthorized

    import subprocess
    import tempfile
    import os

    duration = request.args.get('duration', default=4, type=int)
    duration = min(max(duration, 1), 10)  # Clamp between 1-10 seconds

    # Record audio
    fd, audio_file = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    try:
        # Record using arecord
        subprocess.run([
            'arecord', '-D', DEFAULT_AUDIO_DEVICE,
            '-d', str(duration),
            '-f', 'S16_LE', '-r', '16000', '-c', '1',
            audio_file
        ], check=True, capture_output=True, timeout=duration+5)

        # Transcribe using Whisper
        try:
            import whisper
        except ImportError:
            return jsonify({
                'status': 'error',
                'message': 'Whisper is not installed. Install speech dependencies with: pip install openai-whisper'
            }), 500

        model = whisper.load_model("tiny")
        result = model.transcribe(audio_file)
        text = result["text"].strip()

        # Emit to connected clients
        if text:
            socketio.emit('transcription', {'text': text})

        return jsonify({'status': 'ok', 'text': text})
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'Recording timeout'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

@app.route('/api/audio/devices')
def api_audio_devices():
    """List capture devices from arecord -l for setup/debugging"""
    try:
        result = subprocess.run(['arecord', '-l'], check=True, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'arecord not found'}), 500
    except subprocess.CalledProcessError as e:
        message = (e.stderr or e.stdout or str(e)).strip()
        return jsonify({'status': 'error', 'message': message}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'arecord timed out'}), 500

    cards = []
    current_card = None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        card_match = re.match(r'^card\s+(\d+):\s*([^,]+),\s*device\s+(\d+):\s*(.+)$', line)
        if card_match:
            current_card = {
                'card_index': int(card_match.group(1)),
                'card_name': card_match.group(2).strip(),
                'device_index': int(card_match.group(3)),
                'device_name': card_match.group(4).strip(),
                'subdevices': []
            }
            cards.append(current_card)
            continue

        if current_card and line.startswith('Subdevices:'):
            current_card['subdevices'].append(line)

    devices = [
        {
            'id': f"plughw:{entry['card_index']},{entry['device_index']}",
            'label': f"{entry['card_name']} - {entry['device_name']}",
            **entry,
        }
        for entry in cards
    ]

    return jsonify({
        'status': 'ok',
        'configured_device': DEFAULT_AUDIO_DEVICE,
        'devices': devices,
        'raw': result.stdout,
    })


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial stats
    socketio.emit('stats_update', get_system_stats())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Start background stats thread
    stats_thread = threading.Thread(target=background_stats, daemon=True)
    stats_thread.start()

    bind_host = os.environ.get('PIGUY_BIND_HOST', '127.0.0.1' if IS_PROD else '0.0.0.0')
    bind_port = int(os.environ.get('PIGUY_PORT', '5000'))

    print(f"Pi-Guy Dashboard ({PIGUY_ENV}) starting on http://{bind_host}:{bind_port}")
    socketio.run(
        app,
        host=bind_host,
        port=bind_port,
        debug=not IS_PROD,
        allow_unsafe_werkzeug=not IS_PROD,
    )
