#!/usr/bin/env python3
"""
Pi-Guy Dashboard - Sci-Fi System Monitor
A futuristic dashboard for Raspberry Pi 5
"""

import base64
import uuid
import json
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pi-guy-dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")
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


def _ollama_request(path, payload):
    data = json.dumps(payload).encode("utf-8")
    url = DEFAULT_OLLAMA_HOST.rstrip("/") + path
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


def _ollama_chat(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    response = _ollama_request("/api/chat", payload)
    message = response.get("message", {})
    return message.get("content", "")




class RealtimeRAGOrchestrator:
    """Lightweight multi-tier realtime RAG + agent orchestration."""

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def start_session(self, profile='default'):
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                'profile': profile,
                'created_at': time.time(),
                'history': [],
                'memory_notes': [],
                'tool_events': [],
            }
        return session_id

    def _get_session(self, session_id):
        with self.lock:
            return self.sessions.get(session_id)

    def _classify_mood(self, text):
        value = (text or '').lower()
        if any(token in value for token in ['angry', 'mad', 'furious']):
            return 'angry'
        if any(token in value for token in ['sad', 'sorry', 'down', 'error', 'fail']):
            return 'sad'
        if any(token in value for token in ['think', 'why', 'how', 'consider', 'analyze']):
            return 'thinking'
        if any(token in value for token in ['wow', '!', 'amazing', 'surprising']):
            return 'surprised'
        if any(token in value for token in ['great', 'good', 'love', 'happy', 'awesome']):
            return 'happy'
        return 'neutral'

    def _retrieve_memory(self, session, user_text):
        recent = session['history'][-6:]
        memory_lines = []
        for item in recent:
            memory_lines.append(f"{item['role']}: {item['content'][:180]}")
        retrieval = {
            'recent_context': memory_lines,
            'memory_notes': session['memory_notes'][-6:],
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

    def run_turn(self, session_id, user_text, model=DEFAULT_TEXT_MODEL):
        session = self._get_session(session_id)
        if session is None:
            raise RuntimeError('Invalid realtime session')

        tier_memory = self._retrieve_memory(session, user_text)
        tier_tools = self._tool_router(user_text)
        tier_skill_hints = [
            'conversation',
            'planner' if len(user_text.split()) > 20 else 'quick-response',
            'diagnostics' if 'system_stats' in tier_tools else 'creative',
        ]

        system_context = {
            'memory': tier_memory,
            'tools': tier_tools,
            'skills': tier_skill_hints,
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
        reply = _ollama_chat(messages, model)
        mood = self._classify_mood(reply or user_text)

        with self.lock:
            session['history'].append({'role': 'user', 'content': user_text})
            session['history'].append({'role': 'assistant', 'content': reply})
            session['tool_events'].append({'ts': time.time(), 'tools': tier_tools})
            if user_text and len(user_text) > 24:
                session['memory_notes'].append(user_text[:220])

        return {
            'reply': reply,
            'mood': mood,
            'layers': {
                'retrieval': tier_memory,
                'tool_router': tier_tools,
                'skill_hints': tier_skill_hints,
                'synthesis_model': model,
            },
        }

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
    messages = data.get('messages', [])
    model = data.get('model', DEFAULT_TEXT_MODEL)
    if not messages:
        return jsonify({'status': 'error', 'message': 'No messages provided'}), 400
    try:
        content = _ollama_chat(messages, model)
        return jsonify({'status': 'ok', 'message': content, 'model': model})
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/vision', methods=['POST'])
def api_vision():
    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    image = data.get('image')
    model = data.get('model', DEFAULT_VISION_MODEL)
    if not prompt or not image:
        return jsonify({'status': 'error', 'message': 'Prompt and image required'}), 400
    try:
        messages = [{
            'role': 'user',
            'content': prompt,
            'images': [image],
        }]
        content = _ollama_chat(messages, model)
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
    session_id = data.get('session_id')
    user_text = (data.get('text') or '').strip()
    model = data.get('model', DEFAULT_TEXT_MODEL)
    if not session_id or not user_text:
        return jsonify({'status': 'error', 'message': 'session_id and text required'}), 400
    try:
        result = orchestrator.run_turn(session_id, user_text, model=model)
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

# Background thread for pushing updates
def background_stats():
    """Push system stats every second"""
    while True:
        stats = get_system_stats()
        socketio.emit('stats_update', stats)
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face')
def face():
    response = app.make_response(render_template('face.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/stats')
def api_stats():
    return jsonify(get_system_stats())

@app.route('/api/mood/<mood>')
def set_mood(mood):
    """Set the face mood via HTTP API"""
    valid_moods = ['neutral', 'happy', 'sad', 'angry', 'thinking', 'surprised']
    if mood in valid_moods:
        socketio.emit('set_mood', {'mood': mood})
        return jsonify({'status': 'ok', 'mood': mood})
    return jsonify({'status': 'error', 'message': 'Invalid mood'}), 400

@app.route('/api/blink')
def trigger_blink():
    """Trigger a blink via HTTP API"""
    socketio.emit('blink')
    return jsonify({'status': 'ok'})

@app.route('/api/talk/<action>')
def control_talk(action):
    """Start or stop talking animation via HTTP API"""
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
    x = request.args.get('x', type=float)
    y = request.args.get('y', type=float)
    if x is not None and y is not None:
        socketio.emit('look_at', {'x': x, 'y': y})
        return jsonify({'status': 'ok', 'x': x, 'y': y})
    return jsonify({'status': 'error', 'message': 'Missing x or y'}), 400

@app.route('/api/speak', methods=['POST'])
def api_speak():
    """Generate TTS and send to face with synchronized mood + playback events."""
    data = request.get_json() or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    speaker = data.get('speaker', 'S1')
    if not re.search(r'\[S[12]\]', text):
        text = f"[{speaker}] {text}"

    mood = (data.get('mood') or 'neutral').lower()
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

        socketio.emit('set_mood', {'mood': mood})
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
            'arecord', '-D', 'plughw:2,0',
            '-d', str(duration),
            '-f', 'S16_LE', '-r', '16000', '-c', '1',
            audio_file
        ], check=True, capture_output=True, timeout=duration+5)

        # Transcribe using Whisper
        import whisper
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

    print("Pi-Guy Dashboard starting on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
