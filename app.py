#!/usr/bin/env python3
"""
Pi-Guy Dashboard - Sci-Fi System Monitor
A futuristic dashboard for Raspberry Pi 5
"""

import base64
import json
import os
import psutil
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pi-guy-dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")
_dia2_model = None
_dia2_lock = threading.Lock()

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
    """Generate TTS locally with Dia2 and send to face for playback with lip sync"""
    data = request.get_json() or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    speaker = data.get('speaker', 'S1')
    if not re.search(r'\[S[12]\]', text):
        text = f"[{speaker}] {text}"

    cfg_scale = float(data.get('cfg_scale', 2.0))
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 50))
    use_cuda_graph = bool(data.get('use_cuda_graph', False))

    try:
        from dia2 import GenerationConfig, SamplingConfig

        with _dia2_lock:
            model = get_dia2_model()
            config = GenerationConfig(
                cfg_scale=cfg_scale,
                audio=SamplingConfig(temperature=temperature, top_k=top_k),
                use_cuda_graph=use_cuda_graph,
            )

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                output_path = wav_file.name

            try:
                model.generate(text, config=config, output_wav=output_path, verbose=False)
                with open(output_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

        # Convert audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Send to all connected face clients
        socketio.emit('play_audio', {
            'base64': audio_base64,
            'mime': 'audio/wav'
        })

        return jsonify({
            'status': 'ok',
            'length': len(audio_bytes),
            'base64': audio_base64,
            'mime': 'audio/wav'
        })

    except RuntimeError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Dia2 error: {e}'}), 500

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
