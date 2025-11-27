#!/usr/bin/env python3
"""
Pi-Guy Dashboard - Sci-Fi System Monitor
A futuristic dashboard for Raspberry Pi 5
"""

import psutil
import json
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pi-guy-dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

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
    """Generate TTS with ElevenLabs and send to face for playback with lip sync"""
    import os
    import base64
    import requests

    data = request.get_json() or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    api_key = os.environ.get('ELEVENLABS_API_KEY')
    if not api_key:
        return jsonify({'status': 'error', 'message': 'ELEVENLABS_API_KEY not set'}), 500

    # Default voice - can be overridden
    voice_id = data.get('voice_id', 'pNInz6obpgDQGcFmaJgB')  # Adam voice
    model_id = data.get('model_id', 'eleven_monolingual_v1')

    try:
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers={
                'xi-api-key': api_key,
                'Content-Type': 'application/json'
            },
            json={
                'text': text,
                'model_id': model_id,
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.75
                }
            }
        )

        if response.status_code != 200:
            return jsonify({'status': 'error', 'message': f'ElevenLabs error: {response.text}'}), 500

        # Convert audio to base64
        audio_base64 = base64.b64encode(response.content).decode('utf-8')

        # Send to all connected face clients
        socketio.emit('play_audio', {
            'base64': audio_base64,
            'mime': 'audio/mpeg'
        })

        return jsonify({'status': 'ok', 'length': len(response.content)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
