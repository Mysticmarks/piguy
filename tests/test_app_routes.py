from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as piguy_app


def test_root_and_face_routes_render():
    client = piguy_app.app.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/face').status_code == 200


def test_stats_endpoint_available_without_api_key_in_dev():
    client = piguy_app.app.test_client()
    resp = client.get('/api/stats')
    assert resp.status_code == 200
    payload = resp.get_json()
    assert 'cpu' in payload
    assert 'memory' in payload


def test_model_settings_get_and_post(tmp_path, monkeypatch):
    monkeypatch.setattr(piguy_app, 'MODEL_SETTINGS_PATH', str(tmp_path / 'model-settings.json'))
    monkeypatch.setattr(piguy_app, '_model_settings', None)
    client = piguy_app.app.test_client()

    get_response = client.get('/api/settings/models')
    assert get_response.status_code == 200
    get_payload = get_response.get_json()
    assert get_payload['status'] == 'ok'
    assert get_payload['settings']['text_model']

    post_response = client.post('/api/settings/models', json={
        'provider': 'ollama-local',
        'api_base': 'http://localhost:11434',
        'text_model': 'llama3.2:3b',
        'vision_model': 'llava:7b',
        'fallback': {
            'text_model': 'Xenova/all-MiniLM-L6-v2',
            'diffusion_model': 'https://cdn.jsdelivr.net/npm/@xenova/transformers',
            'audio_model': 'https://cdn.jsdelivr.net/npm/@xenova/transformers',
        },
    })
    assert post_response.status_code == 200
    post_payload = post_response.get_json()
    assert post_payload['status'] == 'ok'
    assert post_payload['settings']['text_model'] == 'llama3.2:3b'


def test_realtime_turn_returns_structured_contract(monkeypatch):
    monkeypatch.setattr(piguy_app, '_chat_completion', lambda messages, model, settings=None: 'Thinking through it now!')
    client = piguy_app.app.test_client()

    start = client.post('/api/realtime/session/start', json={'profile': 'test'})
    session_id = start.get_json()['session_id']

    response = client.post('/api/realtime/turn', json={'session_id': session_id, 'text': 'How are you?'})
    assert response.status_code == 200
    payload = response.get_json()

    assert payload['status'] == 'ok'
    assert 'emotion_state' in payload
    assert 'thought_event' in payload
    assert 'expression_directives' in payload
    assert 'emoji_directive' in payload
    assert payload['mood'] == payload['emotion_state']['mood']
    assert 'layers' in payload
    assert 'digital_being' in payload['layers']
    assert 'sequential_task_list' in payload['layers']

    digital_being = payload['layers']['digital_being']
    assert 'perception' in digital_being
    assert 'appraisal' in digital_being
    assert 'cognition' in digital_being
    assert 'identity' in digital_being
    assert 'action_policy' in digital_being
    assert 'expression' in digital_being

    task_list = payload['layers']['sequential_task_list']
    assert len(task_list) == 6
    statuses = {step['status'] for step in task_list}
    assert 'in_progress' in statuses or 'completed' in statuses


def test_speak_accepts_structured_directives_and_legacy_mood(monkeypatch):
    client = piguy_app.app.test_client()

    monkeypatch.setattr(piguy_app, '_wav_duration_seconds', lambda path: 0.5)
    monkeypatch.setattr(piguy_app, '_schedule_face_reset', lambda mood, delay: None)

    def fake_backend(**kwargs):
        with open(kwargs['output_path'], 'wb') as wav_file:
            wav_file.write(b'RIFF0000WAVEfmt ')

    monkeypatch.setitem(piguy_app.TTS_BACKENDS, 'dia2', fake_backend)

    structured_resp = client.post('/api/speak', json={
        'text': 'Structured speech',
        'expression_directives': {
            'emotion_state': {'mood': 'happy', 'intensity': 0.9},
            'emoji_directive': {'emoji': '😄', 'placement': 'status', 'intensity': 0.8},
        }
    })
    assert structured_resp.status_code == 200
    structured_payload = structured_resp.get_json()
    assert structured_payload['mood'] == 'happy'
    assert structured_payload['expression_directives']['emotion_state']['mood'] == 'happy'

    legacy_resp = client.post('/api/speak', json={'text': 'Legacy speech', 'mood': 'sad'})
    assert legacy_resp.status_code == 200
    legacy_payload = legacy_resp.get_json()
    assert legacy_payload['mood'] == 'sad'
    assert legacy_payload['expression_directives']['compat']['mood'] == 'sad'
def test_realtime_turn_includes_thought_events(monkeypatch):
    monkeypatch.setattr(piguy_app, '_chat_completion', lambda messages, model, settings=None: 'I can help with system stats.')
    client = piguy_app.app.test_client()

    start_resp = client.post('/api/realtime/session/start', json={'profile': 'face-omnimodal'})
    assert start_resp.status_code == 200
    session_id = start_resp.get_json()['session_id']

    turn_resp = client.post('/api/realtime/turn', json={
        'session_id': session_id,
        'text': 'Can you check cpu and memory quickly?',
    })
    assert turn_resp.status_code == 200
    payload = turn_resp.get_json()
    assert payload['status'] == 'ok'
    assert 'thought_events' in payload
    assert isinstance(payload['thought_events'], list)
    assert payload['thought_events']
    first = payload['thought_events'][0]
    assert 'text' in first
    assert 'emotion_tags' in first
    assert 'emoji' in first
    assert 'lifetime_ms' in first
    assert 'intensity' in first


def test_realtime_behavior_state_changes_smoothly(monkeypatch):
    monkeypatch.setattr(piguy_app, '_chat_completion', lambda messages, model, settings=None: 'Great question, I can help!')
    client = piguy_app.app.test_client()

    start_resp = client.post('/api/realtime/session/start', json={'profile': 'face-omnimodal'})
    session_id = start_resp.get_json()['session_id']

    first_turn = client.post('/api/realtime/turn', json={'session_id': session_id, 'text': 'How do we plan this system?'})
    assert first_turn.status_code == 200
    first_payload = first_turn.get_json()
    first_identity = first_payload['layers']['digital_being']['identity']['current']

    second_turn = client.post('/api/realtime/turn', json={'session_id': session_id, 'text': 'Awesome! keep going with high energy!!'})
    assert second_turn.status_code == 200
    second_payload = second_turn.get_json()
    second_identity = second_payload['layers']['digital_being']['identity']['current']

    for key in ['warmth', 'directness', 'energy', 'reflectiveness']:
        assert 0.0 <= first_identity[key] <= 1.0
        assert 0.0 <= second_identity[key] <= 1.0
        assert abs(second_identity[key] - first_identity[key]) < 0.5
