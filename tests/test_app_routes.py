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
