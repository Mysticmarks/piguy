from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as piguy_app


def test_root_and_face_routes_render():
    client = piguy_app.app.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/face').status_code == 200


def test_dashboard_and_face_html_include_core_ui_hooks():
    client = piguy_app.app.test_client()

    dashboard = client.get('/')
    assert dashboard.status_code == 200
    dashboard_html = dashboard.get_data(as_text=True)
    assert 'class="dashboard"' in dashboard_html
    assert 'id="duplex-face"' in dashboard_html

    face = client.get('/face')
    assert face.status_code == 200
    face_html = face.get_data(as_text=True)
    assert 'class="face-container"' in face_html
    assert 'id="waveform-canvas"' in face_html


def test_spa_routes_fallback_to_index_html():
    client = piguy_app.app.test_client()
    resp = client.get('/dashboard/settings')
    assert resp.status_code == 200
    assert b'<!doctype html>' in resp.data.lower()


def test_unknown_api_route_is_not_handled_by_spa_fallback():
    client = piguy_app.app.test_client()
    resp = client.get('/api/does-not-exist')
    assert resp.status_code == 404
    payload = resp.get_json()
    assert payload['status'] == 'error'


def test_stats_endpoint_available_without_api_key_in_dev():
    client = piguy_app.app.test_client()
    resp = client.get('/api/stats')
    assert resp.status_code == 200
    payload = resp.get_json()
    assert 'cpu' in payload
    assert 'memory' in payload


def test_model_settings_get_and_post(tmp_path, monkeypatch):
    settings_path = tmp_path / 'model-settings.json'
    monkeypatch.setattr(piguy_app, 'MODEL_SETTINGS_PATH', str(settings_path))
    monkeypatch.setattr(piguy_app, '_model_settings', None)
    monkeypatch.setattr(piguy_app, '_model_secrets', {})
    monkeypatch.delenv('PIGUY_MODEL_API_KEY', raising=False)
    client = piguy_app.app.test_client()

    get_response = client.get('/api/settings/models')
    assert get_response.status_code == 200
    get_payload = get_response.get_json()
    assert get_payload['status'] == 'ok'
    assert get_payload['settings']['text_model']
    assert 'api_key' not in get_payload['settings']
    assert get_payload['settings']['api_key_configured'] is False

    post_response = client.post('/api/settings/models', json={
        'provider': 'ollama-local',
        'api_base': 'http://localhost:11434',
        'api_key': 'super-secret-key',
        'text_model': 'llama3.2:3b',
        'vision_model': 'llava:7b',
        'fallback': {
            'text_model': 'Xenova/all-MiniLM-L6-v2',
            'diffusion_model': 'https://cdn.jsdelivr.net/npm/@xenova/transformers',
            'audio_model': 'https://cdn.jsdelivr.net/npm/@xenova/transformers',
            'enabled': True,
            'cdn_js_libs': [
                'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2',
                'https://unpkg.com/@xenova/transformers@2.17.2',
            ],
        },
    })
    assert post_response.status_code == 200
    post_payload = post_response.get_json()
    assert post_payload['status'] == 'ok'
    assert post_payload['settings']['text_model'] == 'llama3.2:3b'
    assert 'api_key' not in post_payload['settings']
    assert post_payload['settings']['api_key_configured'] is True

    persisted = settings_path.read_text(encoding='utf-8')
    assert 'super-secret-key' not in persisted

    follow_up_get = client.get('/api/settings/models')
    assert follow_up_get.status_code == 200
    follow_up_payload = follow_up_get.get_json()
    assert 'api_key' not in follow_up_payload['settings']
    assert follow_up_payload['settings']['api_key_configured'] is True




def test_model_settings_accepts_secret_payload_in_dedicated_path(tmp_path, monkeypatch):
    monkeypatch.setattr(piguy_app, 'MODEL_SETTINGS_PATH', str(tmp_path / 'model-settings.json'))
    monkeypatch.setattr(piguy_app, '_model_settings', None)
    monkeypatch.setattr(piguy_app, '_model_secrets', {})
    monkeypatch.delenv('PIGUY_MODEL_API_KEY', raising=False)
    client = piguy_app.app.test_client()

    response = client.post('/api/settings/models', json={
        'provider': 'openai',
        'api_style': 'openai',
        'secrets': {'api_key': 'dedicated-secret'},
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['settings']['api_key_configured'] is True
    assert 'api_key' not in payload['settings']

    settings = piguy_app.get_model_settings()
    assert 'api_key' not in settings or settings['api_key'] == ''

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


def test_stats_snapshot_endpoint_returns_performance_window():
    client = piguy_app.app.test_client()
    resp = client.get('/api/stats/snapshot?seconds=1&interval=0.5&top=3')
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload['status'] == 'ok'
    snapshot = payload['snapshot']
    assert snapshot['window']['sample_count'] >= 1
    assert 'top_processes' in snapshot
    assert len(snapshot['top_processes']) <= 3
    assert 'cpu' in snapshot
    assert 'memory' in snapshot


def test_chat_completion_uses_runtime_fallback_when_model_provider_unavailable(monkeypatch):
    settings = dict(piguy_app.DEFAULT_MODEL_SETTINGS)
    settings['fallback'] = dict(settings['fallback'])

    def fail_request(path, payload, api_base=None):
        raise RuntimeError('Ollama is not reachable. Set OLLAMA_HOST if needed.')

    monkeypatch.setattr(piguy_app, '_ollama_request', fail_request)
    reply = piguy_app._chat_completion([{'role': 'user', 'content': 'Please sanity check orchestration.'}], 'llama3.1:8b', settings=settings)

    assert 'fallback reasoning mode' in reply
    assert 'sanity checks' in reply
    assert 'orchestration' in reply.lower()


def test_realtime_orchestrator_expires_inactive_sessions(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(piguy_app.time, 'time', lambda: fake_now[0])

    orchestrator = piguy_app.RealtimeRAGOrchestrator(session_ttl_seconds=120, cleanup_interval_seconds=5)
    session_id = orchestrator.start_session(profile='ttl-test')
    assert orchestrator.state(session_id) is not None

    fake_now[0] += 130
    evicted = orchestrator._cleanup_expired_sessions(force=True)
    assert evicted == 1
    assert orchestrator.state(session_id) is None

    metrics = orchestrator.get_metrics()
    assert metrics['expired_sessions_total'] >= 1
    assert metrics['active_sessions'] == 0


def test_realtime_orchestrator_bounds_session_lists_and_payload_sizes(monkeypatch):
    monkeypatch.setattr(piguy_app, '_chat_completion', lambda messages, model, settings=None: 'R' * 200)

    orchestrator = piguy_app.RealtimeRAGOrchestrator(
        max_history_items=4,
        max_memory_notes=2,
        max_tool_events=2,
        max_user_text_chars=80,
        max_reply_chars=40,
        max_modality_json_chars=120,
    )
    session_id = orchestrator.start_session(profile='bounds-test')

    oversized_modality = {
        f'k{i}': 'v' * 120
        for i in range(20)
    }
    for _ in range(4):
        orchestrator.run_turn(session_id, 'u' * 200, modality=oversized_modality)

    session = orchestrator._get_session(session_id)
    assert session is not None
    assert len(session['history']) == 4
    assert len(session['memory_notes']) == 2
    assert len(session['tool_events']) == 2

    user_messages = [entry for entry in session['history'] if entry['role'] == 'user']
    assistant_messages = [entry for entry in session['history'] if entry['role'] == 'assistant']
    assert user_messages
    assert assistant_messages
    assert all(len(entry['content']) <= 80 for entry in user_messages)
    assert all(len(entry['content']) <= 128 for entry in assistant_messages)

    metrics = orchestrator.get_metrics()
    assert metrics['truncated_user_payloads'] >= 1
    assert metrics['truncated_modality_payloads'] >= 1
    assert metrics['truncated_reply_payloads'] >= 1


def test_health_liveness_returns_process_alive_signal():
    client = piguy_app.app.test_client()
    resp = client.get('/api/health/liveness')
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload['status'] == 'ok'
    assert payload['alive'] is True
    assert isinstance(payload['pid'], int)


def test_health_readiness_returns_structured_checks(monkeypatch):
    client = piguy_app.app.test_client()

    monkeypatch.setattr(piguy_app, 'get_model_settings', lambda: {
        'api_base': 'http://provider.local',
        'api_style': 'ollama',
    })
    monkeypatch.setattr(piguy_app, '_health_provider_check', lambda settings: {
        'name': 'model_provider',
        'status': 'pass',
        'detail': 'provider reachable',
    })
    monkeypatch.setattr(piguy_app, '_health_tts_checks', lambda: [
        {'name': 'tts_dia2', 'status': 'warn', 'detail': 'not installed'},
        {'name': 'tts_xtts', 'status': 'pass', 'detail': 'available'},
        {'name': 'tts_piper', 'status': 'pass', 'detail': 'available'},
    ])
    monkeypatch.setattr(piguy_app, '_health_prod_config_check', lambda settings: {
        'name': 'prod_config',
        'status': 'pass',
        'detail': 'config present',
    })

    resp = client.get('/api/health/readiness')
    assert resp.status_code == 200
    payload = resp.get_json()

    assert payload['status'] == 'ok'
    assert payload['readiness'] == 'warn'
    assert payload['ready'] is True
    assert isinstance(payload['checks'], list)
    assert {check['status'] for check in payload['checks']} == {'pass', 'warn'}


def test_health_readiness_returns_503_when_any_check_fails(monkeypatch):
    client = piguy_app.app.test_client()

    monkeypatch.setattr(piguy_app, 'get_model_settings', lambda: {
        'api_base': 'http://provider.local',
        'api_style': 'openai',
    })
    monkeypatch.setattr(piguy_app, '_health_provider_check', lambda settings: {
        'name': 'model_provider',
        'status': 'fail',
        'detail': 'provider unreachable',
    })
    monkeypatch.setattr(piguy_app, '_health_tts_checks', lambda: [])
    monkeypatch.setattr(piguy_app, '_health_prod_config_check', lambda settings: {
        'name': 'prod_config',
        'status': 'pass',
        'detail': 'config present',
    })

    resp = client.get('/api/health/readiness')
    assert resp.status_code == 503
    payload = resp.get_json()
    assert payload['readiness'] == 'fail'
    assert payload['ready'] is False
