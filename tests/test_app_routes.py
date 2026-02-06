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
