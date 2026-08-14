from dashboard.app import create_app


def test_camera_and_person_management(tmp_path):
    app = create_app(tmp_path / "survilai.db")
    client = app.test_client()

    camera = client.post('/api/cameras', json={'name': 'Gate 01', 'source': '0'})
    assert camera.status_code == 201
    assert client.get('/api/cameras').get_json()[0]['name'] == 'Gate 01'

    person = client.post('/api/people', json={'name': 'Vikash'})
    assert person.status_code == 201
    assert client.get('/api/people').get_json()[0]['name'] == 'Vikash'

    camera_id = camera.get_json()['id']
    assert client.delete(f'/api/cameras/{camera_id}').status_code == 200
    assert client.delete(f'/api/cameras/{camera_id}').status_code == 404
