from dashboard.app import create_app


def test_dashboard_endpoints(tmp_path):
    app = create_app(tmp_path / "survilai.db")
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["database"] == "local"
    assert client.get("/api/cameras").get_json() == []
    assert client.get("/api/people").get_json() == []
    assert client.get("/api/events").get_json() == []
