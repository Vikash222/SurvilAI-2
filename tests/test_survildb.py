from survildb.database import SurvilDB


def test_database_persists_camera_person_and_event(tmp_path):
    db = SurvilDB(tmp_path / "survilai.db")
    camera_id = db.add_camera("Gate", "0")
    person_id = db.add_person("Vikash")
    event_id = db.add_event("known_person", camera_id, person_id, 0.91, "track-1")

    rows = db.recent_events()
    assert event_id > 0
    assert rows[0]["camera_id"] == camera_id
    assert rows[0]["person_id"] == person_id
    assert rows[0]["event_type"] == "known_person"


def test_audit_log(tmp_path):
    db = SurvilDB(tmp_path / "survilai.db")
    audit_id = db.audit("person.created", actor="admin", target_type="person", target_id="1")
    assert audit_id > 0
