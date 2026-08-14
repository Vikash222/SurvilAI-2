import numpy as np

from survildb.database import SurvilDB


def test_embedding_persistence(tmp_path):
    db = SurvilDB(tmp_path / "survilai.db")
    person_id = db.add_person("Test Person")
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embedding_id = db.add_embedding(person_id, vector.tobytes(), 3)
    assert embedding_id > 0
    with db.connect() as conn:
        row = conn.execute("SELECT person_id, dimension, embedding FROM face_embeddings WHERE id=?", (embedding_id,)).fetchone()
    assert row["person_id"] == person_id
    assert row["dimension"] == 3
    assert np.frombuffer(row["embedding"], dtype=np.float32).tolist() == [1.0, 0.0, 0.0]
