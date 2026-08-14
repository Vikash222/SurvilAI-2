# Phase 6 — Local Persistence

SurvilAI now has a local SQLite persistence layer for the edge-first deployment.

## Stored data

- cameras: source and enablement state
- people: local identities
- face_embeddings: binary embedding records linked to people
- events: recognition/security events with camera/person/track references
- audit_logs: administrative and data-change audit trail

## Retention direction

The database layer intentionally does not retain raw CCTV video by default. Future retention jobs should support event-based TTLs and explicit deletion. Biometric data and snapshots must be protected with access controls and appropriate retention policies before commercial deployment.

## Migration direction

The persistence API is kept small so SQLite can be replaced by PostgreSQL for larger deployments without coupling the recognition engine to SQL details.

## CLI

```bash
python -m survildb.cli --db data/survilai.db camera Gate 0
python -m survildb.cli --db data/survilai.db person Vikash
python -m survildb.cli --db data/survilai.db event known_person --camera-id 1 --person-id 1 --confidence 0.91 --track-id track-1
python -m survildb.cli --db data/survilai.db events
```
