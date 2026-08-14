from __future__ import annotations

import argparse

from .database import SurvilDB


def main() -> None:
    parser = argparse.ArgumentParser(description="SurvilAI local database utility")
    parser.add_argument("--db", default="data/survilai.db")
    sub = parser.add_subparsers(dest="command", required=True)

    camera = sub.add_parser("camera")
    camera.add_argument("name")
    camera.add_argument("source")

    person = sub.add_parser("person")
    person.add_argument("name")

    event = sub.add_parser("event")
    event.add_argument("event_type")
    event.add_argument("--camera-id", type=int)
    event.add_argument("--person-id", type=int)
    event.add_argument("--confidence", type=float)
    event.add_argument("--track-id")

    sub.add_parser("events")

    args = parser.parse_args()
    db = SurvilDB(args.db)

    if args.command == "camera":
        print(f"camera_id={db.add_camera(args.name, args.source)}")
    elif args.command == "person":
        print(f"person_id={db.add_person(args.name)}")
    elif args.command == "event":
        print(f"event_id={db.add_event(args.event_type, args.camera_id, args.person_id, args.confidence, args.track_id)}")
    elif args.command == "events":
        for row in db.recent_events():
            print(dict(row))


if __name__ == "__main__":
    main()
