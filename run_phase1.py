import argparse

from survilai.config import EngineConfig
from survilai.core.detection import HaarFaceDetector
from survilai.core.frame_source import VideoSource
from survilai.core.pipeline import LocalPipeline
from survilai.core.recognition import RecognitionNotConfigured


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SurvilAI Phase 1 local vision engine")
    parser.add_argument("--source", default="0", help="Webcam index or local video file")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    config = EngineConfig(source=args.source, display=not args.no_display)
    pipeline = LocalPipeline(
        source=VideoSource(config.source),
        detector=HaarFaceDetector(),
        recognizer=RecognitionNotConfigured(),
    )
    pipeline.run(display=config.display)


if __name__ == "__main__":
    main()
