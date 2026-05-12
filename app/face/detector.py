import logging
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps InsightFace FaceAnalysis for detection + embedding."""

    def __init__(self):
        self.app = None

    def load(self):
        """Download (if needed) and load the buffalo_s model."""
        from insightface.app import FaceAnalysis  # lazy import after pip install

        logger.info("Loading InsightFace buffalo_s model from %s …", settings.models_dir)
        self.app = FaceAnalysis(
            name="buffalo_s",
            root=settings.models_dir,
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("Face model ready.")

    def detect(self, frame: np.ndarray) -> list:
        """Return list of InsightFace Face objects."""
        if self.app is None:
            return []
        return self.app.get(frame)

    def best_face(self, frame: np.ndarray):
        """Return the highest-confidence face in the frame, or None."""
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f.det_score)


detector = FaceDetector()
