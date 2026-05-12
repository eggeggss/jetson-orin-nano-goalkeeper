import numpy as np
import logging
from typing import Optional, Tuple
from database import get_db
from config import settings

logger = logging.getLogger(__name__)

MatchResult = Tuple[int, str, str, float]  # (person_id, name, department, confidence)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-10))


class FaceMatcher:
    """Loads authorised person embeddings from SQLite and matches against them."""

    def __init__(self):
        # {person_id: {"name": str, "department": str, "embeddings": [np.ndarray]}}
        self.persons: dict = {}

    def reload(self):
        conn = get_db()
        rows = conn.execute("""
            SELECT p.id, p.name, p.department, fe.embedding
            FROM persons p
            JOIN face_embeddings fe ON fe.person_id = p.id
            WHERE p.is_active = 1
        """).fetchall()
        conn.close()

        persons: dict = {}
        for row in rows:
            pid = row["id"]
            if pid not in persons:
                persons[pid] = {
                    "name": row["name"],
                    "department": row["department"] or "",
                    "embeddings": [],
                }
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            persons[pid]["embeddings"].append(emb)

        self.persons = persons
        logger.info("Loaded %d authorised person(s) with embeddings.", len(persons))

    def match(self, embedding: np.ndarray) -> Optional[MatchResult]:
        """
        Compare embedding against all known persons.
        Returns (person_id, name, department, confidence) or None.
        """
        best_score = -1.0
        best_pid = None

        for pid, info in self.persons.items():
            score = max(_cosine_sim(embedding, e) for e in info["embeddings"])
            if score > best_score:
                best_score = score
                best_pid = pid

        if best_pid is not None and best_score >= settings.face_confidence_threshold:
            info = self.persons[best_pid]
            return best_pid, info["name"], info["department"], best_score
        return None


matcher = FaceMatcher()
