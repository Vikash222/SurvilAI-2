"""
SurvilAI — Recognition Layer (Fixed)
Problems fixed: #6, #8, #9, #10, #19, #22

#6  — Har detection "aditya" ban jaati thi → strict cosine threshold
#8  — Single frame se naam decide hota tha → voting system
#9  — Do logon ko ek face samajh liya → per-track voting, not global
#10 — Threshold normal cameras ke liye tha → top-angle adjusted
#19 — Voting system nahi tha → VotingBuffer implement kiya
#22 — DB audit nahi hua → embedding diversity check
"""
from __future__ import annotations

import collections
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Protocol

from survilai.config import (
    RECOGNITION_COSINE_THRESHOLD,
    VOTING_WINDOW_FRAMES,
    VOTING_MIN_VOTES,
    DISPLAY_MIN_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecognitionResult:
    identity: str       # naam ya "unknown"
    score: float        # cosine similarity (0–1)
    confirmed: bool     # voting se confirm hua ya nahi (Problem #8, #19)


# ---------------------------------------------------------------------------
# Cosine similarity utility
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Problem #19 — VotingBuffer: frame-level voting per track
# ---------------------------------------------------------------------------

class VotingBuffer:
    """
    Problem #8 fix: Ek akeli frame mein blur, motion, partial face
    kuch bhi ho sakta hai. System seedha us ek frame se naam bol deta tha.

    Problem #19 fix: 10–15 frames ki majority se naam decide hoga.
    Ek frame ka galat result screen pe nahi aayega.

    Har track_id ka apna alag VotingBuffer hoga (Problem #9 fix).
    """

    def __init__(
        self,
        window: int = VOTING_WINDOW_FRAMES,
        min_votes: int = VOTING_MIN_VOTES,
    ) -> None:
        self.window = window
        self.min_votes = min_votes
        self._votes: collections.deque[str] = collections.deque(maxlen=window)

    def push(self, identity: str) -> None:
        self._votes.append(identity)

    def verdict(self) -> tuple[str, int, bool]:
        """
        Returns: (winning_identity, vote_count, is_confirmed)
        is_confirmed = True sirf tab jab min_votes threshold puri ho.
        """
        if not self._votes:
            return "unknown", 0, False

        counter = collections.Counter(self._votes)
        top_identity, top_count = counter.most_common(1)[0]

        # "unknown" votes ko ignore karo jab real naam aa raha ho
        if top_identity == "unknown" and len(counter) > 1:
            for name, count in counter.most_common():
                if name != "unknown":
                    top_identity, top_count = name, count
                    break

        confirmed = (
            len(self._votes) >= self.window // 2  # buffer half-full ho
            and top_count >= self.min_votes
        )
        return top_identity, top_count, confirmed

    def reset(self) -> None:
        self._votes.clear()


# ---------------------------------------------------------------------------
# Problem #22 — Embedding diversity audit
# ---------------------------------------------------------------------------

def audit_embeddings(embeddings: list[np.ndarray], max_similarity: float = 0.98) -> dict:
    """
    Database mein "aditya" ke kitne embeddings hain aur woh kitne diverse hain.

    Agar sab ek hi angle se hain → bahut similar → generic ban jaate hain
    → har unknown cheez "aditya" ban jaati hai.

    Returns audit report dict.
    """
    n = len(embeddings)
    if n == 0:
        return {"count": 0, "duplicates": 0, "recommendation": "No embeddings found"}

    duplicates = 0
    duplicate_pairs: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= max_similarity:
                duplicates += 1
                duplicate_pairs.append((i, j))

    diversity_score = 1.0 - (duplicates / max(1, n * (n - 1) / 2))

    rec = "OK"
    if diversity_score < 0.5:
        rec = ("CRITICAL: Embeddings bahut similar hain. "
               "Delete karke fresh registration karo — top-angle, "
               "raat ke time, actual camera ke saamne.")
    elif diversity_score < 0.75:
        rec = "WARNING: Aur diverse angles se register karo (low_light, top_angle)."

    return {
        "count": n,
        "duplicates": duplicates,
        "duplicate_pairs": duplicate_pairs,
        "diversity_score": round(diversity_score, 3),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Problem #6 — EmbeddingMatcher: strict threshold + "aditya" dominance fix
# ---------------------------------------------------------------------------

class EmbeddingMatcher:
    """
    Database se embedding match karo — strict cosine threshold ke saath.

    Problem #6 fix:
    - Threshold 0.42 (distance) → sirf genuine matches pass honge
    - Agar koi bhi registered person match nahi karta → "unknown" return karo
    - "aditya" ya koi bhi person ki generic/dominant embedding wali problem
      threshold strictness se handle hoti hai

    Problem #10 fix:
    - Top-angle cameras pe same person ki embedding naturally thodi alag hoti hai
    - Isliye threshold slightly relaxed (0.42 vs standard 0.45 distance)
      lekin confidence score pe bhi filter lagaya hai
    """

    def __init__(self, database: dict[str, list[np.ndarray]]) -> None:
        """
        database: { "aditya": [emb1, emb2, ...], "rahul": [...], ... }
        """
        self.database = database

    def identify(
        self,
        query_embedding: np.ndarray,
        min_confidence: float = DISPLAY_MIN_CONFIDENCE,
    ) -> RecognitionResult:
        """
        Query embedding ko database se match karo.
        Sirf tab naam do jab similarity genuinely high ho.
        """
        if not self.database:
            return RecognitionResult("unknown", 0.0, False)

        best_name = "unknown"
        best_sim = 0.0

        for person_name, embeddings in self.database.items():
            for stored_emb in embeddings:
                sim = cosine_similarity(query_embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = person_name

        # Problem #6 — Strict threshold: low similarity → unknown
        # cosine_similarity 1.0 = perfect match, 0.0 = no match
        # RECOGNITION_COSINE_THRESHOLD 0.42 distance = 0.58 similarity
        similarity_threshold = 1.0 - RECOGNITION_COSINE_THRESHOLD

        if best_sim < similarity_threshold:
            return RecognitionResult("unknown", best_sim, False)

        # Problem #21 — Display filter: confirmed tab jab score bhi high ho
        confirmed = best_sim >= min_confidence
        return RecognitionResult(best_name, best_sim, confirmed)

    def identify_with_voting(
        self,
        query_embedding: np.ndarray,
        voting_buffer: VotingBuffer,
    ) -> RecognitionResult:
        """
        Single-frame result ko voting buffer mein push karo.
        Screen pe naam sirf tab aaye jab voting confirm kare.

        Problem #8, #19 fix.
        """
        # Single frame result (may be wrong)
        frame_result = self.identify(query_embedding)

        # Push to voting buffer
        voting_buffer.push(frame_result.identity)

        # Get majority verdict
        verdict_name, vote_count, is_confirmed = voting_buffer.verdict()

        # Return confirmed result only
        return RecognitionResult(
            identity=verdict_name if is_confirmed else "unknown",
            score=frame_result.score,
            confirmed=is_confirmed,
        )


# ---------------------------------------------------------------------------
# Safe placeholder (Phase 1 — jab recognition model ready nahi ho)
# ---------------------------------------------------------------------------

class RecognitionNotConfigured:
    """Phase 1 placeholder — koi bhi match nahi karta, sab unknown."""

    def identify(self, face_image) -> RecognitionResult:
        return RecognitionResult(identity="unknown", score=0.0, confirmed=False)