import hashlib
import math
import os
import threading
import torch
import numpy as np
import re

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model = None
        self.dimension = 768
        self.model_name = model_name
        self._load_lock = threading.Lock()
        self._load_attempted = False
        self.use_sentence_transformers = os.getenv("USE_SENTENCE_TRANSFORMERS", "true").lower() == "true"

    def _ensure_loaded(self) -> None:
        """
        Lazily load the SentenceTransformer model on first use instead of in
        __init__. Loading torch/SentenceTransformer eagerly at service
        construction time means it can run more than once in close succession
        if something (e.g. Streamlit's cache_resource on a rerun, or two
        worker processes) constructs the service twice before the first call
        finishes — two near-simultaneous torch/OpenMP initializations in the
        same process is a common cause of native segfaults (uncatchable,
        unlike Python exceptions). The lock ensures only one thread ever
        performs the actual load.
        """
        if self.model is not None or self._load_attempted:
            return
        if not (SentenceTransformer and self.use_sentence_transformers):
            self._load_attempted = True
            return

        with self._load_lock:
            if self.model is not None or self._load_attempted:
                return
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(self.model_name, device=device)
            except Exception as e:
                print(f"[EmbeddingModel] Failed to load {self.model_name}: {e}")
                self.model = None
            finally:
                self._load_attempted = True

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        self._ensure_loaded()

        if self.model:
            texts = [
                t[:8000]
                for t in texts
        ]
            vectors = self.model.encode(
                texts, 
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True)
            return vectors.tolist()

        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(
            r"\w+",
            text.lower()
        )

        for token in tokens:
            if len(token) < 2:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1 if digest[4] % 2 == 0 else -1
            vector[index] += sign

        norm = math.sqrt(float(np.dot(vector, vector))) or 1.0
        return (vector / norm).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    left = np.array(a, dtype=np.float32)
    right = np.array(b, dtype=np.float32)
    denominator = (np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
    return float(np.dot(left, right) / denominator)