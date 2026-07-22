"""
Backend service layer for IndustrialMind AI.

All functions in this file are kept IDENTICAL in behavior to the original
Streamlit app (api_get, api_post, upload_document, embedded_get,
embedded_post, embedded_service). Only Streamlit-specific calls
(st.cache_data / st.cache_resource / st.error) have been swapped for
plain-Python equivalents (functools.lru_cache-based TTL cache, print,
and a module-level singleton) since Flask has no Streamlit runtime.
"""

import os
import sys
import time
import threading
from pathlib import Path

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
EMBEDDED_MODE = API_BASE_URL.lower() == "embedded"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
session = requests.Session()

session.headers.update({
        "Accept": "application/json"
    })

# ------------------------------------------------------------------
# Embedded service singleton
# (replaces st.cache_resource(show_spinner="Starting IndustrialMind engine..."))
# ------------------------------------------------------------------
_embedded_service_instance = None
_embedded_service_lock = threading.Lock()


def embedded_service():
    global _embedded_service_instance
    if _embedded_service_instance is not None:
        return _embedded_service_instance

    with _embedded_service_lock:
        if _embedded_service_instance is not None:
            return _embedded_service_instance

        from backend.config import UPLOAD_FOLDER
        # from backend.services import service

        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        raise RuntimeError("Embedded mode disabled")
        # _embedded_service_instance = service
        # return _embedded_service_instance


# ------------------------------------------------------------------
# Simple TTL cache (replaces st.cache_data(ttl=5))
# ------------------------------------------------------------------
class _TTLCache:
    def __init__(self, ttl_seconds=5):
        self.ttl = ttl_seconds
        self._store = {}
        self._lock = threading.Lock()

    def get_or_set(self, key, fn):
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                value, ts = entry
                if now - ts < self.ttl:
                    return value

        value = fn()
        with self._lock:
            self._store[key] = (value, now)
        return value

    def clear(self):
        with self._lock:
            self._store.clear()


_metrics_cache = _TTLCache(ttl_seconds=5)
_documents_cache = _TTLCache(ttl_seconds=5)


def get_metrics():
    return _metrics_cache.get_or_set("metrics", lambda: api_get("/metrics"))


def get_documents():
    return _documents_cache.get_or_set("documents", lambda: api_get("/documents"))


def clear_data_cache():
    """Equivalent to st.cache_data.clear()"""
    _metrics_cache.clear()
    _documents_cache.clear()


# ------------------------------------------------------------------
# api_get / api_post / upload_document — UNCHANGED LOGIC
# ------------------------------------------------------------------
def api_get(path: str):
    if EMBEDDED_MODE:
        return embedded_get(path)

    last_error = None

    for _ in range(5):
        try:
            response = session.get(
                f"{API_BASE_URL}{path}",
                timeout=(5, 120)
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2)

    raise last_error


def api_post(path: str, payload: dict):
    if EMBEDDED_MODE:
        return embedded_post(path, payload)
    
    session = requests.Session()

    session.headers.update({
        "Accept": "application/json"
    })
    try:
        response = session.post(f"{API_BASE_URL}{path}", json=payload, timeout=(10, 300))
        # response.status_code != 200:
        # print(response.text)
        response.raise_for_status()
            
        return response.json()
    except requests.Timeout:
        raise RuntimeError(
            "Backend request timed out."
        )

    except requests.ConnectionError:
        raise RuntimeError(
            "Cannot connect to backend."
        )

    except requests.HTTPError:

        try:
            detail = response.json().get("detail")
        except:
            detail = response.text

        raise RuntimeError(detail)


def upload_document(file_storage):
    """
    file_storage: a werkzeug FileStorage object (from Flask's request.files),
    exposing .filename and .read() / .stream, mirroring the Streamlit
    UploadedFile's .name and .getvalue().
    """
    if EMBEDDED_MODE:
        service = embedded_service()
        from backend.config import UPLOAD_FOLDER

        file_bytes = file_storage.read()
        file_name = file_storage.filename

        safe_name = "".join(char if char.isalnum() or char in ".-_" else "_" for char in file_name)
        destination = Path(UPLOAD_FOLDER) / safe_name
        destination.write_bytes(file_bytes)
        document = service.ingest_path(destination)
        return {
            "document": {
                "id": document["id"],
                "file_name": document["file_name"],
                "document_type": document["document_type"],
                "chunks": document["chunks"],
                "entities": document["entities"],
                "metadata": document["metadata"],
            },
            "message": "Document stored, parsed, chunked, embedded, and linked in the knowledge graph.",
        }

    response = session.post(
        f"{API_BASE_URL}/documents/upload",
        files={
            "file": (
                file_storage.filename, 
                file_storage.stream,
                file_storage.mimetype,
                )
            },
        timeout=(10, 1800),
    )
    response.raise_for_status()
    return response.json()


def embedded_get(path: str):
    service = embedded_service()
    if path == "/":
        return {
            "message": "IndustrialMind AI running in embedded Streamlit mode",
            "storage": {
                "raw_documents": "Streamlit app filesystem",
                "metadata_and_chunks": "SQLite operational database",
                "semantic_search": "Local embeddings with Qdrant support when configured",
                "knowledge_graph": "NetworkX/JSON export with Neo4j support when configured",
            },
            "qdrant_connected": service.vector_store.available,
        }
    if path == "/metrics":
        return service.metrics()
    if path == "/documents":
        return [
            {
                "id": document["id"],
                "file_name": document["file_name"],
                "document_type": document["document_type"],
                "chunks": document["chunks"],
                "entities": document["entities"],
                "metadata": document["metadata"],
            }
            for document in service.documents()
        ]
    if path == "/knowledge-graph":
        return service.graph_payload()
    if path == "/lessons":
        return service.lessons()
    raise ValueError(f"Unsupported embedded GET path: {path}")


def embedded_post(path: str, payload: dict):
    service = embedded_service()
    if path == "/ask":
        return service.ask(payload["question"], payload.get("top_k", 5), payload.get("selected_documents"))
    if path == "/clear":
        return service.clear_knowledge_base()
    if path == "/maintenance":
        return service.maintenance(payload["equipment_tag"])
    if path == "/compliance":
        return service.compliance(payload.get("standard", "Factory Act, OISD, PESO, environmental norms, ISO 9001"))
    raise ValueError(f"Unsupported embedded POST path: {path}")


def get_health():
    """
    Equivalent to the startup health-check loop in the Streamlit app:
    for _ in range(15): try api_get("/") ... time.sleep(2)
    Returns (health_dict_or_None, last_error_or_None).
    """
    global EMBEDDED_MODE
    health = None
    last_error = None

    for _ in range(15):
        try:
            health = api_get("/")
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2)

    if health is None and API_BASE_URL == "http://localhost:8000":
        EMBEDDED_MODE = True
        health = api_get("/")
        last_error = None

    return health, last_error