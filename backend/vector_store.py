from typing import Any
import re
from backend.config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except Exception:  # pragma: no cover
    QdrantClient = None
    models = None


class VectorStore:
    def __init__(self, dimension: int):
        self.available = False
        self.dimension = dimension
        self.client = None
        if not QdrantClient:
            return
        try:
            kwargs = {
                "url": QDRANT_URL,
                "check_compatibility": False,
            }
            if QDRANT_API_KEY:
                kwargs["api_key"] = QDRANT_API_KEY
            self.client = QdrantClient(
                # **kwargs
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
                timeout=60,
                check_compatibility=False,
            )
            print("=" * 60)
            print("QDRANT URL:", QDRANT_URL)
            print("=" * 60)
            self.client.get_collections()
            print("CONNECTED")
            self._ensure_collection()
            print(
                self.client.get_collection(
                    QDRANT_COLLECTION
                )
            )
            
            # Create payload indexes for faster filtering
            for field in ["file_name", "document_type", "document_id"]:
                try:
                    self.client.create_payload_index(
                        collection_name=QDRANT_COLLECTION,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
            self.available = True
        except Exception as e:
            print("Qdrant connection error:", e)
            self.client = None

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not self.available or not self.client:
            return
        points = [
            models.PointStruct(
                id=chunk["qdrant_id"],
                vector=chunk["embedding"],
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "file_name": chunk["file_name"],
                    "document_type": chunk["document_type"],
                    "position": chunk["position"],
                    "text": chunk["text"],
                    "length": chunk["chunk_length"],
                    "word_count": chunk["word_count"],
                    "entities": chunk.get("entities", {}),
                    "knowledge_entities": chunk.get("knowledge_entities", []),
                    "relationships": chunk.get("relationships", []),
                    "metadata": chunk.get("metadata", {}),
                },
            )
            for chunk in chunks
        ]
        self.client.upsert(collection_name=QDRANT_COLLECTION,wait=True, points=points)

    def search(self, query_vector: list[float], top_k: int,selected_documents=None,) -> list[dict[str, Any]]:
        if not self.available or not self.client:
            return []
        try:
            search_filter = None

            if selected_documents:

                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name",
                            match=models.MatchAny(any=selected_documents),
                        )
                    ]
                )
            results = self.client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                query_filter=search_filter,
                limit = top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            try:
                response = self.client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=query_vector,
                    query_filter=search_filter,
                    limit = 2,
                )
                results = response.points
            except Exception as e:
                print("Qdrant search error:", e)
                return []
        hits = []

        for result in results:

            score = float(result.score)
            hits.append({
                **result.payload,
                "score": score,
                "vector_score": score,
                "qdrant_id": result.id,
            })

        seen = set()
        unique = []

        selected = []

        used_documents = {}

        for hit in hits:

            doc = hit["document_id"]

            if used_documents.get(doc, 0) >= 3:
                continue

            selected.append(hit)

            used_documents[doc] = (
                used_documents.get(doc, 0) + 1
            )

        return selected
    
    def batch_search(
        self,
        vectors,
        top_k=5,
    ):

        results = []

        for vector in vectors:

            results.append(
                self.search(
                    vector,
                    top_k
                )
            )

        return results
    
    def search_by_document(
        self,
        query_vector,
        file_name,
        top_k=10,
    ):
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            selected_documents=[file_name],
        )

    def delete_document(self, document_id: str):
        """
        Delete all vectors belonging to a document.
        """

        if not self.available or not self.client:
            return

        try:

            self.client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(
                                    value=document_id
                                ),
                            )
                        ]
                    )
                ),
            )

            print(f"Deleted vectors for document: {document_id}")

        except Exception as e:
            print(f"Qdrant delete error: {e}")
    def clear(self):
        if not self.available or not self.client:
            return

        try:
            self.client.delete_collection(QDRANT_COLLECTION)
            
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

            for field in ["file_name", "document_type", "document_id"]:
                try:
                    self.client.create_payload_index(
                        collection_name=QDRANT_COLLECTION,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

        except Exception as e:
            # pass
            print("Qdrant clear error:", e)

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == QDRANT_COLLECTION for collection in collections):
            return
        self.client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
        ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=1000
            ),
        )

        for field in ["file_name", "document_type", "document_id"]:
            try:
                self.client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def classify_query(self, question: str) -> dict[str, Any]:
        """
        Classify the user's intent so the best retrieval strategy
        can be selected.
        """

        q = question.lower().strip()

        list_patterns = [
            r"\blist\b",
            r"\bshow all\b",
            r"\bdisplay\b",
            r"\bwhat are\b",
            r"\bgive me all\b",
            r"\bnames of\b",
        ]

        count_patterns = [
            r"\bhow many\b",
            r"\bcount\b",
            r"\bnumber of\b",
            r"\btotal\b",
        ]

        comparison_patterns = [
            r"\bcompare\b",
            r"\bdifference\b",
            r"\bvs\b",
            r"\bversus\b",
        ]

        procedure_patterns = [
            r"\bhow to\b",
            r"\bsteps\b",
            r"\bprocedure\b",
            r"\bprocess\b",
        ]

        summary_patterns = [
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\boverview\b",
            r"\bbrief\b",
        ]

        maintenance_patterns = [
            r"\bmaintenance\b",
            r"\brepair\b",
            r"\bservice history\b",
            r"\bfailure\b",
        ]

        compliance_patterns = [
            r"\biso\b",
            r"\boisd\b",
            r"\bpeso\b",
            r"\bcompliance\b",
            r"\bregulation\b",
        ]

        if any(re.search(p, q) for p in list_patterns):
            return {"intent": "LIST"}

        if any(re.search(p, q) for p in count_patterns):
            return {"intent": "COUNT"}

        if any(re.search(p, q) for p in comparison_patterns):
            return {"intent": "COMPARE"}

        if any(re.search(p, q) for p in procedure_patterns):
            return {"intent": "PROCEDURE"}

        if any(re.search(p, q) for p in summary_patterns):
            return {"intent": "SUMMARY"}

        if any(re.search(p, q) for p in maintenance_patterns):
            return {"intent": "MAINTENANCE"}

        if any(re.search(p, q) for p in compliance_patterns):
            return {"intent": "COMPLIANCE"}

        return {"intent": "GENERAL"}
