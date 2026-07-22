import hashlib
import asyncio
import shutil
import threading
import time
from pathlib import Path
import re
from typing import Any

import torch
from fastapi import UploadFile
from PIL import Image, ImageEnhance
import pytesseract
from transformers import AutoModelForCausalLM, AutoProcessor

from backend.agents import (
    ComplianceAgent,
    LessonsAgent,
    MaintenanceAgent,
)
from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    KNOWLEDGE_EXTRACTION_ENABLED,
    MAX_KNOWLEDGE_CHUNKS,
    UPLOAD_FOLDER,
)
from backend.embeddings import (
    EmbeddingModel,
    cosine_similarity,
)
from backend.entity_extractor import EntityExtractor
from backend.knowledge_graph import KnowledgeGraph
from backend.storage import JsonStore
from backend.text_processing import (
    chunk_text,
    excerpt,
)
from backend.vector_store import VectorStore
from ingestion.docling_loader import DoclingLoader
from ingestion.file_detector import FileDetector
from ingestion.metadata import MetadataExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IndustrialMindService:
    """
    IndustrialMind AI Service

    Responsibilities
    ----------------
    • Universal document ingestion
    • OCR & Florence-2 image understanding
    • Embedding generation
    • Hybrid Retrieval (Vector + Keyword)
    • RAG Answer Generation
    • Industrial Entity Extraction
    • Knowledge Graph construction
    • Maintenance Intelligence
    • Compliance Intelligence
    • Lessons Learned Intelligence
    """

    def __init__(self):

        print("=" * 80)
        print("Initializing IndustrialMind AI ...")
        print("=" * 80)

        # --------------------------------------------------
        # Document Processing
        # --------------------------------------------------

        self.loader = DoclingLoader()

        self.store = JsonStore()

        # --------------------------------------------------
        # Embedding Engine
        # --------------------------------------------------

        self.embedder = EmbeddingModel()

        # --------------------------------------------------
        # Vector Database
        # --------------------------------------------------

        self.vector_store = VectorStore(
            self.embedder.dimension
        )

        # --------------------------------------------------
        # Knowledge Extraction
        # --------------------------------------------------

        self.entities = EntityExtractor()

        self.graph = KnowledgeGraph()

        # --------------------------------------------------
        # Industrial AI Agents
        # --------------------------------------------------

        self.maintenance_agent = MaintenanceAgent()

        self.compliance_agent = ComplianceAgent()

        self.lessons_agent = LessonsAgent()

        # --------------------------------------------------
        # Florence Vision Model
        # (lazy loading)
        # --------------------------------------------------

        self.processor = None

        self.vision_model = None

        self._florence_lock = threading.Lock()

        # --------------------------------------------------
        # Retrieval Configuration
        # --------------------------------------------------

        self.DEFAULT_TOP_K = 8

        self.MAX_CONTEXT_CHUNKS = 8

        self.MIN_SIMILARITY = 0.08

        self.KEYWORD_BOOST = 0.15

        self.NEIGHBOR_WINDOW = 1

        print("Embedding Dimension :", self.embedder.dimension)
        print("Vector Store        :", self.vector_store.available)
        print("Florence Loaded     : False")
        print("=" * 80)

    # ==========================================================
    # Florence Loader
    # ==========================================================

    def load_florence(self):
        """
        Lazy-load Florence only when the first image
        is uploaded.

        This saves nearly 2 GB of RAM when the user
        only uploads PDFs or DOCX files.
        """

        if self.processor is not None:
            return

        with self._florence_lock:

            if self.processor is not None:
                return

            print("=" * 80)
            print("Loading Florence-2...")
            print("=" * 80)

            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base",
                revision="refs/pr/6",
                trust_remote_code=True,
            )

            self.vision_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                revision="refs/pr/6",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(DEVICE)

            self.vision_model.eval()

            print("Florence Loaded Successfully")
            print("=" * 80)


            # ==========================================================
    # Upload APIs
    # ==========================================================

    async def ingest_upload(self, upload: UploadFile) -> dict[str, Any]:
        """
        Save an uploaded file and start the ingestion pipeline.

        Returns
        -------
        dict
            Document metadata after successful ingestion.
        """

        if upload is None:
            raise ValueError("No file uploaded.")

        filename = (upload.filename or "").strip()

        if not filename:
            raise ValueError("Uploaded file has no filename.")

        destination = self._unique_upload_path(filename)

        try:

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with destination.open("wb") as output_file:
                shutil.copyfileobj(
                    upload.file,
                    output_file
                )

            print("=" * 80)
            print("UPLOAD SUCCESS")
            print("Saved File :", destination.name)
            print("Location   :", destination)
            print("=" * 80)

            document = await asyncio.to_thread(
                self.ingest_path,
                destination,
            )

            return document

        except Exception as e:

            if destination.exists():
                destination.unlink()

            raise RuntimeError(
                f"Unable to ingest '{filename}'."
            ) from e

        finally:

            try:
                upload.file.close()
            except Exception:
                pass


    # ==========================================================
    # Knowledge Base
    # ==========================================================

    def clear_knowledge_base(self) -> dict[str, Any]:
        """
        Completely reset IndustrialMind.

        Clears

        • SQLite metadata
        • Qdrant vectors
        • Knowledge Graph

        Uploaded files are intentionally NOT deleted.
        """

        print("=" * 80)
        print("CLEARING KNOWLEDGE BASE")
        print("=" * 80)

        cleared = {
            "sqlite": False,
            "qdrant": False,
            "graph": False
        }

        try:

            self.store.clear()

            cleared["sqlite"] = True

            print("✓ SQLite cleared")

        except Exception as e:

            print("SQLite Clear Error:", e)

        try:

            self.vector_store.clear()

            cleared["qdrant"] = True

            print("✓ Qdrant cleared")

        except Exception as e:

            print("Qdrant Clear Error:", e)

        try:

            self.graph.build([])

            cleared["graph"] = True

            print("✓ Knowledge Graph cleared")

        except Exception as e:

            print("Knowledge Graph Error:", e)

        print("=" * 80)
        print("KNOWLEDGE BASE RESET COMPLETE")
        print("=" * 80)

        return {

            "message": "Knowledge base cleared successfully.",

            "status": cleared
        }


        # ==========================================================
    # Helper Methods
    # ==========================================================

    def _unique_upload_path(self, filename: str) -> Path:
        """
        Generate a unique filename inside the upload directory.

        Example
        -------
        report.pdf
        report_2.pdf
        report_3.pdf
        """

        # Remove unsafe characters
        safe_name = "".join(
            c if c.isalnum() or c in "._-" else "_"
            for c in filename
        ).strip()

        if not safe_name:
            safe_name = "document"

        destination = UPLOAD_FOLDER / safe_name

        if not destination.exists():
            return destination

        stem = destination.stem
        suffix = destination.suffix

        counter = 2

        while True:

            candidate = UPLOAD_FOLDER / f"{stem}_{counter}{suffix}"

            if not candidate.exists():
                return candidate

            counter += 1


    # ==========================================================
    # Document ID
    # ==========================================================

    def _document_id(
        self,
        path: Path,
        text: str
    ) -> str:
        """
        Create a deterministic document ID.

        Same document
            -> same id

        Different document
            -> different id
        """

        hasher = hashlib.sha1()

        hasher.update(path.name.encode("utf-8"))

        hasher.update(str(path.stat().st_size).encode())

        hasher.update(text[:10000].encode(
            "utf-8",
            errors="ignore"
        ))

        return hasher.hexdigest()[:16]


    # ==========================================================
    # Qdrant Point ID
    # ==========================================================

    def _qdrant_id(
        self,
        chunk_id: str
    ) -> int:
        """
        Convert a chunk id into a deterministic
        integer accepted by Qdrant.
        """

        digest = hashlib.sha1(
            chunk_id.encode("utf-8")
        ).hexdigest()

        return int(digest[:15], 16)

        # ==========================================================
    # Image Understanding
    # ==========================================================

    def extract_image_text(self, path: Path) -> str:
        """
        Extract structured information from industrial images.

        Supported image types
        ---------------------
        • Equipment photos
        • Flowcharts
        • P&IDs
        • Tables
        • Engineering drawings
        • Screenshots
        • SOP images
        • Scanned documents

        Uses:
            1. OCR (Tesseract)
            2. Florence-2 Vision LLM
            3. Industrial prompt formatting
        """

        self.load_florence()

        print("=" * 80)
        print("IMAGE ANALYSIS")
        print(path.name)
        print("=" * 80)

        image = Image.open(path).convert("RGB")

        # ------------------------------------------------------
        # OCR PREPROCESSING
        # ------------------------------------------------------

        gray = image.convert("L")

        gray = ImageEnhance.Contrast(gray).enhance(2.5)

        ocr_text = pytesseract.image_to_string(
            gray,
            config="--oem 3 --psm 6"
        )

        # ------------------------------------------------------
        # CLEAN OCR
        # ------------------------------------------------------

        seen = set()
        cleaned_lines = []

        for line in ocr_text.splitlines():

            line = line.strip()

            if len(line) < 3:
                continue

            if line in seen:
                continue

            seen.add(line)

            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        # ------------------------------------------------------
        # FLORENCE IMAGE UNDERSTANDING
        # ------------------------------------------------------

        prompt = "<MORE_DETAILED_CAPTION>"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.inference_mode():

            generated_ids = self.vision_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=700,
                num_beams=3,
                do_sample=False
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        result = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=image.size
        )

        caption = result.get(
            "<MORE_DETAILED_CAPTION>",
            ""
        )

        # ------------------------------------------------------
        # BUILD INDUSTRIAL DOCUMENT
        # ------------------------------------------------------

        summary = f"""
    DOCUMENT TYPE
    -------------
    Industrial Image

    IMAGE FILE
    ----------
    {path.name}

    AI DESCRIPTION
    --------------
    {caption}

    VISIBLE TEXT
    ------------
    {cleaned_text}

    IMAGE ANALYSIS INSTRUCTIONS

    If this image is:

    1. Equipment
    - Identify the equipment.
    - Mention visible components.
    - Mention gauges, valves, pipes, motors,
        pumps, switches, sensors if visible.

    2. Flowchart
    - Explain every step.
    - Explain decision blocks.
    - Explain workflow.

    3. P&ID
    - Identify tags.
    - Mention instruments.
    - Explain process flow.

    4. Table
    - Summarize important values.

    5. SOP
    - Summarize procedure.
    - Mention safety instructions.

    6. Diagram
    - Explain relationships.
    - Explain process.

    7. Screenshot
    - Describe software screen.
    - Mention warnings or alarms.

    Generate a professional industrial explanation.
    """

        print("OCR Characters :", len(cleaned_text))
        print("Caption Length :", len(caption))
        print("=" * 80)

        return summary

    def _knowledge_chunk_indices(
        self,
        chunks: list[str],
        max_chunks: int,
    ) -> set[int]:
        """
        Select a evenly spaced subset of chunks for LLM knowledge extraction.

        Large documents can produce hundreds of chunks. Running Groq once per
        chunk makes uploads appear stuck for 20+ minutes.
        """
        eligible = [
            index
            for index, chunk in enumerate(chunks)
            if len(chunk.strip()) >= 40
        ]

        if max_chunks <= 0 or not eligible:
            return set()

        if len(eligible) <= max_chunks:
            return set(eligible)

        step = len(eligible) / max_chunks
        return {
            eligible[int(index * step)]
            for index in range(max_chunks)
        }

    # ==========================================================
    # Document Ingestion Pipeline
    # Part 1.4A
    # ==========================================================

    def ingest_path(self, path: Path) -> dict[str, Any]:
        """
        Ingest a document into IndustrialMind AI.

        Pipeline
        --------
        1. Detect document type
        2. Extract text
        3. Validate extracted content
        4. Continue to chunking (Part 1.4B)
        """

        start_time = time.time()

        print("=" * 80)
        print("INGESTION STARTED")
        print("=" * 80)
        print(f"File : {path.name}")

        if not path.exists():
            raise FileNotFoundError(path)

        # --------------------------------------------------
        # Detect document type
        # --------------------------------------------------

        document_type = FileDetector.detect(str(path))

        print("Document Type :", document_type)

        if document_type == "Unsupported":
            raise ValueError(
                f"Unsupported document type : {path.suffix}"
            )

        suffix = path.suffix.lower()

        IMAGE_TYPES = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp"
        }

        loader_name = ""

        # --------------------------------------------------
        # Image Processing
        # --------------------------------------------------

        if suffix in IMAGE_TYPES:

            print("Processing image with Florence-2...")

            text = self.extract_image_text(path)

            loader_name = "Florence-2"

        # --------------------------------------------------
        # Plain Text
        # --------------------------------------------------

        elif suffix == ".txt":

            print("Reading text file...")

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            loader_name = "Plain Text"

        # --------------------------------------------------
        # Docling Documents
        # --------------------------------------------------

        else:

            print(f"Processing document ({suffix})...")

            loaded = self.loader.load(str(path))

            loader_name = loaded.get(
                "loader",
                "unknown"
            )

            text = loaded.get(
                "text",
                ""
            )

            print("=" * 80)
            print("FIRST 1000 CHARACTERS")
            print(text[:1000])
            print("=" * 80)

            print(
                f"Loader Used : {loader_name}"
            )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if text is None:
            text = ""

        text = text.strip()

        if not text:

            raise ValueError(
                f"No readable content found in '{path.name}'."
            )

        print("=" * 80)
        print("TEXT EXTRACTION COMPLETE")
        print("=" * 80)
        print("Characters :", len(text))
        print("Words      :", len(text.split()))
        print("Loader     :", loader_name)
        print("Elapsed    :", round(time.time() - start_time, 2), "sec")
        print("=" * 80)

        # ==========================================================
        # Continue in Part 1.4B
        # ==========================================================

            # --------------------------------------------------
        # Chunking
        # --------------------------------------------------

        print("=" * 80)
        print("CHUNKING DOCUMENT")
        print("=" * 80)

        chunks = chunk_text(text)

        if not chunks:
            raise ValueError(
                f"No chunks generated from '{path.name}'."
            )

        print(f"Total Chunks : {len(chunks)}")

        total_words = sum(
            len(chunk.split())
            for chunk in chunks
        )

        average_words = (
            total_words / len(chunks)
        )

        print(
            f"Average Words / Chunk : {average_words:.1f}"
        )

        # --------------------------------------------------
        # Entity Extraction
        # --------------------------------------------------

        print("=" * 80)
        print("ENTITY EXTRACTION")
        print("=" * 80)

        entity_start = time.time()

        chunk_entities = []

        total_entities = 0

        for index, chunk in enumerate(chunks):

            print(
                f"Chunk {index + 1}/{len(chunks)}",
                end="\r"
            )

            try:

                entities = self.entities.extract(chunk)

            except Exception as e:

                print(
                    f"\nEntity extraction failed "
                    f"for chunk {index}: {e}"
                )

                entities = {}

            chunk_entities.append(entities)

            if isinstance(entities, dict):

                total_entities += sum(
                    len(values)
                    for values in entities.values()
                    if isinstance(values, list)
                )

        print()

        print(
            f"Entity Extraction Time : "
            f"{time.time()-entity_start:.2f} sec"
        )

        print(
            f"Entities Found : {total_entities}"
        )

        # --------------------------------------------------
        # Merge Document Entities
        # --------------------------------------------------

        print("=" * 80)
        print("MERGING DOCUMENT ENTITIES")
        print("=" * 80)

        document_entities = self.entities.merge(
            chunk_entities
        )

        entity_summary = {}

        if isinstance(document_entities, dict):

            for key, values in document_entities.items():

                entity_summary[key] = len(values)

        print("Entity Summary")

        for key, count in entity_summary.items():

            print(
                f"  {key:<20} {count}"
            )

        print("=" * 80)

        # ==========================================================
        # Continue in Part 1.4C
        # ==========================================================


            # --------------------------------------------------
        # Contextual Knowledge Extraction
        # --------------------------------------------------

        print("=" * 80)
        print("KNOWLEDGE EXTRACTION")
        print("=" * 80)

        knowledge_start = time.time()

        chunk_knowledge = []

        all_knowledge_entities = []

        all_relationships = []

        skipped_chunks = 0

        extracted_chunks = 0

        knowledge_targets = set()

        if (
            KNOWLEDGE_EXTRACTION_ENABLED
            and GROQ_API_KEY
        ):
            knowledge_targets = self._knowledge_chunk_indices(
                chunks,
                MAX_KNOWLEDGE_CHUNKS,
            )

        print(
            f"LLM knowledge extraction: "
            f"{len(knowledge_targets)}/{len(chunks)} chunks "
            f"(cap={MAX_KNOWLEDGE_CHUNKS})"
        )

        for index, chunk in enumerate(chunks):

            print(
                f"Knowledge {index + 1}/{len(chunks)}",
                end="\r"
            )

            if (
                index not in knowledge_targets
                or len(chunk.strip()) < 40
            ):

                skipped_chunks += 1

                knowledge = {
                    "entities": [],
                    "relationships": []
                }

                chunk_knowledge.append(knowledge)

                continue

            try:

                # ------------------------------------------
                # LLM / Rule-based knowledge extraction
                # ------------------------------------------

                knowledge = self.entities.extract_knowledge(chunk)

                if knowledge is None:

                    knowledge = {
                        "entities": [],
                        "relationships": []
                    }

                extracted_chunks += 1

            except Exception as e:

                print(
                    f"\nKnowledge extraction failed "
                    f"for chunk {index}: {e}"
                )

                knowledge = {
                    "entities": [],
                    "relationships": []
                }

            chunk_knowledge.append(knowledge)

            all_knowledge_entities.extend(
                knowledge.get(
                    "entities",
                    []
                )
            )

            all_relationships.extend(
                knowledge.get(
                    "relationships",
                    []
                )
            )

        print()

        print(
            f"Knowledge Extraction Time : "
            f"{time.time()-knowledge_start:.2f} sec"
        )

        print(
            f"Processed Chunks : {extracted_chunks}"
        )

        print(
            f"Skipped Chunks : {skipped_chunks}"
        )

        # --------------------------------------------------
        # Remove Duplicate Knowledge Entities
        # --------------------------------------------------

        unique_entities = {}

        for entity in all_knowledge_entities:

            name = entity.get(
                "name",
                ""
            ).strip()

            entity_type = entity.get(
                "type",
                "Entity"
            ).strip()

            if not name:
                continue

            key = (
                entity_type.lower(),
                name.lower()
            )

            if key not in unique_entities:

                unique_entities[key] = {
                    "name": name,
                    "type": entity_type
                }

        knowledge_entities = sorted(
            unique_entities.values(),
            key=lambda x: (
                x["type"],
                x["name"]
            )
        )

        # --------------------------------------------------
        # Remove Duplicate Relationships
        # --------------------------------------------------

        unique_relationships = {}

        for relationship in all_relationships:

            source = relationship.get(
                "source",
                ""
            ).strip()

            target = relationship.get(
                "target",
                ""
            ).strip()

            relation = relationship.get(
                "type",
                ""
            ).strip().upper()

            if not source:
                continue

            if not target:
                continue

            if not relation:
                continue

            key = (
                source.lower(),
                target.lower(),
                relation
            )

            if key not in unique_relationships:

                unique_relationships[key] = {

                    "source": source,

                    "target": target,

                    "type": relation
                }

        document_relationships = sorted(

            unique_relationships.values(),

            key=lambda x: (

                x["type"],

                x["source"],

                x["target"]

            )

        )

        print("=" * 80)
        print("KNOWLEDGE SUMMARY")
        print("=" * 80)
        print("Knowledge Entities :", len(knowledge_entities))
        print("Relationships      :", len(document_relationships))
        print("=" * 80)

        # ==========================================================
        # Continue in Part 1.4D
        # ==========================================================

            # --------------------------------------------------
        # Generate Embeddings
        # --------------------------------------------------

        print("=" * 80)
        print("GENERATING EMBEDDINGS")
        print("=" * 80)

        embedding_start = time.time()

        vectors = self.embedder.encode(chunks)

        if len(vectors) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        print(f"Embeddings Generated : {len(vectors)}")
        print(
            f"Embedding Time : "
            f"{time.time() - embedding_start:.2f} sec"
        )

        # --------------------------------------------------
        # Document Metadata
        # --------------------------------------------------

        document_id = self._document_id(path, text)

        metadata = MetadataExtractor.extract(str(path))

        document = {

            "id": document_id,

            "file_name": path.name,

            "document_type": document_type,

            "source_path": str(path),

            "loader": loader_name,

            "chunks": len(chunks),

            "metadata": metadata,

            "entities": document_entities,

            "knowledge_entities": knowledge_entities,

            "relationships": document_relationships,
        }

        # --------------------------------------------------
        # Chunk Records
        # --------------------------------------------------

        print("=" * 80)
        print("BUILDING CHUNK RECORDS")
        print("=" * 80)

        chunk_records = []

        for index, chunk in enumerate(chunks):

            chunk_id = f"{document_id}:{index}"

            record = {

                "id": chunk_id,

                "chunk_id": chunk_id,

                "qdrant_id": self._qdrant_id(chunk_id),

                "document_id": document_id,

                "file_name": path.name,

                "document_type": document_type,

                "position": index,

                "text": chunk,

                "embedding": vectors[index],

                "entities": chunk_entities[index],

                "knowledge_entities":
                    chunk_knowledge[index].get(
                        "entities",
                        []
                    ),

                "relationships":
                    chunk_knowledge[index].get(
                        "relationships",
                        []
                    ),

                "chunk_length": len(chunk),

                "word_count": len(chunk.split()),

                "metadata": metadata,
            }

            chunk_records.append(record)

        print(
            f"Chunk Records : {len(chunk_records)}"
        )

        # --------------------------------------------------
        # Save SQLite
        # --------------------------------------------------

        print("=" * 80)
        print("SAVING SQLITE")
        print("=" * 80)

        self.store.add_document(
            document,
            chunk_records
        )

        # --------------------------------------------------
        # Save Qdrant
        # --------------------------------------------------

        print("=" * 80)
        print("SAVING QDRANT")
        print("=" * 80)

        self.vector_store.upsert_chunks(
            chunk_records
        )

        # --------------------------------------------------
        # Build Knowledge Graph
        # --------------------------------------------------

        print("=" * 80)
        print("UPDATING KNOWLEDGE GRAPH")
        print("=" * 80)

        self.graph.add_document(
            document,
            self.store.documents()
        )

        elapsed = time.time() - start_time

        print("=" * 80)
        print("INGESTION COMPLETE")
        print("=" * 80)

        print(f"Document ID : {document_id}")
        print(f"Chunks      : {len(chunks)}")
        print(f"Entities    : {len(knowledge_entities)}")
        print(f"Relations   : {len(document_relationships)}")
        print(f"Time        : {elapsed:.2f} sec")

        print("=" * 80)

        return document

    # ==========================================================
# Metrics
# ==========================================================

    def metrics(self) -> dict[str, Any]:
        """
        Return knowledge base statistics.
        """

        documents = self.store.documents()

        chunks = self.store.chunks_metadata()

        entity_counts = {}

        for document in documents:

            entities = document.get("entities", {})

            if not isinstance(entities, dict):
                continue

            for key, values in entities.items():

                if isinstance(values, list):
                    entity_counts[key] = (
                        entity_counts.get(key, 0)
                        + len(values)
                    )

        return {

            "documents": len(documents),

            "chunks": len(chunks),

            "entity_counts": entity_counts,

            "estimated_time_saved_minutes":
                max(0, len(documents) * 18),

            "vector_store": self.vector_store.available,

            "embedding_dimension": self.embedder.dimension,
        }
    def documents(self) -> list[dict[str, Any]]:
        """
        Return all ingested documents.
        """
        return self.store.documents()

    def delete_documents(self, document_ids: list[str]):

        deleted = []

        for document_id in document_ids:

            try:
                self.store.delete_document(document_id)

                self.vector_store.delete_document(document_id)

                deleted.append(document_id)

            except Exception as e:
                print(f"Failed to delete {document_id}: {e}")

        # Rebuild Knowledge Graph
        self.graph.build(self.store.documents())

        return {
            "status": "success",
            "deleted": deleted
        }
    
    # def graph_payload(self) -> dict[str, Any]:
    #     """
    #     Return the complete knowledge graph.
    #     """
    #     return self.graph.build(self.store.documents())

    def graph_payload(self, selected_documents=None) -> dict[str, Any]:
        documents = self.store.documents()

        if selected_documents:

            documents = [
                doc
                for doc in documents
                if doc["file_name"] in selected_documents
                or doc["id"] in selected_documents
            ]

        return self.graph.build(documents)
    
    def ask(
        self,
        question: str,
        top_k: int = 15,
        selected_documents=None
    ) -> dict[str, Any]:
        
        intent = self.classify_query(question)

        retrieval_top_k = top_k

        if intent["intent"] == "LIST":
            retrieval_top_k = 30

        elif intent["intent"] == "SUMMARY":
            retrieval_top_k = 20

        elif intent["intent"] == "COMPARE":
            retrieval_top_k = 20

        elif intent["intent"] == "COUNT":
            retrieval_top_k = 30

        hits = self.search(
            question,
            retrieval_top_k,
            selected_documents
        )

        if hits is None:
            hits = []

        hits = self._expand_neighbor_chunks(hits)

        hits = sorted(
            hits,
            key=lambda x: x["score"],
            reverse=True
        )

        if not hits:

            diagnostic = ""

            if not self.vector_store.available:

                diagnostic = (
                    " (Note: the vector search backend is not connected — "
                    "check that Qdrant is running and reachable.)"
                )

            return {
                "answer": "No relevant information found." + diagnostic,
                "confidence": "Low",
                "citations": [],
                "entities": {},
            }

        # ------------------------------------------------------
        # Reject very weak matches
        # ------------------------------------------------------

        if hits[0]["score"] < self.MIN_SIMILARITY:

            return {
                "answer":
                    "The uploaded documents do not contain enough relevant "
                    "information to answer this question.",
                "confidence": "Low",
                "citations": [],
                "entities": {},
            }

        # ------------------------------------------------------
        # IMPORTANT FIX
        # Sort the search results returned by search()
        # (The original code incorrectly sorted an empty list.)
        # ------------------------------------------------------

        hits = sorted(
            hits,
            key=lambda x: x["score"],
            reverse=True,
        )

        hits = [
            hit
            for hit in hits
            if hit["score"] >= self.MIN_SIMILARITY
        ]

        hits = hits[:retrieval_top_k]

        if not hits:

            return {
                "answer":
                    "No sufficiently relevant information was found in the uploaded documents.",
                "confidence": "Low",
                "citations": [],
                "entities": {},
            }

        # ------------------------------------------------------
        # Citations
        # ------------------------------------------------------

        citations = []

        for hit in hits:

            citations.append({

                "document_id": hit["document_id"],

                "file_name": hit["file_name"],

                "chunk_id": hit["id"],
                "page": hit.get("metadata", {}).get("page", "Unknown"),

                "score": round(hit["score"], 4),
                "matched_entities": hit.get("knowledge_entities", []),

                "excerpt": excerpt(hit["text"]),

            })

        # ------------------------------------------------------
        # Merge entities
        # ------------------------------------------------------

        top_hits = hits[:5]

        entities = self.entities.merge(
            [
                hit.get("entities", {})
                for hit in top_hits
            ]
        )

        # ------------------------------------------------------
        # Generate answer
        # ------------------------------------------------------

        answer = self._answer_with_llm(
            question,
            hits
        )

        if answer is None:

            print("LLM unavailable. Using extractive answer.")

            answer = self._extractive_answer(
                question,
                hits
            )

        # ------------------------------------------------------
        # Confidence
        # ------------------------------------------------------

        highest = max(
            hit["score"]
            for hit in hits
        )

        average = (
            sum(hit["score"] for hit in hits)
            / len(hits)
        )

        confidence_score = (
            highest * 0.5
            + average * 0.3
            + min(len(hits), 5) / 5 * 0.2
        )

        if confidence_score >= 0.40:

            confidence = "High"

        elif confidence_score >= 0.22:

            confidence = "Medium"

        else:

            confidence = "Low"

        if "do not contain enough relevant information" in answer.lower():

            confidence = "Low"

            confidence_score = min(confidence_score, 0.18)

        print("=" * 80)
        print("QUESTION ANSWERED")
        print("Confidence :", confidence)
        print("=" * 80)

        return {

            "answer": answer,

            "confidence": confidence,

            "confidence_score": round(confidence_score * 100, 1),

            "citations": citations,

            "entities": entities,

        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        selected_documents=None
    ) -> list[dict[str, Any]]:

        # --------------------------------------------------
        # Convert selected document IDs -> filenames
        # --------------------------------------------------

        document_names = {
            doc["id"]: doc["file_name"]
            for doc in self.store.documents()
        }

        selected_documents = [
            document_names.get(value, value)
            for value in (selected_documents or [])
        ]

        # --------------------------------------------------
        # Query Embedding
        # --------------------------------------------------

        # query_vector = self.embedder.encode([query])[0]

        # # --------------------------------------------------
        # # Vector Search
        # # --------------------------------------------------

        # vector_hits = self.vector_store.search(
        #     query_vector=query_vector,
        #     top_k=max(top_k * 4, 20),
        #     selected_documents=selected_documents,
        # )

        # --------------------------------------------------
        # Detect equipment tag queries
        # --------------------------------------------------

        equipment_pattern = r"^[A-Za-z]{1,5}-\d{1,5}$"

        is_equipment_tag = re.match(equipment_pattern, query.strip()) is not None

        # --------------------------------------------------
        # Vector Search
        # --------------------------------------------------

        if is_equipment_tag:
            vector_hits = []
        else:
            query_vector = self.embedder.encode([query])[0]

            vector_hits = self.vector_store.search(
                query_vector=query_vector,
                top_k=max(top_k * 4, 20),
                selected_documents=selected_documents,
            )

        print("=" * 80)
        print("QUESTION :", query)
        print("VECTOR HITS :", len(vector_hits))

        for hit in vector_hits[:5]:
            print(
                f"{hit['score']:.4f} | "
                f"{hit['file_name']} | "
                f"{hit.get('chunk_id', hit.get('id'))}"
            )

        # --------------------------------------------------
        # Keyword Search
        # --------------------------------------------------

        # keyword_hits = self.store.keyword_search(
        #     query,
        #     limit=10
        # )

        if is_equipment_tag:
            keyword_hits = self.store.keyword_search(
                f'"{query}"',
                limit=20
            )
        else:
            keyword_hits = self.store.keyword_search(
                query,
                limit=10
            )

        # --------------------------------------------------
        # Fallback lexical search
        # --------------------------------------------------

        if not keyword_hits:

            terms = {
                term.lower()
                for term in query.split()
                if len(term) > 2
            }

            scored = []

            for chunk in self.store.chunks_metadata():

                if (
                    selected_documents
                    and chunk["file_name"] not in selected_documents
                ):
                    continue

                matches = sum(
                    term in chunk["text"].lower()
                    for term in terms
                )

                if matches:
                    scored.append((matches, chunk))

            scored.sort(
                key=lambda x: x[0],
                reverse=True
            )

            keyword_hits = [
                chunk
                for _, chunk in scored[:10]
            ]

        print("KEYWORD HITS :", len(keyword_hits))

        # --------------------------------------------------
        # Hybrid Retrieval
        # --------------------------------------------------

        combined = {}

        # Vector Hits

        for hit in vector_hits:

            chunk_id = hit.get(
                "chunk_id",
                hit.get("id")
            )

            combined[chunk_id] = {

                **hit,

                "score": hit["score"]

            }

        # Keyword Hits

        for hit in keyword_hits:

            if (
                selected_documents
                and hit["file_name"] not in selected_documents
            ):
                continue

            chunk_id = hit.get(
                "chunk_id",
                hit.get("id")
            )

            if chunk_id in combined:

                combined[chunk_id]["score"] += self.KEYWORD_BOOST

            else:

                combined[chunk_id] = {

                    **hit,

                    "score": 0.15

                }

        hits = list(combined.values())

        # --------------------------------------------------
        # Balance results across documents
        # --------------------------------------------------

        from collections import defaultdict

        grouped = defaultdict(list)

        for hit in hits:

            grouped[hit["document_id"]].append(hit)

        balanced_hits = []

        for doc_hits in grouped.values():

            doc_hits.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            balanced_hits.extend(
                doc_hits[:3]
            )

        # IMPORTANT FIX
        hits = sorted(
            balanced_hits,
            key=lambda x: x["score"],
            reverse=True
        )

        # --------------------------------------------------
        # Load Complete Chunk Metadata
        # --------------------------------------------------

        chunk_ids = [

            hit.get(
                "chunk_id",
                hit.get("id")
            )

            for hit in hits

        ]

        chunks_by_id = {

            chunk["id"]: chunk

            for chunk in self.store.chunks_by_ids(
                chunk_ids
            )

        }

        final_hits = []

        for hit in hits:

            chunk_id = hit.get(
                "chunk_id",
                hit.get("id")
            )

            chunk = chunks_by_id.get(chunk_id)

            if chunk:

                final_hits.append({

                    **chunk,

                    "score": hit["score"]

                })

        # --------------------------------------------------
        # Remove duplicate chunks
        # --------------------------------------------------

        seen = set()

        unique = []

        for hit in final_hits:

            key = (

                hit["document_id"],

                hit["position"]

            )

            if key in seen:
                continue

            seen.add(key)

            unique.append(hit)

        print("=" * 80)
        print("FINAL SEARCH RESULTS")
        for hit in unique[:5]:
            print("=" * 80)
            print(hit["file_name"])
            print("Chunk:", hit["position"])
            print(hit["text"][:500])
            print("=" * 80)

        for hit in unique[:10]:

            print(
                f"{hit['score']:.3f} | "
                f"{hit['file_name']} | "
                f"Chunk {hit['position']}"
            )

        # return unique
        query_upper = query.upper()

        filtered = []

        pattern = rf"\b{re.escape(query_upper)}\b"

        for hit in unique:

            text = hit["text"].upper()

            entities = hit.get("entities", {})

            equipment_tags = []

            if isinstance(entities, dict):
                equipment_tags = [
                    str(x).upper()
                    for x in entities.get("equipment_tags", [])
                ]

            # Match complete equipment tag in text
            if re.search(pattern, text):
                filtered.append(hit)
                continue

            # Match complete equipment tag in extracted entities
            if any(re.search(pattern, tag) for tag in equipment_tags):
                filtered.append(hit)

        if filtered:
            return filtered

        return unique
    
    # ==========================================================
# Maintenance Intelligence
# ==========================================================

    def maintenance(
        self,
        equipment_tag: str
    ) -> dict[str, Any]:
        """
        Analyze maintenance information for a specific
        equipment tag.
        """

        print("=" * 80)
        print("MAINTENANCE ANALYSIS")
        print("Equipment :", equipment_tag)
        print("=" * 80)

        hits = self.search(equipment_tag)

        return self._maintenance_with_llm(
            equipment_tag,
            hits
        )

        # hits = self.search(equipment_tag)

        # maintenance = self._maintenance_with_llm(
        #     equipment_tag,
        #     hits
        # )

        # return maintenance


# ==========================================================
# Compliance Intelligence
# ==========================================================

    def compliance(
        self,
        standard: str
    ) -> dict[str, Any]:

        print("=" * 80)
        print("COMPLIANCE ANALYSIS")
        print("Standard :", standard)
        print("=" * 80)

        # hits = self.search(standard)
        hits = self.store.chunks_metadata()

        return self._compliance_with_llm(
            standard,
            hits
        )

# ==========================================================
# Lessons Learned Intelligence
# ==========================================================

    def lessons(self) -> dict[str, Any]:
        """
        Generate lessons learned summary from all
        uploaded documents.
        """

        print("=" * 80)
        print("GENERATING LESSONS LEARNED")
        print("=" * 80)

        return self.lessons_agent.summarize(
            self.store.chunks_metadata()
        )

    def _extractive_answer(
        self,
        question: str,
        hits: list[dict[str, Any]]
    ) -> str:
        """
        Fallback answer when the LLM is unavailable.
        """

        if not hits:
            return "No relevant information found."

        top_chunks = []

        for hit in hits[:3]:

            top_chunks.append(hit["text"])

        text = "\n\n".join(top_chunks)

        return (
            "Based on the uploaded documents:\n\n"
            + text[:1800]
        )
    
    def _answer_with_llm(
        self,
        question: str,
        hits: list[dict[str, Any]]
    ) -> str | None:

        if not GROQ_API_KEY:
            return None

        if not hits:
            return None

        try:

            from langchain_groq import ChatGroq

            hits = hits[:self.MAX_CONTEXT_CHUNKS]

            context = []

            for hit in hits:

                context.append(
                    f"""
    =================================================
    SOURCE DOCUMENT : {hit['file_name']}
    # Chunk Score     : {hit['score']:.3f}

    Industrial Entities:
    {hit.get("knowledge_entities", [])}

    Relationships:
    {hit.get("relationships", [])}

    CONTENT
    -------
    {hit['text']}
    """
                )

            context = "\n\n".join(context)

            question_lower = question.lower()

            if any(
                word in question_lower
                for word in [
                    "list",
                    "what are",
                    "which"
                ]
            ):

                format_instruction = "Return a bullet list."

            elif any(
                word in question_lower
                for word in [
                    "how",
                    "steps",
                    "procedure"
                ]
            ):

                format_instruction = "Return numbered steps."

            elif any(
                word in question_lower
                for word in [
                    "compare",
                    "difference"
                ]
            ):

                format_instruction = "Return a comparison table."

            else:

                format_instruction = "Return a concise paragraph."

            prompt = f"""
    You are IndustrialMind AI, an Industrial Knowledge Intelligence Assistant.

    You assist maintenance engineers, plant operators, quality engineers, and safety officers.

    Use ONLY the retrieved context.

    When answering:

    1. Identify the equipment involved.
    2. Mention failures or incidents if present.
    3. Mention root causes.
    4. Mention corrective actions.
    5. Mention preventive actions.
    6. Mention standards or regulations (ISO, OISD, PESO, etc.).
    7. Mention confidence if information is incomplete.
    8. Never invent facts.
    9. If multiple documents disagree, explain the difference.
    10. If information is missing, clearly say so.

    Rules:

    1. Never invent facts.

    2. Use only the supplied context.

    3. If multiple documents contain relevant information,
    combine them.

    4. If the answer cannot be found, reply exactly:

    "The uploaded documents do not contain enough relevant information."

    Question:

    {question}

    Context:

    {context}

    Formatting:

    {format_instruction}

    Answer:
    """

            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=400,
            )

            response = llm.invoke(prompt)

            if response is None:
                return None

            if not hasattr(response, "content"):
                return None

            answer = response.content.strip()

            if not answer:
                return None

            print("=" * 80)
            print("LLM ANSWER")
            print("=" * 80)
            print(answer)
            print("=" * 80)

            return answer

        except Exception as e:

            print("=" * 80)
            print("GROQ ERROR")
            print(e)
            print("=" * 80)

            return None
        
    def _maintenance_with_llm(
        self,
        equipment_tag: str,
        hits: list[dict[str, Any]]
    ) -> dict[str, Any]:

        import json

        if not GROQ_API_KEY:
            return self.maintenance_agent.analyze(
                equipment_tag,
                self.store.chunks_metadata()
            )

        if not hits:
            return {
                "equipment_tag": equipment_tag,
                "risk_level": "Unknown",
                "failure_modes": [],
                "recommendations": [
                    "No relevant maintenance information found."
                ],
                "evidence": []
            }

        try:

            from langchain_groq import ChatGroq

            hits = hits[:self.MAX_CONTEXT_CHUNKS]
            hits = hits[:2]

            context = []

            evidence = []

            for hit in hits:

                seen = set()

                for hit in hits:

                    key = (
                        hit["file_name"],
                        hit["position"]
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                evidence.append({
                    "file_name": hit["file_name"],
                    "chunk_id": hit["id"],
                    "page": hit.get("metadata", {}).get("page", "Unknown"),
                    "score": round(hit["score"], 3),
                    "excerpt": excerpt(hit["text"])
                })

                context.append(
                    f"""
    SOURCE DOCUMENT : {hit['file_name']}

    Knowledge Entities:
    {hit.get("knowledge_entities", [])}

    Relationships:
    {hit.get("relationships", [])}

    CONTENT
    -------
    {hit['text']}
    """
                )

            context = "\n\n".join(context)

            prompt = f"""
    You are an experienced Industrial Maintenance Engineer.

    Analyze ONLY the retrieved document context.

    Equipment requested:
    {equipment_tag}

    Retrieved Context:
    {context}

    Instructions:

    1. Identify what asset or system this document describes.

    2. Determine whether the overall maintenance risk is:
    - High
    - Medium
    - Low
    - Unknown

    If the retrieved document is a software manual, system manual,
    design document, user guide, operating manual,
    configuration guide, or reference manual:

    - This document is informational, not incident evidence.
    - Do NOT infer failures.
    If the retrieved document is a system manual, user manual,
    design document, operating manual or software manual:

    Return

    Risk Level = Unknown

    Failure Modes = []

    unless the document explicitly describes an actual failure,
    alarm, incident, defect or abnormal operating condition.
    - Do NOT infer abnormal conditions.
    - Leave failure_modes empty unless an actual failure is explicitly reported.
    - Set risk_level to "Unknown" unless an operational risk is explicitly described.
    - Still provide preventive maintenance or software maintenance recommendations.
    - Still provide maintenance recommendations if preventive
    maintenance, inspection guidance, configuration guidance,
    or software maintenance procedures are described.

    3. Extract ONLY failure modes that are explicitly described as:
    - failures
    - faults
    - defects
    - alarms
    - incidents
    - abnormal conditions

    Do NOT infer a failure from:
    - diagnostic tests
    - maintenance procedures
    - inspections
    - operating instructions
    - configuration management
    - software maintenance
    - version control
    - change control
    - design documents
    - user manuals
    - system manuals

    The following words by themselves are NOT failure modes:

    Deviation
    Configuration
    Maintenance
    Procedure
    Inspection
    Update
    Upgrade
    Test
    Documentation
    Requirement
    Manual

    Only include a failure mode if the document explicitly states that it is an actual equipment or software failure, fault, defect, incident, alarm, or abnormal operating condition.

    Otherwise return:

    "failure_modes":[]

    4. Recommend maintenance actions supported by the retrieved document.

        If the document contains maintenance procedures,
        inspection instructions,
        preventive maintenance,
        operating guidance,
        or software maintenance guidance,
        summarize them as recommendations.

        Only return an empty list if absolutely no maintenance guidance exists.

    5. Identify the primary asset discussed in the retrieved context.

    Examples:
    - Pump P-101
    - Compressor C-203
    - Ground Software Maintenance Facility
    - GCID
    - Simulation Control Station

    If no explicit asset exists, return:

    "asset_name":"Unknown"

    Also classify the asset type:

    Examples:
    Pump
    Compressor
    Valve
    Software System
    Control System
    Motor
    Generator
    Pipeline
    Unknown

    6. Do NOT invent failures.

    7. If the document is about software,
    provide software maintenance recommendations.

    8. If it is mechanical,
    provide mechanical recommendations.

    9. If it is electrical,
    provide electrical recommendations.

    10. Never invent information.

    If no explicit failure exists,
    return:

    "failure_modes":[]

    If maintenance guidance exists,
    still provide recommendations.

    Only return:

    "recommendations":[]

    when absolutely no maintenance guidance is present.

    Return ONLY valid JSON.

    JSON format:

    {{
        "asset_name":"Pump P-101",
        "asset_type":"Pump",
        "risk_level":"Medium",
        "failure_modes":[
            {{
                "name":"Failure name",
                "description":"Short description",
                "occurrences":1
            }}
        ],
        "recommendations":[
            "Recommendation 1",
            "Recommendation 2"
        ]
    }}
    """

            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=300,
            )

            response = llm.invoke(prompt)
            print("="*80)
            print("RAW MAINTENANCE RESPONSE")
            print(response.content)
            print("="*80)

            answer = response.content.strip()

            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            answer = answer.strip()

            # result = json.loads(answer)
            import re

            match = re.search(r"\{.*\}", answer, re.DOTALL)

            if not match:
                raise ValueError("No valid JSON returned from LLM.")

            result = json.loads(match.group(0))

            return {
                "equipment_tag": result.get("asset_name", equipment_tag),
                "asset_type": result.get("asset_type", "Unknown"),
                "risk_level": result.get("risk_level", "Unknown"),
                "failure_modes": result.get("failure_modes", []),
                "recommendations": result.get("recommendations", []),
                "evidence": evidence,
            }

        except Exception as e:

            print("=" * 80)
            print("MAINTENANCE LLM ERROR")
            print(e)
            print("=" * 80)

            return self.maintenance_agent.analyze(
                equipment_tag,
                self.store.chunks_metadata()
            )
        
    def _compliance_with_llm(
        self,
        standard: str,
        hits: list[dict[str, Any]]
    ) -> dict[str, Any]:

        import json
        import re

        if not GROQ_API_KEY:
            return self.compliance_agent.assess(
                self.store.chunks_metadata(),
                standard
            )

        if not hits:
            return {
                "standard": standard,
                "compliance_score": 0,
                "status": "No Evidence",
                "requirements_covered": [],
                "compliance_gaps": [
                    "No relevant compliance information found."
                ],
                "audit_readiness": "Not Ready",
                "recommendations": [],
                "supporting_evidence": []
            }

        try:

            from langchain_groq import ChatGroq

            hits = hits[:15]

            context = []
            evidence = []

            for hit in hits:

                evidence.append({
                    "file_name": hit["file_name"],
                    "page": hit.get("metadata", {}).get("page", "Unknown"),
                    "excerpt": excerpt(hit["text"])
                })

                context.append(f"""
    SOURCE DOCUMENT : {hit['file_name']}

    CONTENT
    -------
    {hit['text']}
    """)

            context = "\n\n".join(context)

            prompt = f"""
    You are an Industrial Compliance Auditor.

    Evaluate ALL uploaded industrial documents against the selected regulatory standard.

    Selected Standard:
    {standard}

    The uploaded documents may include:

    - SOPs
    - Maintenance Records
    - Inspection Reports
    - Work Orders
    - Engineering Drawings
    - P&IDs
    - OEM Manuals
    - Safety Procedures

    Determine whether the uploaded documents provide evidence for compliance.

    Do NOT check only whether the regulation name appears.

    Instead compare the document content with the intent of the selected regulation.

    Return:

    - Compliance Score (0-100)
    - Compliance Status
    - Requirements Covered
    - Compliance Gaps
    - Recommendations
    - Audit Readiness

    Only use the supplied document context.

    Never invent evidence.

    Return ONLY valid JSON.

    Context:

    {context}
    """

            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=800
            )

            response = llm.invoke(prompt)

            answer = response.content.strip()

            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")

            match = re.search(r"\{.*\}", answer, re.DOTALL)

            if not match:
                raise ValueError("Invalid JSON")

            result = json.loads(match.group(0))

            return {
                "standard": standard,
                "compliance_score": result.get("compliance_score", 0),
                "status": result.get("status", "Unknown"),
                "requirements_covered": result.get("requirements_covered", []),
                "compliance_gaps": result.get("compliance_gaps", []),
                "audit_readiness": result.get("audit_readiness", "Unknown"),
                "recommendations": result.get("recommendations", []),
                "supporting_evidence": evidence
            }

        except Exception as e:

            print("=" * 80)
            print("COMPLIANCE LLM ERROR")
            print(e)
            print("=" * 80)

            return self.compliance_agent.assess(
                self.store.chunks_metadata(),
                standard
            )
        
    def _expand_neighbor_chunks(
        self,
        hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Add neighboring chunks around retrieved chunks to
        improve context continuity.
        """

        if self.NEIGHBOR_WINDOW <= 0:
            return hits

        all_chunks = self.store.chunks_metadata()

        lookup = {
            (c["document_id"], c["position"]): c
            for c in all_chunks
        }

        expanded = []

        seen = set()

        for hit in hits:

            doc = hit["document_id"]

            pos = hit["position"]

            for offset in range(
                -self.NEIGHBOR_WINDOW,
                self.NEIGHBOR_WINDOW + 1
            ):

                chunk = lookup.get(
                    (doc, pos + offset)
                )

                if not chunk:
                    continue

                key = (
                    chunk["document_id"],
                    chunk["position"]
                )

                if key in seen:
                    continue

                seen.add(key)

                expanded.append({
                    **chunk,
                    "score": hit["score"] * 0.95
                })

        return expanded
    
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
    
service = IndustrialMindService()