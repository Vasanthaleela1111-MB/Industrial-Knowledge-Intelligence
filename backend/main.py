from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import traceback

from backend.schemas import (
    AskRequest,
    AskResponse,
    ComplianceRequest,
    ComplianceResponse,
    DocumentSummary,
    MaintenanceRequest,
    UploadResponse,
    DeleteDocumentsRequest,
)
from backend.services import service

app = FastAPI(
    title="IndustrialMind AI",
    version="1.0",
    description="Unified asset and operations brain for industrial knowledge intelligence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "IndustrialMind AI API Running",
        "qdrant_connected": service.vector_store.available,
        "storage": {
            "raw_documents": "Persistent upload volume",
            "metadata_and_chunks": "SQLite operational database",
            "semantic_search": "Qdrant vector database with local fallback",
            "knowledge_graph": "Neo4j graph database with JSON export fallback",
        },
        "capabilities": [
            "Docling based universal document ingestion",
            "RAG expert copilot with citations",
            "Industrial entity extraction",
            "Knowledge graph export",
            "Maintenance, compliance, and lessons learned agents",
        ],
    }


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        document = await service.ingest_upload(file)
        return {
                "document": DocumentSummary(**document),
                "message": "Document stored successfully.",
            }

    except Exception as exc:
        import traceback

        print("=" * 80)
        print("UPLOAD FAILED")
        traceback.print_exc()
        print("=" * 80)

        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents():
    return [DocumentSummary(**document) for document in service.documents()]

@app.post("/delete-documents")
def delete_documents(request: DeleteDocumentsRequest):

    return service.delete_documents(
        request.document_ids
    )

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = service.ask(
                request.question,
                request.top_k,
                getattr(request, "selected_documents", None),
            )
        print("=" * 60)
        print("RESULT:", result)
        print("ENTITIES TYPE:", type(result.get("entities")))
        print("ENTITIES VALUE:", result.get("entities"))
        print("=" * 60)

        return AskResponse(**result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/knowledge-graph")
# def knowledge_graph():
#     return service.graph_payload()

from typing import Optional

@app.get("/knowledge-graph")
def knowledge_graph(selected_documents: Optional[str] = None):

    docs = []

    if selected_documents:
        docs = [
            x.strip()
            for x in selected_documents.split(",")
            if x.strip()
        ]

    return service.graph_payload(docs)


@app.post("/maintenance")
def maintenance(request: MaintenanceRequest):
    return service.maintenance(request.equipment_tag)


@app.post("/compliance", response_model=ComplianceResponse)
def compliance(request: ComplianceRequest):
    return service.compliance(request.standard)


@app.get("/lessons")
def lessons():
    return service.lessons()


@app.get("/metrics")
def metrics():
    return service.metrics()

@app.post("/clear")
def clear():
    service.clear_knowledge_base()
    return {
        "status": "success",
        "message": "Knowledge base cleared successfully."
    }