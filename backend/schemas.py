from typing import Any
from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str
    file_name: str
    chunk_id: str
    score: float
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=12)
    selected_documents: Optional[list[str]] = None


class AskResponse(BaseModel):
    answer: str
    confidence: str
    citations: list[Citation]
    entities: dict[str, list[str]]


class DocumentSummary(BaseModel):
    id: str
    file_name: str
    document_type: str
    chunks: int
    entities: dict[str, list[str]]
    metadata: dict[str, Any]


class UploadResponse(BaseModel):
    document: DocumentSummary
    message: str


class MaintenanceRequest(BaseModel):
    equipment_tag: str


class ComplianceRequest(BaseModel):
    standard: str = "Factory Act, OISD, PESO, environmental norms, ISO 9001"

class ComplianceResponse(BaseModel):
    standard: str
    compliance_score: int
    status: str
    requirements_covered: list[str]
    compliance_gaps: list[str]
    audit_readiness: str
    recommendations: list[str]
    supporting_evidence: list[dict[str, Any]]

class DeleteDocumentsRequest(BaseModel):
    document_ids: list[str]