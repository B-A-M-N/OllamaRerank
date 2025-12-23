from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str
    original_index: int = Field(..., description="Original index of the document in the corpus")

# Request Models
class TruncateOptions(BaseModel):
    query_tokens: int = 256
    document_tokens: int = 512


class Options(BaseModel):
    seed: Optional[int] = None
    num_ctx: Optional[int] = None


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[Document] = Field(min_length=1)
    top_n: Optional[int] = None
    return_documents: bool = False
    truncate: Optional[TruncateOptions] = None
    batch_size: int = 16
    options: Optional[Options] = None
    policy_id: Optional[str] = None


# Response Models
class Result(BaseModel):
    index: int = Field(..., description="The original index of the document in the request's `documents` list.")
    original_corpus_index: int = Field(..., description="The original index of the document in the full corpus.")
    score: float
    document: Optional[str] = None
    facts: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    pairs: int
    query_tokens: int
    document_tokens: int
    total_tokens: int


class Timing(BaseModel):
    load_ms: int
    encode_ms: int
    score_ms: int
    total_ms: int


class RerankResponse(BaseModel):
    model: str
    query: str
    results: List[Result]
    usage: Optional[Usage] = None
    timing: Optional[Timing] = None
    warnings: Optional[List[dict]] = None


# OpenAI-ish Models
class OpenAIRerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str] = Field(min_length=1)
    top_n: Optional[int] = None



class OpenAIRerankResult(BaseModel):
    index: int
    relevance_score: float
    original_corpus_index: Optional[int] = None


class OpenAIUsage(BaseModel):
    total_tokens: int


class OpenAIRerankResponse(BaseModel):
    model: str
    data: List[OpenAIRerankResult]
    usage: Optional[OpenAIUsage] = None


# Error Models
class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
