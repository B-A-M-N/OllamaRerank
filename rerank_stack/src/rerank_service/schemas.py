from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class Document(BaseModel):
    text: str
    original_index: int = Field(..., description="Original index of the document in the corpus")
    similarity: Optional[float] = Field(
        default=None,
        description="Optional retrieval similarity to use as final tiebreaker."
    )

# Request Models
class TruncateOptions(BaseModel):
    query_tokens: int = 256
    document_tokens: int = 512


class Options(BaseModel):
    seed: Optional[int] = None
    num_ctx: Optional[int] = None
    debug: Optional[bool] = None


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
    model_config = ConfigDict(extra="allow")
    index: int = Field(..., description="The original index of the document in the request's `documents` list.")
    original_corpus_index: int = Field(..., description="The original index of the document in the full corpus.")
    score: float
    pre_rank_score: Optional[float] = Field(
        default=None, description="Continuous score before policy decision (debug/telemetry)."
    )
    rank_score: Optional[float] = Field(
        default=None, description="Continuous score after decision; 0 when rejected."
    )
    rank_score_reason: Optional[str] = Field(
        default=None, description="Why rank_score took its value (e.g., accepted, rejected)."
    )
    fine_score: Optional[int] = Field(
        default=None, description="Fine-grained continuous score (0-100) from tie-break pass."
    )
    fine_score_norm: Optional[float] = Field(
        default=None, description="Fine-grained score normalized to 0..1."
    )
    tie_breaker_used: Optional[bool] = Field(
        default=None, description="Whether fine scoring was attempted for this result."
    )
    tie_breaker_model: Optional[str] = Field(
        default=None, description="Model used for fine scoring."
    )
    tie_breaker_prompt_id: Optional[str] = Field(
        default=None, description="Prompt id/version used for fine scoring."
    )
    tie_breaker_error: Optional[str] = Field(
        default=None, description="Error message if fine scoring failed."
    )
    tie_breaker_latency_ms: Optional[int] = Field(
        default=None, description="Latency in ms for fine scoring."
    )
    score_raw: Optional[float] = Field(
        default=None, description="Unnormalized/raw score emitted by the policy/model."
    )
    score_norm: Optional[float] = Field(
        default=None, description="Score normalized to 0..1 for easier comparison."
    )
    tier: Optional[int] = Field(
        default=None, description="Discrete bucket (e.g., 0-3) used for gating."
    )
    decision: Optional[str] = Field(
        default=None, description="Policy decision such as ACCEPT or REJECT_TIER_*."
    )
    policy_id: Optional[str] = Field(
        default=None, description="Policy identifier used for this result."
    )
    document: Optional[str] = None
    facts: Dict[str, Any] = Field(default_factory=dict)



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
