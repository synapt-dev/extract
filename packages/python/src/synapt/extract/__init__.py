"""synapt-extract: SynaptExtraction IL v1 schema, validation, and finalization."""

#: The version of this package, available at runtime.
#:
#: Consumers that record which extractor produced a document should read this
#: rather than hand-copying a version string, so the recorded value is evidence
#: of what ran instead of a claim about it.
#:
#: Kept in step with ``packages/python/pyproject.toml`` and the TypeScript
#: package; ``scripts/bump-version.sh`` updates all three and
#: ``tests/python/test_version.py`` fails if any one of them drifts.
__version__ = "0.6.1"

from synapt.extract.schema import (
    SynaptExtraction,
    SynaptEntity,
    SynaptGoal,
    SynaptFact,
    SynaptQuestion,
    SynaptAction,
    SynaptDecision,
    SynaptSentiment,
    SynaptSourceMetadata,
    SynaptRelation,
    SynaptSourceRef,
    SynaptEmbedding,
    SynaptAssertionSignals,
    SynaptTemporalRef,
)
from synapt.extract.validate import validate_extraction, ValidationResult, ValidationError
from synapt.extract.finalize import finalize_extraction, FinalizeContext, FinalizeResult
from synapt.extract.prompt import (
    build_extraction_prompt,
    capability_embedding_input,
    profile_capabilities,
    resolve_capabilities,
    CANONICAL_ORDER,
    CAPABILITY_REGISTRY,
    STANDARD_EMBEDDING_INPUTS,
)
from synapt.extract.builder import (
    ExtractionBuilder,
    build_finalized_extraction_schema,
    build_extraction_schema,
    build_extraction_response_format,
    create_extraction_builder,
)
from synapt.extract.extract import (
    extract,
    normalize_llm_response,
    run_extraction,
    EmbeddingCallback,
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractCallbacks,
    ExtractResult,
    LlmCallback,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LogCallback,
    LogEntry,
    NormalizedLlmResponse,
    UsageSummary,
)
from synapt.extract.artifacts import (
    create_artifact_bundle,
    sha256_text,
    write_artifact_bundle,
)
from synapt.extract.openai import (
    extract_openai,
    OpenAIExtractResult,
)
from synapt.extract.batch import (
    BatchFailureReason,
    BatchInferRequest,
    BatchUnit,
    BatchUnitResult,
    extract_batch,
)

__all__ = [
    "SynaptExtraction",
    "SynaptEntity",
    "SynaptGoal",
    "SynaptFact",
    "SynaptQuestion",
    "SynaptAction",
    "SynaptDecision",
    "SynaptSentiment",
    "SynaptSourceMetadata",
    "SynaptRelation",
    "SynaptSourceRef",
    "SynaptEmbedding",
    "SynaptAssertionSignals",
    "SynaptTemporalRef",
    "validate_extraction",
    "ValidationResult",
    "ValidationError",
    "finalize_extraction",
    "FinalizeContext",
    "FinalizeResult",
    "build_extraction_prompt",
    "capability_embedding_input",
    "profile_capabilities",
    "resolve_capabilities",
    "CANONICAL_ORDER",
    "CAPABILITY_REGISTRY",
    "STANDARD_EMBEDDING_INPUTS",
    "ExtractionBuilder",
    "build_finalized_extraction_schema",
    "build_extraction_schema",
    "build_extraction_response_format",
    "create_extraction_builder",
    "extract",
    "normalize_llm_response",
    "run_extraction",
    "EmbeddingCallback",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ExtractCallbacks",
    "ExtractResult",
    "LlmCallback",
    "LlmMessage",
    "LlmRequest",
    "LlmResponse",
    "LogCallback",
    "LogEntry",
    "NormalizedLlmResponse",
    "UsageSummary",
    "create_artifact_bundle",
    "sha256_text",
    "write_artifact_bundle",
    "extract_openai",
    "OpenAIExtractResult",
    "BatchFailureReason",
    "BatchInferRequest",
    "BatchUnit",
    "BatchUnitResult",
    "extract_batch",
    "__version__",
]
