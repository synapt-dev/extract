"""synapt-extract: SynaptExtraction IL v1 schema, validation, and finalization."""

from synapt_extract.schema import (
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
from synapt_extract.validate import validate_extraction, ValidationResult, ValidationError
from synapt_extract.finalize import finalize_extraction, FinalizeContext, FinalizeResult
from synapt_extract.prompt import build_extraction_prompt, resolve_capabilities
from synapt_extract.builder import (
    ExtractionBuilder,
    build_finalized_extraction_schema,
    build_extraction_schema,
    build_extraction_response_format,
    create_extraction_builder,
)
from synapt_extract.extract import extract, run_extraction, ExtractResult, UsageSummary

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
    "resolve_capabilities",
    "ExtractionBuilder",
    "build_finalized_extraction_schema",
    "build_extraction_schema",
    "build_extraction_response_format",
    "create_extraction_builder",
    "extract",
    "run_extraction",
    "ExtractResult",
    "UsageSummary",
]
