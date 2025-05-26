from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class BaseSample(BaseModel):
    pass


class TextSample(BaseSample):
    """Base class with text validation"""

    text: str = Field(..., description="The text content.")

    @field_validator("text")
    @classmethod
    def validate_text_quality(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Text too short")
        return v.strip()


class MultiClassExample(TextSample):
    """
    Represents an input example sample for training/reference
    Simple structure: just text + label
    """

    label: str = Field(..., description="The label for the example.")


class MultiLabelExample(TextSample):
    """
    Represents an input example sample for multi-label classification
    Simple structure: just text + labels
    """

    labels: List[str] = Field(..., description="List of labels for the example.")

    @property
    def label(self) -> str:
        """Returns the first label, or empty string if no labels exist."""
        return ",".join(self.labels) if self.labels else ""


class GenerationSample(TextSample):
    """
    Represents a generated sample with reasoning and metadata
    Rich structure: includes reasoning traces, confidence, etc.
    """

    label: str = Field(..., description="The predicted/generated label.")
    reasoning_traces: List[str] = Field(
        default_factory=list,
        description="Concise reasoning traces that resulted in the generation.",
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
