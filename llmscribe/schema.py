from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class BaseSample(BaseModel):
    """Base class for all sample types with common configuration."""

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
    }


class TextSample(BaseSample):
    """Base class for samples containing text with validation."""

    text: str = Field(
        ...,
        min_length=10,
        description="The text content (minimum 10 characters).",
    )


class MultiClassExample(TextSample):
    """
    Training/reference example for multi-class classification.
    Contains text and a single label.
    """

    label: str = Field(
        ...,
        min_length=1,
        description="The class label for this example.",
    )


class MultiLabelExample(TextSample):
    """
    Represents an input example sample for multi-label classification
    which can have multiple labels.
    Simple structure: just text + labels
    """

    labels: List[str] = Field(
        ...,
        min_length=1,
        description="List of labels for the example.",
    )

    @field_validator("labels")
    @classmethod
    def validate_labels_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure no empty labels in the list."""
        if not v:
            raise ValueError("At least one label is required")

        empty_labels = [i for i, label in enumerate(v) if not label.strip()]
        if empty_labels:
            raise ValueError(f"Empty labels found at indices: {empty_labels}")

        return [label.strip() for label in v]

    @computed_field
    @property
    def label(self) -> str:
        """Comma-separated string of labels for multi-label classification."""
        return ",".join(self.labels) if self.labels else ""


class GenerationSample(TextSample):
    """
    Represents a generated sample with reasoning and metadata.
    Rich structure: includes reasoning traces, confidence, etc.
    """

    reasoning_traces: List[str] = Field(
        default_factory=list,
        description="Concise reasoning traces that resulted in the generation.",
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the generation (0.0 to 1.0).",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the generation.",
    )


class MultiClassGenerationSample(GenerationSample):
    """
    Represents a generated sample for multi-class classification.
    Inherits from GenerationSample with multi-class specific fields.
    """

    label: str = Field(
        ...,
        min_length=1,
        description=(
            "The predicted/generated class label."
            " It should match one of the labels used in the training examples."
        ),
    )


class MultiLabelGenerationSample(GenerationSample):
    """
    Represents a generated sample for multi-label classification.
    Inherits from GenerationSample with multi-label specific fields.
    """

    labels: List[str] = Field(
        default_factory=list,
        description=(
            "List of predicted/generated labels."
            " It can contain multiple labels, each representing a class."
            " Labels should match those used in the training examples."
            " At least one label is required."
        ),
    )

    @field_validator("labels")
    @classmethod
    def validate_labels_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure no empty labels in the list."""
        if not v:
            raise ValueError("At least one label is required")

        empty_labels = [i for i, label in enumerate(v) if not label.strip()]
        if empty_labels:
            raise ValueError(f"Empty labels found at indices: {empty_labels}")

        return [label.strip() for label in v]

    @computed_field
    @property
    def label(self) -> str:
        """Returns labels as a comma-separated string for compatibility."""
        return ",".join(self.labels) if self.labels else ""
