from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class SampleType(Enum):
    """Enum to identify different types of samples"""

    PLAIN_TEXT = "plain_text"
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


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


class PromptTemplate:
    """Class to handle prompt templates and formatting"""

    @staticmethod
    def get_base_template() -> str:
        return """
Based on these {num_examples} input examples:
{examples_text}

Generate {num_samples} new samples that follow the same pattern and style.

Each generated sample should:
{requirements}

Focus on understanding the underlying patterns in the examples and creating variations that preserve the core characteristics.
"""

    @staticmethod
    def get_requirements_for_type(sample_type: SampleType) -> str:
        """Get requirements text based on sample type"""
        requirements_map = {
            SampleType.PLAIN_TEXT: (
                "- Include reasoning traces explaining why it matches the pattern\n"
                "- Be unique and diverse while maintaining quality"
            ),
            SampleType.SINGLE_LABEL: (
                "- Have the same type of label as the examples above\n"
                "- Include reasoning traces explaining why it matches the pattern\n"
                "- Be unique and diverse while maintaining quality"
            ),
            SampleType.MULTI_LABEL: (
                "- Have labels that follow the same pattern as the examples above "
                "(can be multiple labels)\n"
                "- Include reasoning traces explaining why it matches the pattern\n"
                "- Be unique and diverse while maintaining quality"
            ),
        }
        return requirements_map.get(
            sample_type,
            requirements_map[SampleType.PLAIN_TEXT],
        )
