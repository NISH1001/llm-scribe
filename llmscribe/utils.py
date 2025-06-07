from enum import Enum
from typing import List, Sequence, Set

from pydantic import Field, computed_field, create_model

from llmscribe.schema import GenerationSample, SampleType, TextSample


def create_dynamic_schema_multiclass(allowed_labels: Set[str]) -> type:
    """
    Create a dynamic GenerationSample model with an Enum field for multiclass.
    """
    allowed_labels = set(allowed_labels)
    Label = Enum("Label", {label.upper(): label for label in allowed_labels})

    # Create the dynamic model with field description
    DynamicModel = create_model(
        "MultiClassGenerationSample",
        label=(
            Label,
            Field(
                ...,
                description=(
                    "The predicted/generated class label."
                    " It should match one of the labels used in the training examples."
                ),
            ),
        ),
        __base__=GenerationSample,
    )

    # Add docstring to the dynamic model
    DynamicModel.__doc__ = (
        "Represents a generated sample for multi-class classification.\n"
        "Inherits from GenerationSample with multi-class specific fields.\n"
        f"Allowed labels: {', '.join(sorted(allowed_labels))}"
    )

    return DynamicModel


def create_dynamic_schema_multilabel(allowed_labels: Set[str]) -> type:
    """
    Create a dynamic GenerationSample model with an Enum field for multilabel.
    """
    allowed_labels = set(allowed_labels)
    Label = Enum("Label", {label.upper(): label for label in allowed_labels})

    # Create the base model first with field description
    DynamicModel = create_model(
        "MultiLabelGenerationSample",
        labels=(
            List[Label],
            Field(
                default_factory=list,
                description=(
                    "List of predicted/generated labels."
                    " It can contain multiple labels, each representing a class."
                    " Labels should match those used in the training examples."
                    " At least one label is required."
                ),
            ),
        ),
        __base__=GenerationSample,
    )

    # Add docstring to the dynamic model
    DynamicModel.__doc__ = (
        "Represents a generated sample for multi-label classification.\n"
        "Inherits from GenerationSample with multi-label specific fields.\n"
        f"Allowed labels: {', '.join(sorted(allowed_labels))}"
    )

    # Add the computed field to the dynamic model
    @computed_field
    @property
    def label(self) -> str:
        """Returns labels as a comma-separated string for compatibility."""
        # Convert enum values back to their string representation
        label_strings = [label.value for label in self.labels] if self.labels else []
        return ",".join(label_strings)

    # Attach the computed field to the dynamic model
    DynamicModel.label = label

    return DynamicModel


def detect_sample_type(examples: Sequence[TextSample]) -> SampleType:
    """
    Detect the type of samples being processed

    Args:
        examples: List of example samples

    Returns:
        SampleType enum indicating the detected type
    """
    if not examples:
        return SampleType.PLAIN_TEXT

    first_sample = examples[0]

    # if no label
    if not hasattr(first_sample, "label"):
        return SampleType.PLAIN_TEXT

    # Check if it's multi-label (has 'labels' attribute with list)
    if hasattr(first_sample, "labels") and isinstance(first_sample.labels, list):
        return SampleType.MULTI_LABEL

    return SampleType.SINGLE_LABEL
