from llmscribe._base import BaseGenerationAgent
from llmscribe.schema import (
    GenerationSample,
    MultiClassExample,
    MultiLabelExample,
    TextSample,
)


class GenerationAgent(BaseGenerationAgent):
    pass


class MultiClassGenerationAgent(
    BaseGenerationAgent[MultiClassExample, GenerationSample],
):
    """Agent specifically for multi-class classification data generation"""

    @property
    def _default_system_prompt(self) -> str:
        return """
        You are a data augmentation specialist for multi-class classification tasks.
        Generate new samples that match the style and have the same single label as the examples.

        Focus on:
        - Maintaining the same classification category
        - Creating diverse expressions of the same concept
        - Ensuring clear distinction from other classes
        """


class MultiLabelGenerationAgent(
    BaseGenerationAgent[MultiLabelExample, GenerationSample],
):
    """Agent specifically for multi-label classification data generation"""

    @property
    def _default_system_prompt(self) -> str:
        return """
        You are a data augmentation specialist for multi-label classification tasks.
        Generate new samples that can have multiple labels like the examples.

        Focus on:
        - Understanding label combinations and relationships
        - Creating samples with appropriate label sets
        - Maintaining semantic consistency across labels
        """


class TextGenerationAgent(BaseGenerationAgent[TextSample, GenerationSample]):
    """Agent for plain text generation without labels"""

    @property
    def _default_system_prompt(self) -> str:
        return """
        You are a text generation specialist. Create new text samples
        that match the style, tone, and content patterns of the examples.

        Focus on:
        - Preserving writing style and tone
        - Maintaining thematic consistency
        - Creating natural, diverse variations
        """
