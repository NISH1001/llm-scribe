from typing import cast

from loguru import logger
from pydantic_ai import Agent

from llmscribe._base import BaseGenerationAgent
from llmscribe.schema import GenerationSample, PromptTemplate, SampleType, TextSample
from llmscribe.utils import (
    create_dynamic_schema_multiclass,
    create_dynamic_schema_multilabel,
    detect_sample_type,
    extract_labels_from_examples,
)


class GenerationAgent(BaseGenerationAgent):
    pass


class GenerationAgentWithDynamicSchema[TI: TextSample, TO: GenerationSample](
    BaseGenerationAgent[TI, TO],
):
    """
    Generation agent that builds schemas at runtime based on input examples.

    This agent defers the creation of the pydantic-ai Agent until generation time,
    allowing it to build the appropriate schema based on the labels found in the examples.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        system_prompt: str | None = None,
        prompt_template: PromptTemplate | None = None,
        temperature: float = 0.25,
        debug: bool = False,
    ) -> None:
        """
        Initialize the dynamic schema generation agent.

        Note: Creates a temporary agent that will be replaced when examples are provided.

        Args:
            model (str): The model to use for generation.
            system_prompt (Optional[str]): The system prompt for the agent.
            prompt_template (Optional[PromptTemplate]): Template for prompt formatting.
            temperature (float): The temperature setting for generation.
            debug (bool): Whether to enable debug mode.
        """
        super().__init__(
            model=model,
            output_schema=cast(type[TO], GenerationSample),
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            temperature=temperature,
            debug=debug,
        )

    def _get_agent(self) -> Agent:
        return self.agent

    async def generate(
        self,
        examples: list[TI],
        num_samples: int = 5,
    ) -> list[TO]:
        """
        Generate new samples based on input examples

        Args:
            examples: List of input example samples to learn from
            num_samples: Number of new samples to generate

        Returns:
            List of newly generated samples
        """
        if not examples:
            logger.warning("No examples provided for generation. Returning empty list.")
            return []
        labels = extract_labels_from_examples(examples)
        output_schema = GenerationSample
        if not labels:
            logger.warning(
                "No labels found in examples. Defaulting to plain text generation.",
            )

        sample_type = detect_sample_type(examples)

        if sample_type == SampleType.SINGLE_LABEL:
            output_schema = create_dynamic_schema_multiclass(labels)
        elif sample_type == SampleType.MULTI_LABEL:
            output_schema = create_dynamic_schema_multilabel(labels)

        agent = super().build_pydantic_agent(
            model=self.model,
            output_schema=cast(type[TO], output_schema),
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )
        return await super()._generate(
            agent=agent,
            examples=examples,
            num_samples=num_samples,
        )
