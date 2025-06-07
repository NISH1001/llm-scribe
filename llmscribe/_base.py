from abc import ABC
from typing import Any, Dict, List

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from llmscribe.schema import GenerationSample, PromptTemplate, SampleType, TextSample
from llmscribe.utils import detect_sample_type


class BaseGenerationAgent[TI: TextSample, TO: GenerationSample](ABC):
    """
    Abstract base class for generation agents.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        output_schema: type[TO] = GenerationSample,
        system_prompt: str | None = None,
        prompt_template: PromptTemplate | None = None,
        temperature: float = 0.25,
        debug: bool = False,
    ) -> None:
        """
        Initialize the generation agent with model settings and prompts.

        Args:
            model (str): The model to use for generation.
            output_schema (type): The schema for the output.
            system_prompt (Optional[str]): The system prompt for the agent.
            temperature (float): The temperature setting for generation.
            debug (bool): Whether to enable debug mode.
        """
        self.model = model
        self.output_schema = output_schema
        self.system_prompt = system_prompt or self._default_system_prompt
        self.prompt_template = prompt_template or PromptTemplate()
        self.temperature = temperature
        self.debug = debug
        self.agent = Agent(
            model=self.model,
            output_type=list[self.output_schema],
            system_prompt=self.system_prompt,
            model_settings=ModelSettings(temperature=self.temperature),
        )

    @property
    def _default_system_prompt(self) -> str:
        """Default system prompt for generation"""
        return """
        You are a data augmentation specialist. Given input examples,
        generate new, diverse samples that match the same style and label.

        For each generated sample, provide:
        1. The text content
        2. The appropriate label (same as examples) if examples have labels
        3. Concise reasoning traces explaining why this sample fits the pattern

        Requirements:
        - Generate samples similar in style and content to the examples
        - Keep the same label as the input examples
        - Make each sample unique and diverse
        - Provide clear reasoning for each generated sample
        - Maintain high quality and coherence
        """

    def _format_examples_text(self, examples: List[TI], sample_type: SampleType) -> str:
        """
        Format examples into text representation

        Args:
            examples: List of example samples
            sample_type: The detected sample type

        Returns:
            Formatted string representation of examples
        """
        if sample_type == SampleType.PLAIN_TEXT:
            return "\n".join(f"- Text: '{sample.text}'" for sample in examples)
        else:
            # Both single and multi-label samples have a 'label' property
            return "\n".join(
                f"- Text: '{sample.text}' | Label: '{sample.label}'"
                for sample in examples
            )

    def _get_prompt_context(
        self,
        examples: List[TI],
        num_samples_to_generate: int,
    ) -> Dict[str, Any]:
        """
        Build context dictionary for prompt formatting

        Args:
            examples: List of example samples
            num_samples_to_generate: Number of samples to generate

        Returns:
            Dictionary with context for prompt formatting
        """
        sample_type = detect_sample_type(examples)

        return {
            "num_examples": len(examples),
            "examples_text": self._format_examples_text(examples, sample_type),
            "num_samples": num_samples_to_generate,
            "requirements": self.prompt_template.get_requirements_for_type(sample_type),
            "sample_type": sample_type,
        }

    def _customize_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook for subclasses to customize prompt context

        Args:
            context: Base context dictionary

        Returns:
            Modified context dictionary
        """
        # Default implementation returns context unchanged
        # Subclasses can override this to add custom context
        return context

    def _get_prompt_template(self) -> str:
        """
        Get the prompt template string

        Returns:
            Template string for formatting
        """
        return self.prompt_template.get_base_template()

    def build_prompt(
        self,
        examples: List[TI],
        num_samples_to_generate: int = 5,
    ) -> str:
        """
        Build the generation prompt from input examples

        Args:
            examples: List of example samples to learn from (any TextSample subclass)
            num_samples_to_generate: Number of new samples to generate

        Returns:
            Formatted prompt string
        """
        if not examples:
            raise ValueError("At least one example is required")

        # Build base context
        context = self._get_prompt_context(examples, num_samples_to_generate)

        # Allow customization by subclasses
        context = self._customize_prompt_context(context)

        # Format and return prompt
        template = self._get_prompt_template()
        return template.format(**context)

    async def generate(
        self,
        examples: List[TI],
        num_samples: int = 5,
    ) -> List[TO]:
        """
        Generate new samples based on input examples

        Args:
            examples: List of input example samples to learn from
            num_samples: Number of new samples to generate

        Returns:
            List of newly generated samples
        """

        prompt = self.build_prompt(examples, num_samples)
        if self.debug:
            logger.debug(f"Generated prompt:\n{prompt}\n")
        result = []
        try:
            result = await self.agent.run(prompt)
            return result.output
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = []
        return result
