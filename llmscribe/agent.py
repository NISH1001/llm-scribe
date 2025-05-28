from typing import List, Optional

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from llmscribe.schema import GenerationSample, MultiClassExample


class GenerationAgent:
    """
    Class-based generation agent for creating augmented samples
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        output_schema: type = GenerationSample,
        system_prompt: Optional[str] = None,
        temperature: float = 0.25,
    ):
        self.model = model
        self.output_schema = output_schema
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.temperature = temperature

        # Initialize Pydantic AI agent
        self.agent = Agent(
            model=self.model,
            output_type=List[self.output_schema],
            system_prompt=self.system_prompt,
            model_settings=ModelSettings(
                temperature=self.temperature,
            ),
        )

    def _default_system_prompt(self) -> str:
        """Default system prompt for generation"""
        return """
        You are a data augmentation specialist. Given input examples,
        generate new, diverse samples that match the same style and label.

        For each generated sample, provide:
        1. The text content
        2. The appropriate label (same as examples)
        3. Reasoning traces explaining why this sample fits the pattern

        Requirements:
        - Generate samples similar in style and content to the examples
        - Keep the same label as the input examples
        - Make each sample unique and diverse
        - Provide clear reasoning for each generated sample
        - Maintain high quality and coherence
        """

    def build_prompt(
        self,
        examples: List[MultiClassExample],
        num_samples: int = 5,
    ) -> str:
        """
        Build the generation prompt from input examples

        Args:
            examples: List of example samples to learn from
            num_samples: Number of new samples to generate

        Returns:
            Formatted prompt string
        """
        # Format examples for the prompt
        examples_text = "\n".join(
            [
                f"- Text: '{sample.text}' | Label: '{sample.label}'"
                for sample in examples
            ],
        )

        prompt = f"""
        Based on these {len(examples)} input examples:

        {examples_text}

        Generate {num_samples} new samples that follow the same pattern and style.
        Each generated sample should:
        - Have the same label as the examples above
        - Include reasoning traces explaining why it matches the pattern
        - Be unique and diverse while maintaining quality

        Focus on understanding the underlying patterns in the examples and creating variations that preserve the core characteristics.
        """

        return prompt

    async def generate(
        self,
        examples: List[MultiClassExample],
        num_samples: int = 10,
    ) -> List[GenerationSample]:
        """
        Generate new samples based on input examples

        Args:
            examples: List of input example samples to learn from
            num_samples: Number of new samples to generate

        Returns:
            List of newly generated samples
        """

        prompt = self.build_prompt(examples, num_samples)

        try:
            result = await self.agent.run(prompt)
            return result.output
        except Exception as e:
            print(f"Generation failed: {e}")
            return []

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt and reinitialize agent"""
        self.system_prompt = new_prompt
        self.agent = Agent(
            model=self.model,
            output_type=List[self.output_schema],
            system_prompt=self.system_prompt,
        )

    def update_model(self, new_model: str) -> None:
        """Update the model and reinitialize agent"""
        self.model = new_model
        self.agent = Agent(
            model=self.model,
            output_type=List[self.output_schema],
            system_prompt=self.system_prompt,
        )
