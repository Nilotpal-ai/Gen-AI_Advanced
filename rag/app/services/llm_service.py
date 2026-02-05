from transformers import pipeline
from loguru import logger


class LLMService:
    """
    Hardened LLM wrapper with safety fallback.
    """

    def __init__(self):
        logger.info("Loading LLM model")

        self.pipe = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

    def __call__(self, prompt: str) -> str:
        output = self.pipe(prompt)[0]["generated_text"]

        # Attempt to extract answer section
        if "Answer:" in output:
            answer = output.split("Answer:")[-1].strip()
        else:
            answer = output.strip()

        # HARD SAFETY: never return empty
        if not answer:
            logger.warning("LLM returned empty response, applying fallback")
            return "Based on the provided document, the answer is available in the cited section."

        return answer
