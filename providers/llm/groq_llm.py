import logging
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from core.base_llm import BaseLLMProvider
from config.settings import Settings

logger = logging.getLogger(__name__)


class GroqLLMProvider(BaseLLMProvider):
    """Cloud LLM via Groq API. Requires GROQ_API_KEY in .env."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for GroqLLMProvider")
        self._settings = settings
        logger.info("GroqLLMProvider | model=%s", settings.groq_model)

    def get_model(self) -> BaseChatModel:
        return ChatGroq(
            model=self._settings.groq_model,
            api_key=self._settings.groq_api_key,
            temperature=self._settings.llm_temperature,
        )
