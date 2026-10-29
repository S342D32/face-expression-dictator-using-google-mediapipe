import logging
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

from core.base_llm import BaseLLMProvider
from config.settings import Settings

logger = logging.getLogger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    """Local LLM via Ollama. No API key required."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        logger.info(
            "OllamaLLMProvider | model=%s | host=%s",
            settings.ollama_model,
            settings.ollama_host,
        )

    def get_model(self) -> BaseChatModel:
        return ChatOllama(
            model=self._settings.ollama_model,
            base_url=self._settings.ollama_host,
            temperature=self._settings.llm_temperature,
        )
