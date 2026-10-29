import logging
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from core.base_llm import BaseLLMProvider
from config.settings import Settings

logger = logging.getLogger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """
    Cloud LLM via OpenAI API (or any OpenAI-compatible endpoint).
    Requires OPENAI_API_KEY in .env.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAILLMProvider")
        self._settings = settings
        logger.info("OpenAILLMProvider | model=%s", settings.openai_model)

    def get_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self._settings.openai_model,
            api_key=self._settings.openai_api_key,
            temperature=self._settings.llm_temperature,
        )
