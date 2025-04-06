# mcp_cli/llm/llm_client.py
from mcp_cli.llm.providers.base import BaseLLMClient

def get_llm_client(provider="openai", model="gpt-4o-mini", api_key=None, api_base=None) -> BaseLLMClient:
    if provider == "openai":
        # import
        from mcp_cli.llm.providers.openai_client import OpenAILLMClient

        # return the open ai client
        return OpenAILLMClient(model=model, api_key=api_key, api_base=api_base)
    elif provider == "ollama":
        # import
        from mcp_cli.llm.providers.ollama_client import OllamaLLMClient

        # return the ollama client
        # return the ollama client
        return OllamaLLMClient(model=model)
    elif provider == "gemini":
        # import
        from mcp_cli.llm.providers.gemini_client import GeminiLLMClient
        # return the gemini client
        # Note: Gemini client uses GEMINI_API_KEY env var by default,
        # but we pass api_key=None here as it's not explicitly needed for init
        # unless overriding the env var. The model name is passed.
        return GeminiLLMClient(model=model, api_key=api_key) # Pass model and optional api_key override
    else:
        # unsupported provider
        raise ValueError(f"Unsupported provider: {provider}")
