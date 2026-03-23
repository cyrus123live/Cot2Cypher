"""Provider configuration for CoT generation."""

import os

# Model
MODEL = "gpt-oss-120b"

# Provider base URLs (all OpenAI-compatible)
PROVIDERS = {
    "cerebras": "https://api.cerebras.ai/v1",
    "galaxy": "https://api.galaxy.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "openai": "https://api.openai.com/v1",
}

# Model name per provider (may differ)
MODEL_NAMES = {
    "cerebras": "gpt-oss-120b",
    "galaxy": "gpt-oss-120b",
    "openrouter": "openai/gpt-oss-120b",
    "deepinfra": "openai/gpt-oss-120b",
    "openai": "gpt-oss-120b",
}


def get_provider() -> str:
    return os.environ.get("COT_PROVIDER", "galaxy")


def get_base_url() -> str:
    provider = get_provider()
    return PROVIDERS[provider]


def get_model() -> str:
    provider = get_provider()
    return MODEL_NAMES.get(provider, MODEL)


def get_api_key() -> str:
    key = os.environ.get("COT_API_KEY", "")
    if not key:
        raise ValueError("Set COT_API_KEY environment variable")
    return key


# Generation defaults
MAX_CONCURRENCY = 50
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# Paths
OUTPUT_PATH = "data/cot_training_data.jsonl"
