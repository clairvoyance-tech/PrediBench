from predibench.agent.models import ModelInfo

MODELS_BY_PROVIDER = {
    "xai": [
        ModelInfo(
            model_id="grok-4-0709",
            model_pretty_name="Grok 4",
            inference_provider="xai",
            company_pretty_name="xAI",
        ),
    ],
    "openai": [
        ModelInfo(
            model_id="gpt-5",
            model_pretty_name="GPT-5",
            inference_provider="openai",
            company_pretty_name="OpenAI",
            agent_type="toolcalling",
            sdk="openai",
        ),
        ModelInfo(
            model_id="gpt-5-mini",
            model_pretty_name="GPT-5 Mini",
            inference_provider="openai",
            company_pretty_name="OpenAI",
            agent_type="toolcalling",
            sdk="openai",
        ),
        ModelInfo(
            model_id="gpt-4.1",
            model_pretty_name="GPT-4.1",
            inference_provider="openai",
            company_pretty_name="OpenAI",
            agent_type="toolcalling",
            sdk="openai",
        ),
    ],
    "o3-deep-research": [
        ModelInfo(
            model_id="o3-deep-research",
            model_pretty_name="O3 Deep Research",
            inference_provider="openai",
            company_pretty_name="OpenAI",
            agent_type="deepresearch",
        ),
    ],
    "huggingface-openai": [
        ModelInfo(
            model_id="openai/gpt-oss-120b",
            model_pretty_name="GPT-OSS 120B",
            inference_provider="fireworks-ai",
            company_pretty_name="OpenAI",
            open_weights=True,
            agent_type="toolcalling",
        ),
    ],
    "huggingface-qwen": [
        ModelInfo(
            model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
            model_pretty_name="Qwen3 Coder 480B",
            inference_provider="fireworks-ai",
            company_pretty_name="Qwen",
            open_weights=True,
        ),
        ModelInfo(
            model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
            model_pretty_name="Qwen3 235B",
            inference_provider="fireworks-ai",
            company_pretty_name="Qwen",
            open_weights=True,
        ),
    ],
    "huggingface-deepseek": [
        ModelInfo(
            model_id="deepseek-ai/DeepSeek-R1",
            model_pretty_name="DeepSeek R1",
            inference_provider="fireworks-ai",
            company_pretty_name="DeepSeek",
            open_weights=True,
        ),
        ModelInfo(
            model_id="deepseek-ai/DeepSeek-V3.1",
            model_pretty_name="DeepSeek V3.1",
            inference_provider="fireworks-ai",
            company_pretty_name="DeepSeek",
            open_weights=True,
            agent_type="toolcalling",
        ),
    ],
    "huggingface-meta": [
        ModelInfo(
            model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            model_pretty_name="Llama 4 Maverick",
            inference_provider="fireworks-ai",
            company_pretty_name="Meta",
            open_weights=True,
        ),
        ModelInfo(
            model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            model_pretty_name="Llama 4 Scout",
            inference_provider="fireworks-ai",
            company_pretty_name="Meta",
            open_weights=True,
        ),
        ModelInfo(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            model_pretty_name="Llama 3.3 70B",
            inference_provider="groq",
            company_pretty_name="Meta",
            open_weights=True,
        ),
    ],
    "google": [
        ModelInfo(
            model_id="gemini-2.5-flash",
            model_pretty_name="Gemini 2.5 Flash",
            inference_provider="google",
            company_pretty_name="Google",
            agent_type="code",
        ),
        ModelInfo(
            model_id="gemini-2.5-pro",
            model_pretty_name="Gemini 2.5 Pro",
            inference_provider="google",
            company_pretty_name="Google",
            agent_type="code",
        ),
    ],
    "perplexity": [
        ModelInfo(
            model_id="sonar-deep-research",
            model_pretty_name="Sonar Deep Research",
            inference_provider="perplexity",
            company_pretty_name="Perplexity",
            agent_type="deepresearch",
        ),
    ],
    "anthropic": [
        ModelInfo(
            model_id="claude-sonnet-4-20250514",
            model_pretty_name="Claude Sonnet 4",
            inference_provider="anthropic",
            company_pretty_name="Anthropic",
        ),
        ModelInfo(
            model_id="claude-opus-4-1-20250805",
            model_pretty_name="Claude Opus 4.1",
            inference_provider="anthropic",
            company_pretty_name="Anthropic",
        ),
    ],
}

MODEL_MAP = [model for models in MODELS_BY_PROVIDER.values() for model in models]

BACKWARD_MODE_MODELS = [
    ModelInfo(
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        model_pretty_name="Qwen3 Coder 480B",
        inference_provider="fireworks-ai",
        company_pretty_name="Qwen",
        open_weights=True,
        agent_type="code",
    ),
    ModelInfo(
        model_id="openai/gpt-oss-120b",
        model_pretty_name="GPT-OSS 120B",
        inference_provider="fireworks-ai",
        company_pretty_name="OpenAI",
        open_weights=True,
        agent_type="toolcalling",
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-V3.1",
        model_pretty_name="DeepSeek V3.1",
        inference_provider="fireworks-ai",
        company_pretty_name="DeepSeek",
        open_weights=True,
        agent_type="code",
    ),
]
