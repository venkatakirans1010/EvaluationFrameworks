from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class DatalabSettings:
    api_key: str
    endpoint: str = "https://www.datalab.to/api/v1/marker"
    ocr_endpoint: str = "https://www.datalab.to/api/v1/ocr"
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 120.0
    fallback_endpoints: Tuple[str, ...] = ()
    ocr_fallback_endpoints: Tuple[str, ...] = ()
    # Multi-run evaluation configuration
    multi_run_enabled: bool = False
    multi_run_runs: int = 3
    multi_run_require_all_success: bool = False


@dataclass(frozen=True)
class DeepInfraSettings:
    """Settings for DeepInfra (DeepSeek OCR via OpenAI-compatible API)."""
    api_token: str
    base_url: str = "https://api.deepinfra.com/v1/openai"
    model: str = "deepseek-ai/DeepSeek-OCR"
    max_tokens: int = 4096
    temperature: float = 0.1


@dataclass(frozen=True)
class OpenRouterSettings:
    """Settings for OpenRouter (Qwen3 VL via OpenAI-compatible API)."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen3-vl-8b-instruct"  # Replace with actual model id
    max_tokens: int = 4096
    temperature: float = 0.1
    site_url: str | None = None
    site_title: str | None = None


def _coerce_section(raw: Dict[str, Any]) -> DatalabSettings:
    missing = [key for key in ("api_key",) if not raw.get(key)]
    if missing:
        raise ValueError(
            "Missing required config value(s): "
            + ", ".join(missing)
            + ". Did you copy config.example.toml to config.toml?"
        )
    return DatalabSettings(
        api_key=raw["api_key"],
        endpoint=raw.get("endpoint", DatalabSettings.endpoint),
        ocr_endpoint=raw.get("ocr_endpoint", DatalabSettings.ocr_endpoint),
        poll_interval_seconds=float(
            raw.get("poll_interval_seconds", DatalabSettings.poll_interval_seconds)
        ),
        poll_timeout_seconds=float(
            raw.get("poll_timeout_seconds", DatalabSettings.poll_timeout_seconds)
        ),
        request_timeout_seconds=float(
            raw.get("request_timeout_seconds", DatalabSettings.request_timeout_seconds)
        ),
        fallback_endpoints=tuple(
            str(url).strip()
            for url in raw.get("fallback_endpoints", [])
            if str(url).strip()
        ),
        ocr_fallback_endpoints=tuple(
            str(url).strip()
            for url in raw.get("ocr_fallback_endpoints", [])
            if str(url).strip()
        ),
        multi_run_enabled=bool(raw.get("multi_run_enabled", False)),
        multi_run_runs=int(raw.get("multi_run_runs", DatalabSettings.multi_run_runs)),
        multi_run_require_all_success=bool(raw.get("multi_run_require_all_success", DatalabSettings.multi_run_require_all_success)),
    )


@lru_cache(maxsize=1)
def get_settings(config_path: Path | str | None = None) -> DatalabSettings:
    """
    Load datalab-related settings from a TOML file.

    The values are cached, so repeated calls are cheap.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Unable to find config file at {path}. "
            "Copy config.example.toml to config.toml and set your API key."
        )
    content = tomllib.loads(path.read_text(encoding="utf-8"))
    if "datalab" not in content:
        raise KeyError("Config file must contain a [datalab] section.")
    return _coerce_section(content["datalab"])


def _coerce_deepinfra_section(raw: Dict[str, Any]) -> DeepInfraSettings:
    missing = [key for key in ("api_token",) if not raw.get(key)]
    if missing:
        raise ValueError(
            "Missing required config value(s) for [deepinfra]: "
            + ", ".join(missing)
            + ". Set your DeepInfra API token in config.toml."
        )
    return DeepInfraSettings(
        api_token=str(raw["api_token"]),
        base_url=str(raw.get("base_url", DeepInfraSettings.base_url)),
        model=str(raw.get("model", DeepInfraSettings.model)),
        max_tokens=int(raw.get("max_tokens", DeepInfraSettings.max_tokens)),
        temperature=float(raw.get("temperature", DeepInfraSettings.temperature)),
    )


@lru_cache(maxsize=1)
def get_deepinfra_settings(config_path: Path | str | None = None) -> DeepInfraSettings:
    """Load DeepInfra settings from config.toml ([deepinfra] section)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Unable to find config file at {path}. Copy config.example.toml to config.toml and set your DEEPINFRA token."
        )
    content = tomllib.loads(path.read_text(encoding="utf-8"))
    if "deepinfra" not in content:
        raise KeyError("Config file must contain a [deepinfra] section for DeepSeek OCR.")
    return _coerce_deepinfra_section(content["deepinfra"])


def _coerce_openrouter_section(raw: Dict[str, Any]) -> OpenRouterSettings:
    missing = [key for key in ("api_key",) if not raw.get(key)]
    if missing:
        raise ValueError(
            "Missing required config value(s) for [openrouter]: "
            + ", ".join(missing)
            + ". Set your OpenRouter API key in config.toml or OPENROUTER_API_KEY env var."
        )
    return OpenRouterSettings(
        api_key=str(raw["api_key"]),
        base_url=str(raw.get("base_url", OpenRouterSettings.base_url)),
        model=str(raw.get("model", OpenRouterSettings.model)),
        max_tokens=int(raw.get("max_tokens", OpenRouterSettings.max_tokens)),
        temperature=float(raw.get("temperature", OpenRouterSettings.temperature)),
        site_url=str(raw.get("site_url")) if raw.get("site_url") else None,
        site_title=str(raw.get("site_title")) if raw.get("site_title") else None,
    )


@lru_cache(maxsize=1)
def get_openrouter_settings(config_path: Path | str | None = None) -> OpenRouterSettings:
    """Load OpenRouter settings from config.toml ([openrouter] section).

    Note: You may also set env vars OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL.
    """
    # Env override support
    import os
    api_key_env = os.environ.get("OPENROUTER_API_KEY")
    if api_key_env:
        return OpenRouterSettings(
            api_key=str(api_key_env),
            base_url=str(os.environ.get("OPENROUTER_BASE_URL", OpenRouterSettings.base_url)),
            model=str(os.environ.get("OPENROUTER_MODEL", OpenRouterSettings.model)),
            site_url=os.environ.get("OPENROUTER_SITE_URL"),
            site_title=os.environ.get("OPENROUTER_SITE_TITLE"),
        )

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Unable to find config file at {path}. Copy config.example.toml to config.toml and set your OPENROUTER key."
        )
    content = tomllib.loads(path.read_text(encoding="utf-8"))
    if "openrouter" not in content:
        raise KeyError("Config file must contain an [openrouter] section for Qwen3 VL.")
    return _coerce_openrouter_section(content["openrouter"])

