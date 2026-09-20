"""Provider registry."""

from __future__ import annotations

from cachelint.providers.anthropic import AnthropicProvider
from cachelint.providers.base import Provider, canonical
from cachelint.providers.openai import OpenAIProvider

_REGISTRY: dict[str, Provider] = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
}


def get_provider(name: str) -> Provider:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; known: {sorted(_REGISTRY)}") from None


def provider_names() -> list[str]:
    return sorted(_REGISTRY)


def detect_provider(url: str) -> Provider | None:
    for provider in _REGISTRY.values():
        if provider.matches(url):
            return provider
    return None


def register_provider(provider: Provider) -> None:
    """Add or replace a provider adapter (for custom gateways or new APIs)."""
    _REGISTRY[provider.name] = provider


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "Provider",
    "canonical",
    "detect_provider",
    "get_provider",
    "provider_names",
    "register_provider",
]
