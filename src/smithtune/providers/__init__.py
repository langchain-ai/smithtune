"""Resolve training adapters without loading optional provider SDKs."""

from importlib import import_module

from smithtune.providers.base import PipelineError, TrainingProvider


PROVIDERS = {
    "fireworks": ("smithtune.providers.fireworks", "FireworksProvider"),
    "baseten": ("smithtune.providers.baseten", "BasetenProvider"),
}


def get_provider(name: str) -> TrainingProvider:
    try:
        module_name, class_name = PROVIDERS[name]
    except KeyError as exc:
        raise PipelineError(f"unknown training provider: {name}") from exc
    return getattr(import_module(module_name), class_name)()
