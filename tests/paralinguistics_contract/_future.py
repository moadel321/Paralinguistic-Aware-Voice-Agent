from __future__ import annotations

import importlib
from typing import Any

import pytest


def require_attr(module_name: str, attr_name: str) -> Any:
    """Load a future implementation symbol and fail with a clear TDD message."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"Expected implementation module {module_name!r} to exist: {exc}",
            pytrace=False,
        )

    try:
        return getattr(module, attr_name)
    except AttributeError:
        pytest.fail(
            f"Expected {module_name!r} to expose {attr_name!r}.",
            pytrace=False,
        )


def assert_no_transcript(signal: object) -> None:
    assert not hasattr(signal, "text")
    assert not hasattr(signal, "transcript")
    assert not hasattr(signal, "clean_text")
