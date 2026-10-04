"""Execute the usage examples embedded in the domain modules as real tests.

This guarantees every ``>>>`` example in the core stays correct without enabling global
``--doctest-modules`` (which would try to import optional, heavy adapter dependencies).
"""

from __future__ import annotations

import doctest
import importlib

import pytest

_DOMAIN_MODULES = [
    "rag.domain.models",
    "rag.domain.cleaning",
    "rag.domain.chunking",
    "rag.domain.prompt",
    "rag.domain.pipeline",
    "rag.domain.exceptions",
]


@pytest.mark.parametrize("module_name", _DOMAIN_MODULES)
def test_module_doctests(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False)
    assert result.failed == 0, f"{module_name}: {result.failed} doctest failure(s)"
