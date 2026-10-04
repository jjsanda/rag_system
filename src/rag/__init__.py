"""rag: a provider-agnostic Retrieval-Augmented Generation service.

The package is organised as a hexagonal (ports-and-adapters) architecture:

* :mod:`rag.domain` holds the framework-free core (models, ports, chunking,
  prompt assembly and the ingestion/query pipeline).
* :mod:`rag.adapters` holds interchangeable implementations of the domain ports
  (embedding providers, LLMs, vector stores, loaders and caches).
* :mod:`rag.config` exposes the 12-factor settings.
* :mod:`rag.observability` provides structured logging, metrics and tracing.
* :mod:`rag.api` is the FastAPI driving adapter; :mod:`rag.factory` wires
  everything together from configuration (the composition root).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
