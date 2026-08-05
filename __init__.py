"""TELOS — Prompt Reuse via Indexed Span Marking.

Python reference implementation of the three-tier cache-friendly prompt protocol.

Public entry points:
    from telos import (
        # IR
        Band, TelosBlock, TelosMessage, TelosIR,
        # Bridge
        Bridge,
        # Factories
        load_harness, load_engine,
    )

See telos/README.zh.md for details.
"""

from telos.ir import (
    Band,
    CacheBoundary,
    CacheIntent,
    LogicalPosition,
    TelosBlock,
    TelosHints,
    TelosIR,
    TelosInvariantError,
    TelosMessage,
    UsageReport,
)
from telos.bridge import Bridge
from telos.engine.base import (
    BidirectionalEngineAdapter,
    CacheCapabilities,
    EngineAdapter,
    EngineCapabilities,
    UnsupportedWireProtocolError,
    ProbeResult,
)
from telos.registry import load_engine, load_harness
from telos.scripts.transport import TelosTransport, HarnessPreset, PRESETS

__all__ = [
    "Band",
    "LogicalPosition",
    "CacheBoundary",
    "CacheIntent",
    "TelosBlock",
    "TelosHints",
    "TelosMessage",
    "TelosIR",
    "TelosInvariantError",
    "UsageReport",
    "Bridge",
    "EngineAdapter",
    "EngineCapabilities",
    "CacheCapabilities",
    "UnsupportedWireProtocolError",
    "BidirectionalEngineAdapter",
    "ProbeResult",
    "load_harness",
    "load_engine",
    "TelosTransport",
    "HarnessPreset",
    "PRESETS",
]
