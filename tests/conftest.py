"""Process-wide numerical backend policy for deterministic GTM tests."""

from __future__ import annotations

import os

# These variables must exist before pytest imports any module that imports JAX.
# Unit and trajectory regressions audit CPU float64 semantics; selecting a GPU
# would make correctness depend on desktop allocator state and available VRAM.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")
