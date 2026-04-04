from __future__ import annotations

from typing import Any

JIT_AVAILABLE: bool

def ensure_jit_runtime(*, auto_install: bool = True) -> bool:
    """Ensure the current runtime can import Numba-backed JIT helpers."""
    ...

def njit(*args: Any, **kwargs: Any) -> Any:
    """Numba ``njit`` decorator, or a no-op fallback when Numba is unavailable."""
    ...

def precompile() -> None:
    """Compile and cache built-in JIT kernels ahead of time."""
    ...

def precompile_jit() -> None:
    """Compile and cache built-in JIT kernels ahead of time."""
    ...

__all__: list[str]