"""JIT-accelerated math kernels for Infernux hot paths.

Uses Numba ``@njit(cache=True)`` when available, with pure-Python fallbacks.
Call :func:`precompile` once at engine startup to hide JIT latency.

Example::

    from Infernux._jit_kernels import jit_smooth_damp, precompile
    precompile()                         # warm Numba cache once
    val, vel = jit_smooth_damp(0, 10, 0, 0.3, float('inf'), 0.016)
"""

from __future__ import annotations

from typing import Any

JIT_AVAILABLE: bool

def njit(*args: Any, **kwargs: Any) -> Any:
    """Numba ``njit`` decorator, or a no-op fallback when Numba is unavailable."""
    ...

def jit_smooth_damp(
    current: float,
    target: float,
    current_velocity: float,
    smooth_time: float,
    max_speed: float,
    delta_time: float,
) -> tuple[float, float]:
    """Critically-damped spring interpolation.

    Args:
        current: Current value.
        target: Desired value.
        current_velocity: Velocity from the previous frame (mutated output).
        smooth_time: Approximate time to reach the target (seconds, ≥ 0.0001).
        max_speed: Maximum speed clamp.
        delta_time: Frame delta time.

    Returns:
        ``(new_value, new_velocity)`` tuple.
    """
    ...

def jit_contains_point_rotated(
    px: float,
    py: float,
    rx: float,
    ry: float,
    rw: float,
    rh: float,
    sin_a: float,
    cos_a: float,
) -> bool:
    """Test if point *(px, py)* lies inside a rotated rectangle.

    The rectangle is defined by its top-left corner *(rx, ry)*, size *(rw, rh)*,
    and a rotation described by ``sin_a`` / ``cos_a``.

    Args:
        px: Point X coordinate.
        py: Point Y coordinate.
        rx: Rectangle origin X.
        ry: Rectangle origin Y.
        rw: Rectangle width.
        rh: Rectangle height.
        sin_a: Sine of the rotation angle.
        cos_a: Cosine of the rotation angle.
    """
    ...

def jit_wire_sphere_trig(segments: int) -> tuple[list[float], list[float]]:
    """Return ``(cos_table, sin_table)`` arrays for a unit circle.

    Each list has *segments* entries: cos/sin of ``2π·i / segments``.

    Args:
        segments: Number of subdivisions around the circle.
    """
    ...

def precompile() -> None:
    """Force Numba to compile and cache all kernels to disk.

    Call once at engine startup. Does nothing when Numba is not installed.
    """
    ...

__all__: list[str]
