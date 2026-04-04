"""
JIT-accelerated math kernels for Infernux hot paths.

Uses Numba ``@njit(cache=True)`` when available, with pure-Python fallbacks.
The ``precompile()`` function triggers ahead-of-time compilation so the
engine startup pays the cost once (subsequent imports read the on-disk cache).

Usage::

    from Infernux._jit_kernels import (
        jit_smooth_damp,
        jit_contains_point_rotated,
        jit_wire_sphere_verts,
        precompile,
    )
"""

from __future__ import annotations

import math

_HAS_NUMBA = False
try:
    from numba import njit  # type: ignore[import-untyped]
    _HAS_NUMBA = True
except ImportError:
    # Provide a no-op decorator fallback
    def njit(*args, **kwargs):  # noqa: D103
        def _wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrap


JIT_AVAILABLE = _HAS_NUMBA


# ====================================================================
# Kernel: smooth_damp (critically-damped spring)
# ====================================================================

@njit(cache=True)
def jit_smooth_damp(current: float, target: float,
                    current_velocity: float, smooth_time: float,
                    max_speed: float, delta_time: float):
    """Critically-damped spring — returns (new_value, new_velocity)."""
    if smooth_time < 0.0001:
        smooth_time = 0.0001
    omega = 2.0 / smooth_time
    x = omega * delta_time
    exp_factor = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    original_to = target

    max_change = max_speed * smooth_time
    if change < -max_change:
        change = -max_change
    elif change > max_change:
        change = max_change
    target = current - change

    temp = (current_velocity + omega * change) * delta_time
    new_velocity = (current_velocity - omega * temp) * exp_factor
    output = target + (change + temp) * exp_factor

    # Prevent overshoot
    if (original_to - current > 0.0) == (output > original_to):
        output = original_to
        if delta_time > 0.0:
            new_velocity = (output - original_to) / delta_time
        else:
            new_velocity = 0.0

    return output, new_velocity


# ====================================================================
# Kernel: rotated point containment test
# ====================================================================

@njit(cache=True)
def jit_contains_point_rotated(px: float, py: float,
                                rx: float, ry: float,
                                rw: float, rh: float,
                                sin_a: float, cos_a: float) -> bool:
    """Test if (px, py) lies inside a rotated rect.

    ``sin_a``, ``cos_a`` are from the element's rotation angle.
    Inverse-rotates the point into local space for AABB check.
    """
    dx = px - (rx + rw * 0.5)
    dy = py - (ry + rh * 0.5)
    # Inverse rotation (negate sin)
    lx = dx * cos_a + dy * sin_a + rw * 0.5
    ly = -dx * sin_a + dy * cos_a + rh * 0.5
    return (0.0 <= lx <= rw) and (0.0 <= ly <= rh)


# ====================================================================
# Kernel: wire sphere trig table
# ====================================================================

@njit(cache=True)
def jit_wire_sphere_trig(segments: int):
    """Return (cos_table, sin_table) arrays for a unit circle.

    Each has *segments* entries: cos/sin of ``2π * i / segments``.
    """
    two_pi = 2.0 * math.pi
    cos_tab = [0.0] * segments
    sin_tab = [0.0] * segments
    for i in range(segments):
        angle = two_pi * i / segments
        cos_tab[i] = math.cos(angle)
        sin_tab[i] = math.sin(angle)
    return cos_tab, sin_tab


# ====================================================================
# Pre-compilation trigger
# ====================================================================

def precompile():
    """Force Numba to compile all kernels (and cache to disk).

    Call once at engine startup to hide JIT latency.
    Does nothing if Numba is not installed.
    """
    if not _HAS_NUMBA:
        return

    # Trigger compilation with representative argument types
    jit_smooth_damp(0.0, 1.0, 0.0, 0.3, float('inf'), 0.016)
    jit_contains_point_rotated(50.0, 50.0, 0.0, 0.0, 100.0, 100.0, 0.0, 1.0)
    jit_wire_sphere_trig(24)


__all__ = [
    "njit",
    "JIT_AVAILABLE",
    "jit_smooth_damp",
    "jit_contains_point_rotated",
    "jit_wire_sphere_trig",
    "precompile",
]
