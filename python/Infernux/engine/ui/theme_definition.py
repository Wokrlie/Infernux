"""Data classes for the theme system.

Plugins pass a ``ThemeDefinition`` to ``Theme.register()``. The engine
resolves it into an immutable ``ThemeSnapshot`` at apply time and pushes it to
C++ via the editor theme registry.

The token names plugins use (``BG_BASE``, ``ACCENT``, ``STATE_ERROR`` …) are
the canonical engine token set. See ``InfernuxThemes/Default/package/plugin_pages/tokens-mapping.md``
for the mapping from the legacy ``Theme.X`` class attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


# ─────────────────────────────────────────────────────────────────────────────
#  Primitive value types
# ─────────────────────────────────────────────────────────────────────────────

def _srgb_to_linear(s: float) -> float:
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4


@dataclass(frozen=True, slots=True)
class RGBA:
    """Linear-space RGBA in [0, 1]. Convert at the boundary via ``from_srgb_hex``."""

    r: float
    g: float
    b: float
    a: float = 1.0

    def __post_init__(self) -> None:
        for name in ("r", "g", "b", "a"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"RGBA.{name} = {v} not in [0, 1]")

    @classmethod
    def from_srgb_hex(cls, hex_str: str, alpha: float = 1.0) -> "RGBA":
        """Build from a 6-char sRGB hex string (``"#RRGGBB"``)."""
        s = hex_str.lstrip("#")
        if len(s) != 6:
            raise ValueError(f"hex must be 6 chars, got {hex_str!r}")
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return cls(
            r=_srgb_to_linear(r),
            g=_srgb_to_linear(g),
            b=_srgb_to_linear(b),
            a=alpha,
        )

    def to_srgb_hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(
            round(self.r * 255),
            round(self.g * 255),
            round(self.b * 255),
        )


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LogicalSize:
    """DPI-independent size. Engine multiplies by DPI at apply time."""

    value: float


@dataclass(frozen=True, slots=True)
class IconRef:
    """Phase 2+. Asset GUID reference. Never a filesystem path."""

    guid: str
    fallback_guid: str | None = None
    tint: RGBA | None = None


# ─────────────────────────────────────────────────────────────────────────────
#  Theme kind
# ─────────────────────────────────────────────────────────────────────────────

class ThemeKind(str, Enum):
    """Closed enum. Add new variants, do not rename existing ones."""

    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"


# ─────────────────────────────────────────────────────────────────────────────
#  Namespaced extension (Phase 3 — plugin-to-plugin contracts)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThemeExtension:
    """Plugin-owned namespace of additional tokens. Read via ``Theme.token(ns, key)``."""

    namespace: str                                # must be ≥ 2 segments, snake_case.lower
    tokens: Mapping[str, RGBA | float | Vec2 | LogicalSize | IconRef]


# ─────────────────────────────────────────────────────────────────────────────
#  ThemeDefinition — what plugins pass to register()
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThemeDefinition:
    """A theme plugin's contribution. Partial by design — missing keys are
    filled from ``based_on`` and finally from the engine defaults."""

    schema_version: int = 1
    id: str = ""                                   # globally unique, "<author>/<name>"
    display_name: str = ""
    kind: ThemeKind = ThemeKind.DARK

    # Token overrides (partial)
    colors: Mapping[str, RGBA] = field(default_factory=dict)
    imgui_style: Mapping[str, float | Vec2] = field(default_factory=dict)
    sizes: Mapping[str, float] = field(default_factory=dict)

    # Phase 2+
    icons: Mapping[str, IconRef] = field(default_factory=dict)

    # Namespaced extensions
    extends: Sequence[ThemeExtension] = ()

    # Metadata
    author: str = ""
    description: str = ""
    based_on: str | None = None                    # single-level inheritance


# ─────────────────────────────────────────────────────────────────────────────
#  ThemeSnapshot — resolved, immutable, fully populated
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThemeSnapshot:
    """Result of resolving a ThemeDefinition. Every token has a value.
    This is what gets pushed to C++ on each apply."""

    id: str
    display_name: str
    kind: ThemeKind
    owner: str                                    # who registered this

    colors: Mapping[str, RGBA]
    imgui_style: Mapping[str, float | Vec2]
    sizes: Mapping[str, float]
    icons: Mapping[str, IconRef]
    extensions: Sequence[ThemeExtension]


# ─────────────────────────────────────────────────────────────────────────────
#  ThemeChanged event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThemeChanged:
    """Fired when a new snapshot becomes active. ``old`` may be None on first
    apply."""

    old: ThemeSnapshot | None
    new: ThemeSnapshot | None
