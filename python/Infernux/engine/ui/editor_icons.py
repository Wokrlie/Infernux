"""Centralized editor icon texture loader.

Lazily uploads PNG icons from ``resources/icons/`` to GPU and caches
their ImGui texture IDs.  All panels share a single cache.

Usage::

    from .editor_icons import EditorIcons
    tid = EditorIcons.get(native_engine, "plus")   # -> int texture id
"""

import os
from Infernux.lib import TextureLoader
import Infernux.resources as _resources

_cache: dict[str, int] = {}
_loaded: bool = False


def _ensure_loaded(native_engine) -> None:
    """Upload all known editor icons (once)."""
    global _loaded
    if _loaded or native_engine is None:
        return

    _ICONS = [
        "plus", "minus", "remove", "picker",
        "warning", "error",
        "ui_text", "ui_image", "ui_button",
        "tool_none", "tool_move", "tool_rotate", "tool_scale",
    ]
    for name in _ICONS:
        tex_name = f"__edicon__{name}"
        if native_engine.has_imgui_texture(tex_name):
            _cache[name] = native_engine.get_imgui_texture_id(tex_name)
            continue
        path = os.path.join(_resources.file_type_icons_dir, f"{name}.png")
        if not os.path.isfile(path):
            continue
        td = TextureLoader.load_from_file(path)
        if td and td.is_valid():
            tid = native_engine.upload_texture_for_imgui(
                tex_name, td.get_pixels_list(), td.width, td.height)
            if tid != 0:
                _cache[name] = tid
    _loaded = True


class EditorIcons:
    """Thin façade around the module-level icon cache."""

    @staticmethod
    def get(native_engine, name: str) -> int:
        """Return ImGui texture id for *name*, or 0 if unavailable."""
        _ensure_loaded(native_engine)
        return _cache.get(name, 0)

    @staticmethod
    def get_cached(name: str) -> int:
        """Return a previously loaded icon id, or 0.  No engine required."""
        return _cache.get(name, 0)

    @staticmethod
    def reset():
        """Clear the cache (e.g. after engine re-init)."""
        global _loaded
        _cache.clear()
        _loaded = False
