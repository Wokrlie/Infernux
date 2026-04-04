"""Shared Python helpers for the native C++ Inspector panel.

The inspector UI itself now lives in C++. This module keeps the small set of
Python-side utilities still needed by bootstrap wiring and undo/cache
invalidation.
"""

from __future__ import annotations

import os

from .theme import Theme


def _sample_rgba_bilinear(raw: bytes, src_w: int, src_h: int,
                          x: float, y: float) -> tuple[float, float, float, float]:
    """Return a bilinearly filtered RGBA sample from packed byte pixels."""
    if src_w <= 0 or src_h <= 0:
        return 0.0, 0.0, 0.0, 0.0

    x = min(max(x, 0.0), float(src_w - 1))
    y = min(max(y, 0.0), float(src_h - 1))

    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, src_w - 1)
    y1 = min(y0 + 1, src_h - 1)
    fx = x - x0
    fy = y - y0

    def _px(ix: int, iy: int) -> tuple[float, float, float, float]:
        idx = (iy * src_w + ix) * 4
        return (
            float(raw[idx]),
            float(raw[idx + 1]),
            float(raw[idx + 2]),
            float(raw[idx + 3]),
        )

    c00 = _px(x0, y0)
    c10 = _px(x1, y0)
    c01 = _px(x0, y1)
    c11 = _px(x1, y1)

    out = []
    for i in range(4):
        top = c00[i] + (c10[i] - c00[i]) * fx
        bottom = c01[i] + (c11[i] - c01[i]) * fx
        out.append(top + (bottom - top) * fy)
    return out[0], out[1], out[2], out[3]


def _resample_rgba_supersampled(raw: bytes, src_w: int, src_h: int,
                                dst_w: int, dst_h: int,
                                taps: int = 4) -> list[int]:
    """Downsample RGBA pixels with supersampled bilinear filtering."""
    dst_w = max(1, int(dst_w))
    dst_h = max(1, int(dst_h))
    taps = max(1, int(taps))
    total_samples = float(taps * taps)
    out: list[int] = []

    for dy in range(dst_h):
        for dx in range(dst_w):
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            acc_a = 0.0
            for sy in range(taps):
                sample_y = ((dy + (sy + 0.5) / taps) * src_h / dst_h) - 0.5
                for sx in range(taps):
                    sample_x = ((dx + (sx + 0.5) / taps) * src_w / dst_w) - 0.5
                    r, g, b, a = _sample_rgba_bilinear(raw, src_w, src_h, sample_x, sample_y)
                    acc_r += r
                    acc_g += g
                    acc_b += b
                    acc_a += a

            out.append(int(round(acc_r / total_samples)))
            out.append(int(round(acc_g / total_samples)))
            out.append(int(round(acc_b / total_samples)))
            out.append(int(round(acc_a / total_samples)))

    return out


def prepare_component_icon_pixels(tex_data) -> tuple[list[int], int, int]:
    """Bake a centered, inspector-sized component icon texture."""
    canvas_px = max(1, int(Theme.COMPONENT_ICON_SIZE))
    inner_pad = 1 if canvas_px > 4 else 0
    max_content_px = max(1, canvas_px - inner_pad * 2)
    src_w, src_h = int(tex_data.width), int(tex_data.height)

    if src_w <= 0 or src_h <= 0:
        return [], 0, 0

    scale = min(max_content_px / float(src_w), max_content_px / float(src_h), 1.0)
    dst_w = max(1, min(max_content_px, int(round(src_w * scale))))
    dst_h = max(1, min(max_content_px, int(round(src_h * scale))))

    raw = tex_data.get_pixels()
    if dst_w != src_w or dst_h != src_h:
        resized = _resample_rgba_supersampled(raw, src_w, src_h, dst_w, dst_h)
    else:
        resized = tex_data.get_pixels_list()

    canvas = [0] * (canvas_px * canvas_px * 4)
    offset_x = (canvas_px - dst_w) // 2
    offset_y = (canvas_px - dst_h) // 2
    row_stride = dst_w * 4

    for y in range(dst_h):
        src_off = y * row_stride
        dst_off = ((offset_y + y) * canvas_px + offset_x) * 4
        canvas[dst_off:dst_off + row_stride] = resized[src_off:src_off + row_stride]

    return canvas, canvas_px, canvas_px


_component_structure_version = 0
_component_tracker_reset_version = 0


def bump_component_structure_version() -> None:
    """Increment structure/reset versions so the native Inspector drops stale refs."""
    global _component_structure_version, _component_tracker_reset_version
    _component_structure_version += 1
    _component_tracker_reset_version += 1


def get_component_structure_version() -> int:
    """Return the current component structure version for cache invalidation."""
    return _component_structure_version


def ensure_material_file_path(material) -> str:
    """Ensure *material* has a stable ``file_path`` for autosave and undo."""
    if getattr(material, 'file_path', ''):
        return material.file_path
    guid = getattr(material, 'guid', '') or ''
    if guid:
        try:
            from Infernux.lib import AssetRegistry
            adb = AssetRegistry.instance().get_asset_database()
            if adb:
                resolved = adb.get_path_from_guid(guid)
                if resolved:
                    material.file_path = resolved
                    return resolved
        except (ImportError, RuntimeError, AttributeError):
            pass
    from Infernux.engine.project_context import get_project_root
    project_root = get_project_root()
    if not project_root:
        return ""
    materials_dir = os.path.join(project_root, "materials")
    os.makedirs(materials_dir, exist_ok=True)
    mat_name = getattr(material, 'name', 'DefaultUnlit')
    if mat_name == "DefaultLit":
        mat_file = os.path.join(materials_dir, "default_lit.mat")
    elif mat_name == "DefaultUnlit":
        mat_file = os.path.join(materials_dir, "default_unlit.mat")
    else:
        import re as _re
        file_name = _re.sub(r'([A-Z])', r'_\1', mat_name).lower().strip('_') + ".mat"
        mat_file = os.path.join(materials_dir, file_name)
    material.file_path = mat_file
    return mat_file


__all__ = [
    "bump_component_structure_version",
    "ensure_material_file_path",
    "get_component_structure_version",
    "prepare_component_icon_pixels",
]