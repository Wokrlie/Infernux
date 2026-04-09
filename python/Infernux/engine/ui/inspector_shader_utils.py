"""
Shader file parsing and lookup utilities for the Inspector panel.

All functions are stateless except for an optional *cache* dict parameter
that callers can provide for per-session caching.
"""

import os

# Global generation counter, bumped on every successful shader hot-reload.
# Inspector sync keys include this so that property lists refresh automatically.
_shader_property_generation: int = 0
_shader_catalog_cache: dict[str, dict[str, object]] = {}
_shader_properties_cache: dict[tuple[str, str], list] = {}


def bump_shader_property_generation():
    """Increment the property generation counter (called after shader hot-reload)."""
    global _shader_property_generation
    _shader_property_generation += 1
    _shader_catalog_cache.clear()
    _shader_properties_cache.clear()


def get_shader_property_generation() -> int:
    """Return the current property generation counter."""
    return _shader_property_generation


def _get_shader_search_roots() -> list[str]:
    """Return the project and built-in shader roots."""
    from Infernux.engine.project_context import get_project_root

    project_root = get_project_root()
    search_roots = []
    if project_root:
        search_roots.append(os.path.join(project_root, "Assets"))

    from Infernux.resources import resources_path
    builtin_root = os.path.join(resources_path, "shaders")
    search_roots.append(builtin_root)
    return search_roots


def _scan_shader_catalog(ext: str) -> dict[str, object]:
    """Scan shader roots once and cache both candidates and id->path mapping."""
    items = []
    seen_shader_ids = set()
    shader_paths = {}

    for root in _get_shader_search_roots():
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not fname.lower().endswith(ext):
                    continue
                full_path = os.path.join(dirpath, fname)

                shader_id = parse_shader_id(full_path)
                if not shader_id:
                    continue

                shader_paths.setdefault(shader_id, full_path)
                if shader_id in seen_shader_ids:
                    continue
                seen_shader_ids.add(shader_id)

                if is_shader_hidden(full_path):
                    continue

                items.append((shader_id, shader_id))

    if not items:
        items = [("(No shaders found)", "")]

    return {
        "items": items,
        "paths": shader_paths,
    }


def _get_shader_catalog(ext: str) -> dict[str, object]:
    """Return cached shader catalog for the requested extension."""
    catalog = _shader_catalog_cache.get(ext)
    if catalog is None:
        catalog = _scan_shader_catalog(ext)
        _shader_catalog_cache[ext] = catalog
    return catalog


def _get_shader_properties_cached(shader_id: str, ext: str) -> list:
    """Return cached @property metadata for a shader id."""
    if not shader_id:
        return []

    cache_key = (shader_id, ext)
    cached = _shader_properties_cache.get(cache_key)
    if cached is not None:
        return cached

    shader_path = get_shader_file_path(shader_id, ext)
    if not shader_path:
        return []

    props = parse_shader_properties(shader_path)
    _shader_properties_cache[cache_key] = props
    return props


def parse_shader_id(filepath: str) -> str:
    """Parse @shader_id annotation from shader file (new @ format only)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 20:
                break
            line = line.strip()
            if line.startswith('@shader_id:'):
                return line[11:].strip()
    return None


def parse_shader_properties(filepath: str) -> list:
    """Parse @property annotations from shader file.
    Returns list of dicts: [{'name': str, 'type': str, 'default': any, 'hdr': bool}, ...]

    Format: ``@property: name, Type, default[, HDR]``
    """
    import json
    properties = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 50:
                break
            line = line.strip()
            if line.startswith('@property:'):
                prop_str = line[10:].strip()
                parts = prop_str.split(',', 2)
                if len(parts) >= 3:
                    name = parts[0].strip()
                    prop_type = parts[1].strip()
                    rest = parts[2].strip()
                    # rest = "default[, HDR]"
                    # For array defaults like [1.0, 0.0, 0.0, 1.0], find the
                    # closing ']' first, then check for trailing flags.
                    hdr = False
                    if prop_type == 'Texture2D':
                        # e.g. "white" or "white, HDR" (unlikely but safe)
                        tail_parts = rest.rsplit(',', 1)
                        default_val = tail_parts[0].strip()
                        if len(tail_parts) > 1 and tail_parts[1].strip().upper() == 'HDR':
                            hdr = True
                    elif rest.startswith('['):
                        bracket_end = rest.index(']') + 1
                        default_val = json.loads(rest[:bracket_end])
                        trailer = rest[bracket_end:].strip()
                        if trailer.startswith(','):
                            trailer = trailer[1:].strip()
                        if trailer.upper() == 'HDR':
                            hdr = True
                    else:
                        # Scalar: "0.5" or "0.5, HDR"
                        tail_parts = rest.split(',', 1)
                        default_val = json.loads(tail_parts[0].strip())
                        if len(tail_parts) > 1 and tail_parts[1].strip().upper() == 'HDR':
                            hdr = True
                    properties.append({
                        'name': name,
                        'type': prop_type,
                        'default': default_val,
                        'hdr': hdr,
                    })
    return properties


def is_shader_hidden(filepath: str) -> bool:
    """Check if shader file has @hidden annotation (internal shader)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 20:
                break
            stripped = line.strip().lstrip('/ ')
            if stripped == '@hidden':
                return True
    return False


def get_shader_file_path(shader_id: str, ext: str) -> str:
    """Find the file path for a given shader_id by scanning project and built-in dirs."""
    if not shader_id:
        return None
    return _get_shader_catalog(ext).get("paths", {}).get(shader_id)


def shader_display_from_value(value: str, items):
    """Map a shader value to its display string for UI."""
    for display, v in items:
        if v == value:
            return display
    return value


def get_shader_candidates(ext: str, cache: dict = None):
    """Collect shader files from project and built-in shader folders.
    Only shaders with @shader_id annotations are listed.
    Each unique shader_id appears only once in the list.
    
    If *cache* is provided and already contains entries for *ext*, the
    cached result is returned immediately.
    """
    if cache is not None and cache.get(ext) is not None:
        return cache[ext]

    items = _get_shader_catalog(ext).get("items", [("(No shaders found)", "")])

    if cache is not None:
        cache[ext] = items
    return items


_SHADER_TYPE_MAP = {
    'Float': 0,
    'Float2': 1,
    'Float3': 2,
    'Float4': 3,
    'Color': 7,
    'Int': 4,
    'Mat4': 5,
    'Texture2D': 6,
}


def _apply_shader_props_to_mat(mat_data: dict, all_props: list[dict],
                                remove_unknown: bool = False):
    """Apply a merged list of shader @property dicts to mat_data.

    Shared implementation for `sync_properties_from_shader` and
    `sync_all_shader_properties`.
    """
    if not all_props:
        return

    props = mat_data.setdefault("properties", {})

    seen_names: set[str] = set()
    ordered_names: list[str] = []
    for sp in all_props:
        name = sp.get('name', '')
        if name and name not in seen_names:
            ordered_names.append(name)
            seen_names.add(name)
    mat_data["_shader_property_order"] = ordered_names

    shader_prop_names: set[str] = set()
    for sp in all_props:
        name = sp.get('name', '')
        ptype_str = sp.get('type', 'Float')
        default = sp.get('default')
        hdr = sp.get('hdr', False)

        if not name:
            continue

        shader_prop_names.add(name)
        ptype = _SHADER_TYPE_MAP.get(ptype_str, 0)

        if name in props:
            props[name]['type'] = ptype
            props[name]['hdr'] = hdr
        else:
            if ptype == 6:
                props[name] = {'type': ptype, 'guid': "", 'hdr': hdr}
            else:
                props[name] = {'type': ptype, 'value': default, 'hdr': hdr}

    if remove_unknown:
        for k in [k for k in props if k not in shader_prop_names]:
            del props[k]


def sync_properties_from_shader(mat_data: dict, shader_id: str, ext: str,
                                remove_unknown: bool = False):
    """Sync material properties from shader's @property annotations.
    Adds new properties from shader, keeps existing values if property exists.
    If *remove_unknown* is True, removes properties not defined in shader.
    """
    shader_props = _get_shader_properties_cached(shader_id, ext)
    if not shader_props:
        # Shader file may be temporarily incomplete during hot-reload.
        # Do NOT clear properties or ordering metadata — preserve existing
        # state so the inspector doesn't flicker.
        return
    _apply_shader_props_to_mat(mat_data, shader_props, remove_unknown=remove_unknown)


def sync_all_shader_properties(mat_data: dict, vert_shader_id: str, frag_shader_id: str,
                               remove_unknown: bool = False):
    """Sync material properties from both vertex and fragment shader annotations.

    Merges @property annotations from both shaders.  Vertex properties appear
    first in the display order, followed by fragment properties.
    If *remove_unknown* is True, removes properties not defined in either shader.
    """
    all_props: list[dict] = []
    if vert_shader_id:
        all_props.extend(_get_shader_properties_cached(vert_shader_id, ".vert"))
    if frag_shader_id:
        all_props.extend(_get_shader_properties_cached(frag_shader_id, ".frag"))
    _apply_shader_props_to_mat(mat_data, all_props, remove_unknown=remove_unknown)


def get_all_shader_property_names(vert_shader_id: str, frag_shader_id: str) -> list[str]:
    """Return all declared material property names from the active vertex and fragment shaders."""
    ordered_names: list[str] = []
    seen_names: set[str] = set()

    for shader_id, ext in ((vert_shader_id, ".vert"), (frag_shader_id, ".frag")):
        if not shader_id:
            continue
        for sp in _get_shader_properties_cached(shader_id, ext):
            name = sp.get("name", "")
            if name and name not in seen_names:
                ordered_names.append(name)
                seen_names.add(name)

    return ordered_names


def get_material_property_display_order(mat_data: dict) -> list[str]:
    """Return material properties in shader declaration order only.

    Properties not declared in the shader (phantom / stale) are excluded.
    """
    props = mat_data.get("properties", {})
    if not props:
        return []

    shader_order = mat_data.get("_shader_property_order", [])
    shader_set = set(shader_order) if shader_order else None

    ordered = []
    seen = set()
    for name in shader_order:
        if name in props and name not in seen:
            ordered.append(name)
            seen.add(name)

    # Only include extras if there is no shader metadata (e.g. unloaded shader)
    if shader_set is None:
        for name in sorted(props.keys()):
            if name not in seen:
                ordered.append(name)

    return ordered
