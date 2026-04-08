"""
Internationalization (i18n) for the Infernux editor.

Provides a simple key-based translation system with two supported locales:
``"en"`` (English) and ``"zh"`` (Simplified Chinese).

Translation strings are stored in external JSON files under ``locales/``:

- ``locales/en.json``
- ``locales/zh.json``

Usage::

    from Infernux.engine.i18n import t

    label = t("menu.project")        # "Project" or "项目"
    label = t("menu.preferences")    # "Preferences" or "偏好设置"

The active locale is persisted to ``Documents/Infernux/preferences.json``
so it survives across sessions.
"""

from __future__ import annotations

import json
import os
import pathlib
from Infernux.debug import Debug

# ---------------------------------------------------------------------------
# Locale state
# ---------------------------------------------------------------------------

_current_locale: str = "zh"

# ---------------------------------------------------------------------------
# Translation tables — loaded from locales/*.json at module init
# ---------------------------------------------------------------------------

_LOCALES_DIR = os.path.join(os.path.dirname(__file__), "locales")

_tables: dict[str, dict[str, str]] = {}


def _load_locale_table(locale: str) -> dict[str, str]:
    """Load and cache a single locale JSON file."""
    if locale in _tables:
        return _tables[locale]
    path = os.path.join(_LOCALES_DIR, f"{locale}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        data = {}
    _tables[locale] = data
    return data


def _load_all_locales() -> None:
    """Pre-load all discovered locale files."""
    if not os.path.isdir(_LOCALES_DIR):
        return
    for name in os.listdir(_LOCALES_DIR):
        if name.endswith(".json"):
            _load_locale_table(name[:-5])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def t(key: str) -> str:
    """Return the translated string for *key* in the current locale.

    Falls back to English, then returns the key itself if not found.
    """
    table = _tables.get(_current_locale)
    if table:
        val = table.get(key)
        if val is not None:
            return val
    en_table = _tables.get("en")
    if en_table:
        val = en_table.get(key)
        if val is not None:
            return val
    return key


def get_locale() -> str:
    """Return the current locale code (``"en"`` or ``"zh"``)."""
    return _current_locale


def set_locale(locale: str) -> None:
    """Set the active locale and persist to disk."""
    global _current_locale
    if locale not in _tables:
        return
    _current_locale = locale
    _save_preference()


# ---------------------------------------------------------------------------
# Persistence — Documents/Infernux/preferences.json
# ---------------------------------------------------------------------------

_PREFS_FILE = "preferences.json"


def _prefs_path() -> str:
    """Return the path to the global preferences file."""
    if os.name == "nt":
        docs = pathlib.Path.home() / "Documents"
        try:
            import ctypes.wintypes
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
            if buf.value:
                docs = pathlib.Path(buf.value)
        except (OSError, ValueError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    else:
        docs = pathlib.Path.home() / "Documents"
    prefs_dir = docs / "Infernux"
    os.makedirs(prefs_dir, exist_ok=True)
    return str(prefs_dir / _PREFS_FILE)


def _load_preference() -> None:
    """Load the locale from the preferences file."""
    global _current_locale
    path = _prefs_path()
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        locale = data.get("language", "zh")
        if locale in _tables:
            _current_locale = locale
    except (json.JSONDecodeError, OSError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


def _save_preference() -> None:
    """Save the current locale to the preferences file."""
    path = _prefs_path()
    data: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data["language"] = _current_locale
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


# ---------------------------------------------------------------------------
# Module init — load locale files, then restore persisted preference
# ---------------------------------------------------------------------------

_load_all_locales()
_load_preference()
