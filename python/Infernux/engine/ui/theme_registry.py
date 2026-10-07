"""ThemeRegistry — central manager for theme definitions and snapshots.

Single instance per Editor/Player process. Owns:

* the dictionary of ``ThemeDefinition`` (keyed by id, paired with the
  contributing owner)
* the active ``ThemeSnapshot``
* the pending theme id (queued at safe frame boundary)
* subscriber callbacks for ``ThemeChanged``
* a fallback ``engine_defaults`` definition that always exists and provides
  every token

The plugin preload lifecycle removes themes automatically via
``unregister(owner)``; see :func:`bind_to_preload_cleanup` for the hook into
``PreloadManager._remove_editor_contribution_owner``.

Public surface is exposed through ``python/Infernux/engine/ui/theme.py``'s
``Theme`` class — plugins should not import this module directly.
"""

from __future__ import annotations

import threading
from typing import Callable, Sequence

from Infernux.engine.interaction._contributions import current_owner

from .theme_definition import (
    IconRef,
    RGBA,
    ThemeChanged,
    ThemeDefinition,
    ThemeExtension,
    ThemeKind,
    ThemeSnapshot,
    Vec2,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

class ThemeRegistry:
    """Process-wide theme registry. One instance, lazily created."""

    _instance: "ThemeRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        # id → (owner, definition)
        self._definitions: dict[str, tuple[str, ThemeDefinition]] = {}
        self._active_id: str | None = None
        self._active_snapshot: ThemeSnapshot | None = None
        self._pending_apply: str | None = None
        self._subscribers: list[Callable[[ThemeChanged], None]] = []
        # The always-available engine fallback. Provides every token by id.
        self._engine_defaults: ThemeDefinition | None = None
        # Track which owners have registered (used for bulk cleanup)
        self._owners_to_ids: dict[str, set[str]] = {}

    @classmethod
    def instance(cls) -> "ThemeRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Engine defaults ────────────────────────────────────────────────

    def set_engine_defaults(self, definition: ThemeDefinition) -> None:
        """Register the always-available engine fallback. Should be called
        once at engine startup with a definition that covers every token.

        If no theme is active yet, the engine defaults become the active
        snapshot so readers always have a value.
        """
        self._engine_defaults = definition
        if self._active_snapshot is None:
            self._active_snapshot = self._resolve(definition, owner="engine")
            self._active_id = definition.id

    def engine_defaults(self) -> ThemeDefinition | None:
        return self._engine_defaults

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        definition: ThemeDefinition,
        *,
        owner: str | None = None,
        replace: bool = False,
    ) -> None:
        """Add or update a theme definition.

        Owner is auto-captured from ``current_owner`` (set by
        ``_editor_contribution_scope`` during plugin preload) when not
        provided. Calling with the same owner+id updates in place.
        """
        if not definition.id:
            raise ValueError("ThemeDefinition.id must be non-empty")
        if owner is None:
            owner = current_owner.get() or "<unknown>"

        existing = self._definitions.get(definition.id)
        if existing is not None:
            existing_owner, _ = existing
            if existing_owner != owner and not replace:
                raise ValueError(
                    f"Theme id {definition.id!r} already registered by "
                    f"owner={existing_owner!r}"
                )

        self._definitions[definition.id] = (owner, definition)
        self._owners_to_ids.setdefault(owner, set()).add(definition.id)

    def unregister(self, owner: str) -> int:
        """Remove every theme definition owned by ``owner``. Returns count.

        Called by ``PreloadManager._remove_editor_contribution_owner`` when a
        package is uninstalled, reloaded, or fails to load.
        """
        ids = list(self._owners_to_ids.get(owner, ()))
        for tid in ids:
            self._definitions.pop(tid, None)
        self._owners_to_ids.pop(owner, None)

        # If the active theme just got removed, fall back to engine defaults
        # (NOT silently to nothing — the dev explicitly required this).
        if (
            self._active_snapshot is not None
            and self._active_snapshot.owner == owner
        ):
            self._pending_apply = None
            if self._engine_defaults is not None:
                self._apply_sync(self._engine_defaults.id)

        return len(ids)

    def unregister_id(self, theme_id: str, owner: str | None = None) -> bool:
        """Remove one specific theme id. Returns True if it was removed."""
        existing = self._definitions.get(theme_id)
        if existing is None:
            return False
        existing_owner, _ = existing
        if owner is not None and existing_owner != owner:
            return False
        self._definitions.pop(theme_id, None)
        ids_set = self._owners_to_ids.get(existing_owner)
        if ids_set is not None:
            ids_set.discard(theme_id)
            if not ids_set:
                self._owners_to_ids.pop(existing_owner, None)
        return True

    # ── Lookup ────────────────────────────────────────────────────────

    def list(self) -> Sequence[tuple[str, str, str]]:
        """Return ``[(id, display_name, owner), ...]`` for every registered theme."""
        return tuple(
            (tid, defn.display_name, owner)
            for tid, (owner, defn) in self._definitions.items()
        )

    def list_for_owner(self, owner: str) -> Sequence[str]:
        """Return theme ids owned by ``owner``."""
        return tuple(self._owners_to_ids.get(owner, ()))

    def get(self, theme_id: str) -> ThemeDefinition | None:
        entry = self._definitions.get(theme_id)
        return entry[1] if entry else None

    def owner_of(self, theme_id: str) -> str | None:
        entry = self._definitions.get(theme_id)
        return entry[0] if entry else None

    def current(self) -> ThemeSnapshot | None:
        return self._active_snapshot

    def active_id(self) -> str | None:
        return self._active_id

    # ── Apply ─────────────────────────────────────────────────────────

    def set_active(self, theme_id: str) -> None:
        """Queue a theme switch. The actual apply happens at the next safe
        frame boundary via :meth:`apply_pending`. Calling with the currently
        active id is a no-op.
        """
        if theme_id not in self._definitions:
            raise KeyError(f"Theme not registered: {theme_id!r}")
        if theme_id == self._active_id and self._pending_apply is None:
            return
        self._pending_apply = theme_id

    def has_pending(self) -> bool:
        return self._pending_apply is not None

    def apply_pending(self) -> bool:
        """Apply the pending theme switch. Call from the editor's main loop
        (frame-boundary safe). Returns True if a switch actually occurred.
        """
        if self._pending_apply is None:
            return False
        theme_id = self._pending_apply
        self._pending_apply = None
        return self._apply_sync(theme_id)

    def _apply_sync(self, theme_id: str) -> bool:
        entry = self._definitions.get(theme_id)
        if entry is None:
            return False
        owner, definition = entry
        old_snapshot = self._active_snapshot
        new_snapshot = self._resolve(definition, owner=owner)
        self._active_id = theme_id
        self._active_snapshot = new_snapshot
        self._fire(ThemeChanged(old=old_snapshot, new=new_snapshot))
        return True

    # ── Subscribers ───────────────────────────────────────────────────

    def subscribe(
        self,
        callback: Callable[[ThemeChanged], None],
    ) -> Callable[[], None]:
        """Register a ``ThemeChanged`` listener. Returns an unsubscribe callable."""
        self._subscribers.append(callback)

        def _unsub() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return _unsub

    def _fire(self, event: ThemeChanged) -> None:
        # Snapshot the list so unsubscribes during dispatch don't break iteration.
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                # Subscriber errors don't kill the apply pipeline.
                pass

    # ── Resolution ───────────────────────────────────────────────────

    def _resolve(self, definition: ThemeDefinition, owner: str) -> ThemeSnapshot:
        """Build an immutable snapshot. Honors ``based_on`` by overlaying on
        the resolved parent. Fills any remaining gaps with engine defaults.
        """
        base_colors: dict = {}
        base_sizes: dict = {}
        base_style: dict = {}
        base_icons: dict = {}

        # Inheritance (single level; cycle-safe because we only follow one hop
        # and the parent's owner is set to "<inherit>" to disambiguate).
        if definition.based_on:
            base_def = self.get(definition.based_on)
            if base_def is not None and base_def is not definition:
                base_snap = self._resolve(base_def, owner="<inherit>")
                base_colors = dict(base_snap.colors)
                base_sizes = dict(base_snap.sizes)
                base_style = dict(base_snap.imgui_style)
                base_icons = dict(base_snap.icons)

        # Apply this definition's overrides
        base_colors.update(definition.colors)
        base_sizes.update(definition.sizes)
        base_style.update(definition.imgui_style)
        base_icons.update(definition.icons)

        # Fill gaps with engine defaults (unless this IS the defaults)
        if self._engine_defaults is not None and self._engine_defaults is not definition:
            defaults_snap = self._resolve(self._engine_defaults, owner="<defaults>")
            for k, v in defaults_snap.colors.items():
                base_colors.setdefault(k, v)
            for k, v in defaults_snap.sizes.items():
                base_sizes.setdefault(k, v)
            for k, v in defaults_snap.imgui_style.items():
                base_style.setdefault(k, v)
            for k, v in defaults_snap.icons.items():
                base_icons.setdefault(k, v)

        return ThemeSnapshot(
            id=definition.id,
            display_name=definition.display_name,
            kind=definition.kind,
            owner=owner,
            colors=base_colors,
            imgui_style=base_style,
            sizes=base_sizes,
            icons=base_icons,
            extensions=tuple(definition.extends),
        )

    # ── Namespaced token reader (Phase 3) ────────────────────────────

    def token(self, namespace: str, key: str):
        """Read a namespaced extension token from the active snapshot, or
        None if not present. Phase 3 feature; safe to call earlier (returns
        None until extensions are populated).
        """
        snap = self._active_snapshot
        if snap is None:
            return None
        for ext in snap.extensions:
            if ext.namespace == namespace:
                return ext.tokens.get(key)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Preload cleanup hook
# ─────────────────────────────────────────────────────────────────────────────

def bind_to_preload_cleanup() -> None:
    """Wire ``ThemeRegistry.unregister`` into
    ``PreloadManager._remove_editor_contribution_owner`` so that any time a
    plugin is uninstalled, reloaded, or fails to load, its themes are
    retracted automatically.

    Idempotent: safe to call more than once.
    """
    from Infernux.plugins.preload import _remove_editor_contribution_owner

    registry = ThemeRegistry.instance()
    original = _remove_editor_contribution_owner

    def patched(owner: str, *, runtime: bool) -> bool:
        result = original(owner, runtime=runtime)
        if not runtime:
            registry.unregister(owner)
        return result

    # Patch the module-level symbol so existing callers see the change.
    import Infernux.plugins.preload as _preload_mod
    if getattr(_preload_mod, "_theme_cleanup_patched", False):
        return
    _preload_mod._remove_editor_contribution_owner = patched
    _preload_mod._theme_cleanup_patched = True
