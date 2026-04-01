"""
InxComponent - Base class for all Python-defined components.

Provides Unity-style lifecycle methods and property injection.
Users inherit from this class to create custom game logic.

Example:
    from Infernux.components import InxComponent, serialized_field
    
    class PlayerController(InxComponent):
        speed: float = serialized_field(default=5.0)
        
        def start(self):
            print("Player started!")
        
        def update(self, delta_time: float):
            pos = self.transform.position
            self.transform.position = Vector3(pos.x + self.speed * delta_time, pos.y, pos.z)
"""

from typing import Optional, Dict, Any, Type, TYPE_CHECKING, List
import copy
import threading
import weakref

from Infernux.lib import GameObject

if TYPE_CHECKING:
    from Infernux.lib import GameObject, Transform
    from Infernux.coroutine import Coroutine


class InxComponent:
    """
    Base class for Python-defined components.
    
    Lifecycle methods (override as needed):
        - awake(): Called when the component first becomes active in the hierarchy
        - start(): Called before first update, after all awake()
        - update(delta_time): Called every frame
        - late_update(delta_time): Called after all update()
        - on_destroy(): Called when component is removed/destroyed
        - on_enable(): Called when component becomes enabled
        - on_disable(): Called when component becomes disabled
    
    Serialized fields:
        - Use class-level ``serialized_field()`` declarations.

    Injected properties (read-only):
        - game_object: The GameObject this component is attached to
        - transform: Shortcut to game_object.get_transform()
        - enabled: Whether this component is active
        - component_id: Stable unique ID for this component instance
    """
    
    # Class-level storage for serialized field metadata
    _serialized_fields_: Dict[str, Any] = {}

    # Schema version for serialization compatibility
    __schema_version__ = 1

    # Active instance registry: go_id (int) → list of live InxComponent instances.
    # Used by GizmosCollector to skip the expensive get_all_objects() + get_py_components()
    # scene walk — instead we iterate only objects that actually have Python components.
    # Populated by _set_game_object(), removed by _call_on_destroy() or
    # _invalidate_native_binding() during scene rebuild/destruction.
    # Use a plain dict (not WeakValueDictionary) so we hold strong refs; instances
    # remove themselves from the registry in _call_on_destroy().
    _active_instances: Dict[int, List['InxComponent']] = {}

    # Component category for the Add Component menu.
    # Override in subclasses to group related components together.
    # Examples: "Physics", "Rendering", "Audio", "UI", etc.
    # When empty, script components default to the "Scripts" group.
    _component_category_: str = ""

    # Gizmo visibility: when True, on_draw_gizmos() is called every frame
    # for this component.  When False, on_draw_gizmos() is only called when
    # the owning GameObject (or one of its ancestors) is selected.
    # Subclasses can override: ``_always_show = False``
    _always_show: bool = True
    
    # Thread-safe component ID generator
    _next_component_id: int = 1
    _id_lock: threading.Lock = threading.Lock()
    
    def __init_subclass__(cls, **kwargs):
        """
        Called when a subclass is created. Collect class-level fields as serialized fields.
        
        This enables Unity-style field declaration:
            class Kobe(InxComponent):
                speed = 5.0
                count = int_field(10)
        """
        super().__init_subclass__(**kwargs)

        # ---- Enforce lifecycle: forbid __init__ override ----
        if '__init__' in cls.__dict__:
            raise TypeError(
                f"{cls.__qualname__} overrides __init__, which is forbidden. "
                f"InxComponent manages its own initialization internally. "
                f"Use awake() for one-time setup or start() for deferred init."
            )
        
        # Always create a fresh dict for this class (don't inherit from parent)
        cls._serialized_fields_ = {}
        
        # Collect own-class annotations for the annotation-only pass below
        own_annotations = cls.__dict__.get('__annotations__', {})

        # Only scan attributes defined directly on this class (not inherited)
        for attr_name in list(cls.__dict__):
            if attr_name.startswith('_'):
                continue
            
            # Get the raw attribute from class __dict__ to avoid descriptor protocol
            attr = cls.__dict__[attr_name]
            
            # Skip methods, properties, classmethods, staticmethods
            if callable(attr):
                continue
            if isinstance(attr, (property, classmethod, staticmethod)):
                continue
            
            # Register this field if it's a simple value or FieldMetadata
            from .serialized_field import (
                FieldMetadata, HiddenField, SerializedFieldDescriptor,
                infer_field_type_from_value, resolve_annotation,
            )
            
            # Skip hidden fields (marked with hide_field())
            if isinstance(attr, HiddenField):
                continue

            # CppProperty — delegates to a C++ component attribute.
            # Recognised via duck-typing marker to avoid circular imports
            # with builtin_component.py.
            if getattr(attr, '_is_cpp_property', False):
                if hasattr(attr, 'metadata'):
                    attr.metadata.name = attr_name
                    cls._serialized_fields_[attr_name] = attr.metadata
                continue
            
            # SerializedFieldDescriptor — __set_name__ already stored its
            # metadata, but we wiped _serialized_fields_ above.  Restore it.
            if isinstance(attr, SerializedFieldDescriptor):
                cls._serialized_fields_[attr_name] = attr.metadata
            elif isinstance(attr, FieldMetadata):
                # Already a field metadata object
                cls._serialized_fields_[attr_name] = attr
            elif attr is None:
                # Value is None — check type annotation for reference types
                # e.g. ``text: UIText = None`` or ``mat: Material = None``
                ann = own_annotations.get(attr_name)
                if ann is not None:
                    meta = resolve_annotation(ann)
                    if meta is not None:
                        meta.name = attr_name
                        descriptor = SerializedFieldDescriptor(meta)
                        descriptor.__set_name__(cls, attr_name)
                        setattr(cls, attr_name, descriptor)
                        cls._serialized_fields_[attr_name] = meta
            else:
                # Create metadata AND a descriptor from the plain value.
                # This ensures __set__ is intercepted for dirty-tracking
                # and undo recording, just like serialized_field() fields.
                from enum import Enum as _Enum
                field_type = infer_field_type_from_value(attr)
                enum_type = type(attr) if isinstance(attr, _Enum) else None
                metadata = FieldMetadata(
                    name=attr_name,
                    field_type=field_type,
                    default=attr,
                    enum_type=enum_type,
                )
                descriptor = SerializedFieldDescriptor(metadata)
                descriptor.__set_name__(cls, attr_name)
                setattr(cls, attr_name, descriptor)
                cls._serialized_fields_[attr_name] = metadata

        # ── Annotation-only pass: fields with a type hint but no value ──
        # e.g. ``text: UIText`` (no ``= ...`` at all)
        from .serialized_field import (
            FieldMetadata, HiddenField, SerializedFieldDescriptor,
            get_annotation_default, resolve_annotation,
        )
        for attr_name, ann in own_annotations.items():
            # Skip if already processed above (has an entry in __dict__)
            if attr_name in cls.__dict__:
                continue
            # Skip if already registered (e.g. inherited descriptor)
            if attr_name in cls._serialized_fields_:
                continue

            if attr_name.startswith('_'):
                default_value = get_annotation_default(ann)
                if default_value is not None:
                    hidden = HiddenField(default=default_value)
                    hidden.__set_name__(cls, attr_name)
                    setattr(cls, attr_name, hidden)
                continue

            meta = resolve_annotation(ann)
            if meta is not None:
                meta.name = attr_name
                descriptor = SerializedFieldDescriptor(meta)
                descriptor.__set_name__(cls, attr_name)
                setattr(cls, attr_name, descriptor)
                cls._serialized_fields_[attr_name] = meta
    
    def __init__(self):
        """Internal framework initialization — **do not override**.

        Subclasses must use lifecycle methods instead:
        - ``awake()`` — called once when the component is first created
        - ``start()`` — called before the first ``update()``, after all ``awake()``
        - ``on_destroy()`` — called when the component is removed / scene unloaded

        Overriding ``__init__`` is enforced as a ``TypeError`` at class
        creation time (see ``__init_subclass__``).
        """
        # Generate stable component ID immediately (thread-safe)
        with InxComponent._id_lock:
            self._component_id = InxComponent._next_component_id
            InxComponent._next_component_id += 1
        
        self._game_object: Optional['GameObject'] = None  # Reference to the owning GameObject
        self._game_object_ref: Optional[weakref.ref] = None  # Weak reference for safety
        self._cpp_component = None  # Native lifecycle authority (PyComponentProxy or built-in C++ component)
        self._enabled = True
        self._execution_order = 0
        self._has_started = False
        self._awake_called = False
        self._is_destroyed = False  # Track destruction state
        self._component_name = self.__class__.__name__
        self._script_guid: Optional[str] = None
        self._registered_go_id: Optional[int] = None  # go_id this comp is registered under
        self._native_generation: int = 0
        
        # Coroutine scheduler (lazy-created on first start_coroutine call)
        self._coroutine_scheduler = None
        
        # Initialize serialized fields with defaults (from class-level declarations)
        self._init_serialized_fields()

    def _init_serialized_fields(self):
        """Initialize all serialized fields with their default values."""
        from .serialized_field import get_serialized_fields, SerializedFieldDescriptor
        fields = get_serialized_fields(self.__class__)
        for name, metadata in fields.items():
            descriptor = self.__class__.__dict__.get(name)
            try:
                default_value = copy.deepcopy(metadata.default)
            except Exception as exc:
                # Some defaults (e.g. lambdas, C++ objects) can't be deepcopied
                default_value = metadata.default

            if isinstance(descriptor, SerializedFieldDescriptor):
                inst_id = id(self)
                with descriptor._lock:
                    descriptor._values[inst_id] = default_value
                    descriptor._weak_refs[inst_id] = weakref.ref(self, descriptor._make_ref_callback(inst_id))
            elif hasattr(descriptor, '_is_cpp_property'):
                # Skip ALL CppProperty descriptors — their values come from C++.
                continue
            elif metadata.default is not None:
                setattr(self, name, default_value)

    # ========================================================================
    # Property Injection (set by the engine)
    # ========================================================================

    @staticmethod
    def _is_native_game_object_alive(game_object: Optional['GameObject']) -> bool:
        """Return True when a native GameObject wrapper still points to live data."""
        if game_object is None:
            return False
        try:
            return isinstance(game_object, GameObject) and int(game_object.id) > 0
        except Exception:
            return False

    @staticmethod
    def _is_native_component_alive(component: Any) -> bool:
        """Return True when a native Component wrapper still points to live data."""
        if component is None:
            return False
        try:
            return int(component.component_id) > 0
        except Exception:
            return False
    
    def _try_get_game_object(self) -> Optional['GameObject']:
        """Return the owning GameObject, or None when the component is unbound."""
        if self._is_destroyed:
            return None
        cpp_component = self._get_bound_native_component()
        if cpp_component is not None:
            try:
                go = cpp_component.game_object
            except RuntimeError:
                self._invalidate_native_binding()
                return None
            if go is None:
                self._invalidate_native_binding()
                return None
            if not self._is_native_game_object_alive(go):
                self._invalidate_native_binding()
                return None
            self._game_object = go
            self._game_object_ref = weakref.ref(go)
            return go
        # Use weak reference if available for safety
        if self._game_object_ref is not None:
            go = self._game_object_ref()
            if go is None:
                # GameObject was destroyed, mark component as invalid
                self._game_object = None
                return None
            if not self._is_native_game_object_alive(go):
                self._invalidate_native_binding()
                return None
            return go
        if self._game_object is not None and not self._is_native_game_object_alive(self._game_object):
            self._invalidate_native_binding()
            return None
        return self._game_object

    @property
    def game_object(self) -> 'GameObject':
        """Get the owning GameObject during normal component lifetime."""
        go = self._try_get_game_object()
        if go is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.game_object is unavailable because the component is not bound to a live GameObject"
            )
        return go

    def _try_get_transform(self) -> Optional['Transform']:
        """Return the attached Transform, or None when the component is invalid."""
        go = self._try_get_game_object()
        if go is None:
            return None
        try:
            return go.get_transform()
        except RuntimeError:
            self._invalidate_native_binding()
            return None
    
    @property
    def transform(self) -> 'Transform':
        """Get the attached Transform during normal component lifetime."""
        transform = self._try_get_transform()
        if transform is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.transform is unavailable because the component is not bound to a live GameObject"
            )
        return transform
    
    @property
    def is_valid(self) -> bool:
        """Check if this component is still valid (not destroyed, has game_object)."""
        return not self._is_destroyed and self._try_get_game_object() is not None
    
    @property
    def enabled(self) -> bool:
        """Check if this component is enabled."""
        cpp_component = self._get_bound_native_component()
        if cpp_component is not None:
            try:
                self._enabled = bool(cpp_component.enabled)
            except RuntimeError:
                self._invalidate_native_binding()
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable this component.

        Native lifecycle is always authoritative once the component is bound.
        Before binding, the value is simply staged on the Python instance and
        will be consumed by the native proxy when attached.
        """
        if self._is_destroyed:
            return
        value = bool(value)
        if self._enabled == value and getattr(self, '_cpp_component', None) is None:
            return

        cpp_component = getattr(self, '_cpp_component', None)
        if cpp_component is not None:
            cpp_component.enabled = value
            return

        self._enabled = value
    
    @property
    def type_name(self) -> str:
        """Get the component type name."""
        return self._component_name

    @property
    def execution_order(self) -> int:
        """Execution order (lower value runs earlier)."""
        cpp_component = self._get_bound_native_component()
        if cpp_component is not None:
            try:
                return int(cpp_component.execution_order)
            except RuntimeError:
                self._invalidate_native_binding()
                return int(getattr(self, '_execution_order', 0))
        return int(getattr(self, '_execution_order', 0))

    @execution_order.setter
    def execution_order(self, value: int):
        cpp_component = getattr(self, '_cpp_component', None)
        if cpp_component is not None:
            cpp_component.execution_order = int(value)
            return
        self._execution_order = int(value)

    @property
    def component_id(self) -> int:
        """Get the stable component ID (assigned at construction)."""
        return self._component_id
    
    # ========================================================================
    # Internal methods (called by the engine)
    # ========================================================================
    
    def _set_game_object(self, game_object: Optional['GameObject']):
        """Internal: Set the owning GameObject. Called by the engine."""
        # ---- Update active-instances registry ----
        self._remove_from_active_registry()

        self._game_object = game_object
        # Also store weak reference for safe access
        if game_object is not None:
            self._game_object_ref = weakref.ref(game_object)
            # Register into active-instances table (works in both edit & play modes)
            go_id = game_object.id
            lst = InxComponent._active_instances.get(go_id)
            if lst is None:
                InxComponent._active_instances[go_id] = [self]
            else:
                if self not in lst:   # guard against duplicate calls
                    lst.append(self)
            self._registered_go_id = go_id
        else:
            self._game_object_ref = None

    def _remove_from_active_registry(self):
        """Remove this component from the active-instance registry."""
        old_id = getattr(self, '_registered_go_id', None)
        if old_id is None:
            return

        lst = InxComponent._active_instances.get(old_id)
        if lst is not None:
            if self in lst:
                lst.remove(self)
            if not lst:
                InxComponent._active_instances.pop(old_id, None)
        self._registered_go_id = None

    def _bind_native_component(self, cpp_component, game_object=None):
        """Bind this Python instance to its native lifecycle authority."""
        self._cpp_component = cpp_component
        self._native_generation += 1
        if game_object is not None:
            self._set_game_object(game_object)

        if cpp_component is not None:
            try:
                self._component_id = int(cpp_component.component_id)
            except Exception as exc:
                from Infernux.debug import Debug
                Debug.log_warning(f"[InxComponent] Failed to read component_id during bind: {exc}")
            try:
                self._execution_order = int(cpp_component.execution_order)
            except Exception as exc:
                from Infernux.debug import Debug
                Debug.log_warning(f"[InxComponent] Failed to read execution_order during bind: {exc}")
            try:
                self._enabled = bool(cpp_component.enabled)
            except Exception as exc:
                from Infernux.debug import Debug
                Debug.log_warning(f"[InxComponent] Failed to read enabled during bind: {exc}")

    def _sync_native_state(
        self,
        enabled: bool,
        awake_called: bool,
        has_started: bool,
        is_destroyed: bool,
        execution_order: int,
    ):
        """Mirror native lifecycle state into Python for diagnostics and tooling."""
        self._enabled = bool(enabled)
        self._awake_called = bool(awake_called)
        self._has_started = bool(has_started)
        self._is_destroyed = bool(is_destroyed)
        self._execution_order = int(execution_order)

    def _invalidate_native_binding(self):
        """Invalidate native references after scene rebuild/destruction."""
        self._cpp_component = None
        self._enabled = False
        self._awake_called = False
        self._has_started = False
        self._is_destroyed = True
        self._remove_from_active_registry()
        self._game_object = None
        self._game_object_ref = None

    def _get_bound_native_component(self):
        """Return the native component if still alive, otherwise invalidate it."""
        cpp_component = getattr(self, '_cpp_component', None)
        if cpp_component is None:
            return None
        try:
            comp_id = int(cpp_component.component_id)
        except RuntimeError:
            self._invalidate_native_binding()
            return None
        except Exception:
            self._invalidate_native_binding()
            return None
        if comp_id <= 0:
            self._invalidate_native_binding()
            return None
        return cpp_component
    
    # ------------------------------------------------------------------
    # Internal helper: safely call a user-overridden lifecycle method,
    # routing any exception to the engine Console so it is visible in
    # the editor (not just stderr).
    # ------------------------------------------------------------------
    def _safe_lifecycle_call(self, method_name: str, *args) -> bool:
        """Call *method_name* on self, catching and logging any exception."""
        try:
            getattr(self, method_name)(*args)
            return True
        except Exception as exc:
            # Route to DebugConsole so the Console Panel shows the error.
            try:
                from Infernux.debug import debug
                debug.log_exception(exc, context=self)
            except (ImportError, RuntimeError):
                # Absolute fallback if debug itself cannot be imported.
                import traceback
                traceback.print_exc()
            return False

    def _disable_after_awake_exception(self):
        """Unity disables a script component if its Awake throws."""
        self._enabled = False

        cpp_component = getattr(self, '_cpp_component', None)
        if cpp_component is None:
            return

        try:
            cpp_component.enabled = False
        except RuntimeError:
            self._invalidate_native_binding()
    
    def _call_awake(self):
        """Internal: Trigger awake lifecycle."""
        if self._awake_called:
            return
        self._awake_called = True
        if not self._safe_lifecycle_call("awake"):
            self._disable_after_awake_exception()
    
    def _call_start(self):
        """Internal: Trigger start lifecycle if not already called."""
        if self._has_started:
            return
        self._has_started = True
        self._safe_lifecycle_call("start")
    
    def _call_update(self, delta_time: float):
        """Internal: Trigger update lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("update", delta_time)
        # Tick coroutines after user update (matching Unity order)
        self._tick_coroutines_update(delta_time)

    def _call_fixed_update(self, fixed_delta_time: float):
        """Internal: Trigger fixed_update lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("fixed_update", fixed_delta_time)
        # Tick coroutines waiting for fixed_update
        self._tick_coroutines_fixed_update(fixed_delta_time)

    def _call_late_update(self, delta_time: float):
        """Internal: Trigger late_update lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("late_update", delta_time)
        # Tick coroutines waiting for end-of-frame
        self._tick_coroutines_late_update(delta_time)

    def _tick_coroutines_update(self, delta_time: float):
        """Advance coroutine work scheduled for the Update phase."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.tick_update(delta_time)

    def _tick_coroutines_fixed_update(self, fixed_delta_time: float):
        """Advance coroutine work scheduled for the fixed-update phase."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.tick_fixed_update(fixed_delta_time)

    def _tick_coroutines_late_update(self, delta_time: float):
        """Advance coroutine work scheduled for the late-update phase."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.tick_late_update(delta_time)

    def _stop_coroutines_for_game_object_deactivate(self):
        """Unity stops all coroutines when the owning GameObject is deactivated."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop_all()
            self._coroutine_scheduler = None
    
    @classmethod
    def _clear_all_instances(cls) -> None:
        """Clear the active-instances registry (call on scene unload/reload)."""
        cls._active_instances.clear()

    def _call_on_destroy(self):
        """Internal: Trigger on_destroy lifecycle."""
        if self._is_destroyed:
            return  # Already destroyed, don't call again
        self._is_destroyed = True
        self._enabled = False
        # Stop all coroutines before on_destroy callback
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop_all()
            self._coroutine_scheduler = None
        # Remove from active-instances registry (safety net; _set_game_object(None)
        # should have done this already, but guard against missed calls)
        self._remove_from_active_registry()
        if self._awake_called:
            self._safe_lifecycle_call("on_destroy")
        # Clear references to help garbage collection
        self._cpp_component = None
        self._game_object = None
        self._game_object_ref = None

    def _call_on_enable(self):
        """Internal: Trigger on_enable lifecycle."""
        self._enabled = True
        self._safe_lifecycle_call("on_enable")

    def _call_on_disable(self):
        """Internal: Trigger on_disable lifecycle."""
        self._enabled = False
        self._safe_lifecycle_call("on_disable")

    def _call_on_validate(self):
        """Internal: Trigger on_validate lifecycle (editor only)."""
        self._safe_lifecycle_call("on_validate")

    def _call_reset(self):
        """Internal: Trigger reset lifecycle (editor only)."""
        self._safe_lifecycle_call("reset")

    def _call_on_after_deserialize(self):
        """Internal: Trigger on_after_deserialize lifecycle."""
        self._safe_lifecycle_call("on_after_deserialize")

    def _call_on_before_serialize(self):
        """Internal: Trigger on_before_serialize lifecycle."""
        self._safe_lifecycle_call("on_before_serialize")

    # ------------------------------------------------------------------
    # Physics collision / trigger callbacks (Unity-style)
    # ------------------------------------------------------------------

    def _call_on_collision_enter(self, collision):
        """Internal: Trigger on_collision_enter lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_collision_enter", collision)

    def _call_on_collision_stay(self, collision):
        """Internal: Trigger on_collision_stay lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_collision_stay", collision)

    def _call_on_collision_exit(self, collision):
        """Internal: Trigger on_collision_exit lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_collision_exit", collision)

    def _call_on_trigger_enter(self, other):
        """Internal: Trigger on_trigger_enter lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_trigger_enter", other)

    def _call_on_trigger_stay(self, other):
        """Internal: Trigger on_trigger_stay lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_trigger_stay", other)

    def _call_on_trigger_exit(self, other):
        """Internal: Trigger on_trigger_exit lifecycle."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_trigger_exit", other)

    def _call_on_draw_gizmos(self):
        """Internal: Trigger on_draw_gizmos lifecycle (editor only)."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_draw_gizmos")

    def _call_on_draw_gizmos_selected(self):
        """Internal: Trigger on_draw_gizmos_selected lifecycle (editor only)."""
        if not self.enabled:
            return
        self._safe_lifecycle_call("on_draw_gizmos_selected")

    # ========================================================================
    # Lifecycle methods (override in subclasses)
    # ========================================================================
    
    def awake(self):
        """
        Called when the component first becomes active in the hierarchy.
        Use for initialization that doesn't depend on other components.
        """
        pass
    
    def start(self):
        """
        Called before the first update, after all awake() calls.
        Use for initialization that depends on other components being ready.
        """
        pass
    
    def update(self, delta_time: float):
        """
        Called every frame.
        
        Args:
            delta_time: Time in seconds since last frame
        """
        pass

    def fixed_update(self, fixed_delta_time: float):
        """
        Called at a fixed time step (default 50 Hz).
        Use for physics and deterministic logic.
        
        Args:
            fixed_delta_time: Fixed time step in seconds
        """
        pass

    def late_update(self, delta_time: float):
        """
        Called every frame after all update() calls.
        Useful for camera follow, physics cleanup, etc.
        
        Args:
            delta_time: Time in seconds since last frame
        """
        pass
    
    def destroy(self):
        """Remove this component from its owning GameObject (Unity-style).

        The component's ``on_destroy`` lifecycle hook will be called.
        After this call the component is considered destroyed and should
        not be used further.
        """
        if self._is_destroyed:
            return
        go = self._try_get_game_object()
        if go is None:
            return
        # Use the C++ binding appropriate for the component type
        cpp = self._get_bound_native_component()
        if cpp is not None:
            go.remove_component(cpp)
        else:
            go.remove_py_component(self)

    def on_destroy(self):
        """
        Called when the component is being destroyed.
        Use for cleanup (unsubscribe events, release resources).
        """
        pass

    def on_enable(self):
        """
        Called when the component becomes enabled.
        Use for subscribing to events, starting coroutines, etc.
        """
        pass

    def on_disable(self):
        """
        Called when the component becomes disabled.
        Use for unsubscribing from events and releasing active subscriptions.
        Disabling the script itself does not stop Unity-style coroutines; deactivating
        the owning GameObject does.
        """
        pass

    def on_validate(self):
        """
        Called in the editor when the component is loaded or a value changes.
        Use for validation and clamping of serialized values.
        NOTE: This is editor-only; do not use for gameplay logic.
        """
        pass

    def reset(self):
        """
        Called in the editor when the user selects "Reset" on the component.
        Use to restore default values.
        NOTE: This is editor-only; do not use for gameplay logic.
        """
        pass

    def on_after_deserialize(self):
        """
        Called after component fields have been deserialized.
        Use to rebuild runtime references that weren't serialized.
        
        Example:
            def on_after_deserialize(self):
                # Rebuild cached references
                self._cached_renderer = self.game_object.get_component(MeshRenderer)
        """
        pass

    def on_before_serialize(self):
        """
        Called before component fields are serialized.
        Use to prepare data for serialization.
        """
        pass

    # ========================================================================
    # Physics collision / trigger callbacks (Unity-style)
    # ========================================================================

    def on_collision_enter(self, collision):
        """
        Called when this collider starts touching another collider.

        Args:
            collision: CollisionInfo with contact details (collider,
                       game_object, contact_point, contact_normal,
                       relative_velocity, impulse).
        """
        pass

    def on_collision_stay(self, collision):
        """
        Called every fixed-update while two colliders remain in contact.

        Args:
            collision: CollisionInfo with contact details.
        """
        pass

    def on_collision_exit(self, collision):
        """
        Called when two colliders stop touching.

        Args:
            collision: CollisionInfo with contact details.
        """
        pass

    def on_trigger_enter(self, other):
        """
        Called when another collider enters this trigger volume.

        Args:
            other: The other Collider that entered.
        """
        pass

    def on_trigger_stay(self, other):
        """
        Called every fixed-update while another collider is inside this trigger.

        Args:
            other: The other Collider that is inside.
        """
        pass

    def on_trigger_exit(self, other):
        """
        Called when another collider exits this trigger volume.

        Args:
            other: The other Collider that exited.
        """
        pass

    def on_draw_gizmos(self):
        """
        Called every frame in the editor to draw gizmos for this component.

        Override this to draw custom visual aids (wireframes, lines, etc.)
        using the ``Gizmos`` API.  When ``always_show`` is False on this
        component, this callback is only invoked when the owning
        GameObject (or one of its ancestors) is selected.

        Example::

            from Infernux.gizmos import Gizmos

            def on_draw_gizmos(self):
                Gizmos.color = (0, 1, 0)
                Gizmos.draw_wire_sphere(self.transform.position, 2.0)
        """
        pass

    def on_draw_gizmos_selected(self):
        """
        Called every frame in the editor ONLY when this object is selected.

        Use this for gizmos that should only appear when the user is
        inspecting this specific object.

        Example::

            from Infernux.gizmos import Gizmos

            def on_draw_gizmos_selected(self):
                Gizmos.color = (1, 1, 0)
                Gizmos.draw_wire_cube(self.transform.position, (1, 1, 1))
        """
        pass

    # ========================================================================
    # Coroutine support (Unity-style)
    # ========================================================================

    def start_coroutine(self, generator) -> 'Coroutine':
        """Start a coroutine on this component.

        Args:
            generator: A generator object (call your generator function first).

        Returns:
            A :class:`~Infernux.coroutine.Coroutine` handle that can be passed
            to :meth:`stop_coroutine` or ``yield``-ed from another coroutine to
            wait for completion.

        Example::

            from Infernux.coroutine import WaitForSeconds

            class Enemy(InxComponent):
                def start(self):
                    self.start_coroutine(self.patrol())

                def patrol(self):
                    while True:
                        debug.log("Moving left")
                        yield WaitForSeconds(2)
                        debug.log("Moving right")
                        yield WaitForSeconds(2)
        """
        from Infernux.coroutine import CoroutineScheduler
        if self._coroutine_scheduler is None:
            self._coroutine_scheduler = CoroutineScheduler()
        return self._coroutine_scheduler.start(generator, owner=self)

    def stop_coroutine(self, coroutine) -> None:
        """Stop a specific coroutine previously started with :meth:`start_coroutine`.

        Args:
            coroutine: The :class:`~Infernux.coroutine.Coroutine` handle.
        """
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop(coroutine)

    def stop_all_coroutines(self) -> None:
        """Stop **all** coroutines running on this component."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop_all()

    # ========================================================================
    # Serialization (used by Play Mode snapshot)
    # ========================================================================
    
    def _serialize_fields(self) -> str:
        """
        Serialize all serialized fields to JSON string.
        Called by C++ PyComponentProxy::Serialize().
        
        Returns:
            JSON string of field values
        """
        import json
        from .serialized_field import get_serialized_fields
        
        # Call on_before_serialize hook
        self._call_on_before_serialize()
        
        fields = get_serialized_fields(self.__class__)
        data = {
            "__schema_version__": getattr(self, "__schema_version__", 1),
            "__type_name__": self.__class__.__name__,
            "__component_id__": self._component_id,
        }
        from .serialized_field import get_raw_field_value
        for name, meta in fields.items():
            try:
                value = get_raw_field_value(self, name)
                data[name] = self._serialize_value(value)
            except Exception as exc:
                import logging
                logging.getLogger("Infernux.serialize").error(
                    "Failed to serialize field '%s' on %s: %s",
                    name, self.__class__.__name__, exc,
                )
                data[name] = None
        
        return json.dumps(data)
    
    def _deserialize_fields(self, json_str: str, *, _skip_on_after_deserialize: bool = False):
        """
        Restore serialized field values from JSON string.
        Calls on_after_deserialize() after restoration unless suppressed.
        
        Args:
            json_str: JSON string of field values
            _skip_on_after_deserialize: If True, suppress the automatic
                on_after_deserialize callback.  Used during batch component
                restoration where the caller issues the callback explicitly
                after all components are attached to the scene.
        """
        import json
        from .serialized_field import get_serialized_fields
        
        data = json.loads(json_str)
        schema_version = data.get("__schema_version__", None)
        current_version = getattr(self, "__schema_version__", 1)

        if schema_version is not None and schema_version != current_version:
            # Python-side schema migration for user scripts.
            # NOTE: C++ components have a *separate* schema_version tracked
            # in Component::Serialize/Deserialize (Component.cpp).  That
            # version covers the base wire format; this Python version
            # covers per-script field layout changes.  The two systems
            # are independent — keep both in mind when adding new base fields.
            migrate = getattr(self.__class__, '__migrate__', None)
            if migrate is not None:
                try:
                    data = migrate(data, schema_version)
                except Exception as exc:
                    from Infernux.debug import Debug
                    Debug.log_error(
                        f"Schema migration failed for {self.__class__.__name__} "
                        f"(v{schema_version}→v{current_version}): {exc}"
                    )
            else:
                from Infernux.debug import Debug
                Debug.log_warning(
                    f"Component schema mismatch: {self.__class__.__name__} "
                    f"(saved={schema_version}, current={current_version}). "
                    f"Define a __migrate__(data, from_version) classmethod to handle this."
                )

        # Restore component ID if present
        saved_id = data.get("__component_id__")
        if saved_id is not None:
            self._component_id = int(saved_id)
            # Ensure ID generator is ahead of restored ID
            with InxComponent._id_lock:
                if InxComponent._next_component_id <= self._component_id:
                    InxComponent._next_component_id = self._component_id + 1

        fields = get_serialized_fields(self.__class__)

        self._inf_deserializing = True
        try:
            for name, value in data.items():
                if name.startswith("__"):
                    continue
                if name in fields:
                    meta = fields[name]
                    value = self._deserialize_value(value, meta)
                    setattr(self, name, value)
        finally:
            self._inf_deserializing = False

        # Call on_after_deserialize hook
        if not _skip_on_after_deserialize:
            self._call_on_after_deserialize()


    def _serialize_value(self, value: Any):
        """Serialize a value into JSON-friendly format."""
        if isinstance(value, (bool, int, float, str, type(None))):
            return value

        # Recursively serialize list/dict so that nested refs/enums survive
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Enum support: store type name + member name
        from enum import Enum as _Enum
        if isinstance(value, _Enum):
            return {"__enum__": type(value).__qualname__, "name": value.name}

        # SerializableObject — nested data objects
        from Infernux.components.serializable_object import SerializableObject
        if isinstance(value, SerializableObject):
            return value._serialize()

        # ComponentRef — component reference
        from Infernux.components.ref_wrappers import GameObjectRef, MaterialRef, ComponentRef
        if isinstance(value, ComponentRef):
            return value._serialize()

        # GameObjectRef (null-safe wrapper) — store persistent ID
        if isinstance(value, GameObjectRef):
            return {"__game_object__": value.persistent_id}

        # PrefabRef (asset reference) — store GUID + path hint
        from Infernux.components.ref_wrappers import PrefabRef
        if isinstance(value, PrefabRef):
            return value._serialize()

        # MaterialRef (null-safe wrapper) — store GUID + path_hint
        if isinstance(value, MaterialRef):
            d = {"__material_ref__": value.guid}
            if value._path_hint:
                d["__path_hint__"] = value._path_hint
            return d

        # TextureRef — store GUID + path_hint
        from Infernux.core.asset_ref import TextureRef, ShaderRef, AudioClipRef
        if isinstance(value, TextureRef):
            d: dict = {"__texture_ref__": value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

        # ShaderRef — store GUID + path_hint
        if isinstance(value, ShaderRef):
            d = {"__shader_ref__": value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

        # AudioClipRef — store GUID + path_hint
        if isinstance(value, AudioClipRef):
            d = {"__audio_clip_ref__": value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

        # Raw GameObject reference — store persistent ID (scene-stable)
        if hasattr(value, 'id') and hasattr(value, 'name') and hasattr(value, 'transform'):
            return {"__game_object__": int(value.id)}

        # Raw Material reference — store GUID via AssetDatabase
        try:
            from Infernux.core.material import Material
            if isinstance(value, Material):
                guid = MaterialRef._extract_guid(value)
                if guid:
                    return {"__material_ref__": guid}
                # Fallback: store path if no GUID available
                path = getattr(value.native, 'file_path', '') if value.native else ''
                return {"__material_ref__": path or value.name}
        except ImportError:
            pass

        # Vec2/3/4 support (pybind types expose x/y/z/w)
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z") and hasattr(value, "w"):
            return [float(value.x), float(value.y), float(value.z), float(value.w)]
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            return [float(value.x), float(value.y), float(value.z)]
        if hasattr(value, "x") and hasattr(value, "y"):
            return [float(value.x), float(value.y)]

        # Fallback: warn and return None (do NOT silently degrade to str)
        import logging
        logging.getLogger("Infernux.serialize").warning(
            "Cannot serialize value of type %s — returning None. "
            "Define a SerializableObject or register a custom adapter.",
            type(value).__name__,
        )
        return None

    def _deserialize_value(self, value: Any, field_meta_or_type):
        """Deserialize a value based on FieldType."""
        from Infernux.components.serialized_field import FieldType
        from Infernux.components._serialize_helpers import make_null_ref, deserialize_dict_ref
        from Infernux.math import Vector2, Vector3, vec4f

        if hasattr(field_meta_or_type, 'field_type'):
            field_type = field_meta_or_type.field_type
            element_type = getattr(field_meta_or_type, 'element_type', None)
        else:
            field_type = field_meta_or_type
            element_type = None

        # Null values for ref types → return a null ref wrapper (not raw None)
        if value is None:
            return make_null_ref(field_type, field_meta_or_type)

        if field_type == FieldType.SERIALIZABLE_OBJECT:
            if isinstance(value, dict):
                from Infernux.components.serializable_object import SerializableObject
                if '__serializable_type__' in value:
                    return SerializableObject._deserialize(value)
                so_cls = getattr(field_meta_or_type, 'serializable_class', None)
                if so_cls:
                    return so_cls._deserialize(value)
            return value

        if field_type == FieldType.COMPONENT:
            if isinstance(value, dict) and '__component_ref__' in value:
                from Infernux.components.ref_wrappers import ComponentRef
                return ComponentRef._from_dict(value['__component_ref__'])
            return value

        if field_type == FieldType.LIST:
            if not isinstance(value, list):
                return []
            if element_type == FieldType.SERIALIZABLE_OBJECT:
                from Infernux.components.serializable_object import SerializableObject
                elem_cls = getattr(field_meta_or_type, 'element_class', None)
                result = []
                for item in value:
                    if isinstance(item, dict):
                        if '__serializable_type__' in item:
                            result.append(SerializableObject._deserialize(item))
                        elif elem_cls:
                            result.append(elem_cls._deserialize(item))
                        else:
                            result.append(item)
                    else:
                        result.append(item)
                return result
            return [self._deserialize_value(item, element_type or FieldType.UNKNOWN) for item in value]

        if field_type == FieldType.VEC2:
            return self._to_vec(value, 2, Vector2)
        if field_type == FieldType.VEC3:
            return self._to_vec(value, 3, Vector3)
        if field_type == FieldType.VEC4:
            return self._to_vec(value, 4, vec4f)
        if field_type == FieldType.ENUM and isinstance(value, dict) and "__enum__" in value:
            return self._deserialize_enum(value)

        if isinstance(value, dict):
            return deserialize_dict_ref(value)

        return value

    def _deserialize_enum(self, value: dict):
        """Reconstruct an enum member from {__enum__, name}."""
        from .serialized_field import get_serialized_fields
        fields = get_serialized_fields(self.__class__)
        for meta in fields.values():
            if meta.enum_type is not None and meta.enum_type.__qualname__ == value["__enum__"]:
                return meta.enum_type[value["name"]]
        return value  # fallback: return dict as-is

    def _to_vec(self, value: Any, n: int, ctor):
        """Convert list/tuple/string to vec type if possible."""
        if isinstance(value, (list, tuple)) and len(value) >= n:
            return ctor(*[float(value[i]) for i in range(n)])
        if isinstance(value, str):
            cleaned = value.strip().replace("<", "").replace(">", "").replace("(", "").replace(")", "")
            parts = [p for p in cleaned.split(",") if p.strip()]
            if len(parts) >= n:
                nums = [float(p.strip()) for p in parts[:n]]
                return ctor(*nums)
        return value
    
    # ========================================================================
    # Utility methods
    # ========================================================================
    
    # ========================================================================
    # Tag & Layer convenience properties
    # ========================================================================

    @property
    def tag(self) -> str:
        """Get the tag of the attached GameObject."""
        go = self._try_get_game_object()
        if go is not None and hasattr(go, 'tag'):
            return go.tag
        return "Untagged"

    @tag.setter
    def tag(self, value: str):
        """Set the tag of the attached GameObject."""
        go = self._try_get_game_object()
        if go is not None and hasattr(go, 'tag'):
            go.tag = value

    @property
    def game_object_layer(self) -> int:
        """Get the layer of the attached GameObject."""
        go = self._try_get_game_object()
        if go is not None and hasattr(go, 'layer'):
            return go.layer
        return 0

    @game_object_layer.setter
    def game_object_layer(self, value: int):
        """Set the layer of the attached GameObject."""
        go = self._try_get_game_object()
        if go is not None and hasattr(go, 'layer'):
            go.layer = value

    def compare_tag(self, tag: str) -> bool:
        """Returns True if the attached GameObject's tag matches the given tag."""
        go = self._try_get_game_object()
        if go is not None and hasattr(go, 'compare_tag'):
            return go.compare_tag(tag)
        return False

    def __repr__(self) -> str:
        return f"<{self._component_name} id={self._component_id} enabled={self.enabled}>"


# ═══════════════════════════════════════════════════════════════════════════
#  BrokenComponent — placeholder for scripts that failed to load
# ═══════════════════════════════════════════════════════════════════════════

class BrokenComponent(InxComponent):
    """Placeholder attached when a script fails to load or throws at import time.

    Keeps the original serialized field JSON so that saving the scene
    preserves the data verbatim.  The Inspector shows an error banner
    instead of field widgets.  Play mode is blocked while any
    BrokenComponent exists.
    """

    _is_broken: bool = True
    _broken_error: str = ""
    _broken_fields_json: str = ""
    _broken_type_name: str = ""

    @property
    def type_name(self) -> str:
        return self._broken_type_name or "BrokenComponent"

    def _serialize_fields(self) -> str:
        return self._broken_fields_json or "{}"
