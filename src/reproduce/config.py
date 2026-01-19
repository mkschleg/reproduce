from __future__ import annotations
import os
import yaml
import itertools
from dataclasses import (
    dataclass,
    fields,
    field,
    make_dataclass,
    asdict,
    is_dataclass,
    MISSING
)
from typing import Any, get_type_hints
import typing
import inspect
from copy import deepcopy
import shutil
import hashlib

import inspect
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, make_dataclass
from typing import Any, get_type_hints


# ============================================================
# 1) Core helpers
# ============================================================

def _filter_args(func, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter a keyword dict down to only parameters accepted by `func`.

    This is used when instantiating class-backed factories or calling function-backed
    factories, so that extra keys (e.g. registry type tags, nested config keys, etc.)
    do not get passed into an implementation signature that does not accept them.

    Notes:
      - "self" is always excluded, since filtering is typically done against __init__.
      - Only parameters present in the function signature are retained.
    """
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters and k != "self"}


def _default_for_dc_field(f) -> Any:
    """
    Return the effective default value for a dataclass Field.

    Behavior:
      - If the field has an explicit default, return it.
      - If it has a default_factory, call it and return the produced value.
      - Otherwise return MISSING to indicate the field is required.
    """
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[attr-defined]
        return f.default_factory()         # type: ignore[misc]
    return MISSING


def _coerce_value(typ, val):
    """
    Coerce a config value into a typed value for dataclass construction.

    Current behavior:
      - If the field type exposes a `from_config` classmethod and val is not None,
        delegate parsing to that method. This enables nested config parsing for
        factory dataclasses or other typed nodes.
      - Otherwise, return the value unchanged.

    This is intentionally lightweight: it does not do generic typing coercions.
    """
    # Recursive config parsing: if the field type supports from_config and we got a value.
    # print(typ, val)
    if val is not None and hasattr(typ, "from_config"):
        return typ.from_config(val)
    return val


def _build_instance(inst, args: tuple[Any, ...] = (), context: dict[str, Any] | None = None):
    """
    Build an object tree from a config dataclass instance.

    This is the unified "build" engine used by two modes:
      1) Plain config dataclasses created via `dataclass_factory(...)`:
         - No positional args, context is keyword-only.
      2) Callable/class-backed factories created via `make_dataclass_from_callable(...)`:
         - A fixed number of positional "context args" may be passed (e.g., RNG keys),
           identified via `_context_arg_names`.
         - Additional build-time context can also be passed as keywords.

    Parameters
    ----------
    inst:
        The config dataclass instance (factory dataclass subclass) being built.
    args:
        Positional context arguments, mapped to `_context_arg_names` by position.
        Only relevant for callable/class-backed factories.
    context:
        Keyword context injected into the build. This is merged with positional context.

    Context propagation rules
    ------------------------
    - `ctx_map` is the global context visible to all fields, combining:
        * `context` (keyword context)
        * `args` mapped onto `_context_arg_names`
    - For nested fields, you can optionally provide a *namespaced* context entry
      with the same name as the field, containing a dict. Example:
        build(policy_net={"rng": ...})
      That dict is merged on top of `ctx_map` for the nested node build call.

    Build semantics
    --------------
    For each dataclass field:
      - If the field value has a `.build(...)` method, it is treated as a child node
        and built recursively with the composed context.
      - Otherwise, the value is used as-is.

    Final construction
    ------------------
    - If `inst` is a class-backed factory (`_impl_cls` is set), instantiate the
      implementation class.
    - If `inst` is a function-backed factory (`_construct_fn` is set), call it.
    - Otherwise, return a rebuilt config dataclass with built children (pure config mode).
    """
    context = dict(context or {})
    ctx_names = tuple(getattr(inst, "_context_arg_names", ()))
    ctx_map = {**context, **{n: a for n, a in zip(ctx_names, args)}}

    # 1) Build child fields (recursively where applicable).
    built: dict[str, Any] = {}
    for f in fields(inst):
        val = getattr(inst, f.name)

        # Namespaced context for nested node if caller provided e.g. policy_net={...}
        nested_ctx = context.get(f.name, {})
        subctx = {**ctx_map, **nested_ctx} if isinstance(nested_ctx, dict) else ctx_map

        if hasattr(val, "build"):
            # Pass the same positional context args plus keyword context to children.
            val = val.build(*args, **subctx)
        built[f.name] = val

    # 2) Merge built values with context to prepare arguments for implementation calls.
    merged = {**ctx_map, **built}
    # Remove the factory discriminant key (e.g. "type") so it isn't forwarded.
    merged.pop(getattr(inst, "_factory_key", None), None)

    impl_cls = getattr(inst, "_impl_cls", None)
    construct_fn = getattr(inst, "_construct_fn", None)

    # 3) Class-backed factory: instantiate the implementation class.
    if impl_cls is not None:
        # Positional context args are passed first, in declared order.
        vals = [ctx_map[n] for n in ctx_names if n in ctx_map]
        # Only pass kwargs accepted by __init__.
        filtered = _filter_args(impl_cls.__init__, merged)
        # Avoid passing context args both positionally and as kwargs.
        for n in ctx_names:
            filtered.pop(n, None)
        return impl_cls(*vals, **filtered)

    # 4) Function-backed factory: call the constructor function.
    if construct_fn is not None:
        vals = [ctx_map[n] for n in ctx_names if n in ctx_map]
        filtered = _filter_args(construct_fn, merged)
        for n in ctx_names:
            filtered.pop(n, None)
        return construct_fn(*vals, **filtered)

    # 5) Pure dataclass build: return a rebuilt config dataclass with built children.
    return type(inst)(**built)


# ============================================================
# 2) Factory decorator with optional registry/dispatch
# ============================================================

def dataclass_factory(key_name: str | None = None, *, strict: bool = True, buildable: bool = True):
    """
    Decorator: turn a dataclass into a config factory with optional registry dispatch.

    Features
    --------
    1) Dataclass conversion:
       The decorated class is converted to a dataclass via `@dataclass`.

    2) Optional registry/dispatch (if `key_name` is provided):
       - Subclasses can register themselves under a string key via:
           class MySub(Base, type_value="my_key"): ...
       - `Base.from_config(cfg)` will dispatch to the appropriate subclass based on
         `cfg[key_name]` (or a literal key when dispatching via `create`).

    3) from_config(cfg: dict) -> instance:
       - Validates keys when `strict=True`.
       - Applies nested parsing if a field type provides `from_config`.
       - Preserves the discriminant key in the created instance (if missing in cfg).

    4) create(key_value, **kwargs) -> instance:
       - Convenience constructor that can also dispatch using a registry key.

    5) build(**context) -> built object:
       - If `buildable=True`, attaches a `.build(**context)` method that delegates to
         `_build_instance`.
       - In pure dataclass mode (no backing impl), `.build(...)` returns a rebuilt config
         dataclass with built children.
       - In callable/class-backed mode, `.build(...)` can construct real objects.

    Parameters
    ----------
    key_name:
        Name of the discriminant field (commonly "type"). If provided, registry and
        dispatch logic is enabled.
    strict:
        If True, reject unexpected keys in config dicts passed to `from_config`.
    buildable:
        If True, attach `.build(**context)` to the decorated class.
    """

    def decorator(cls):
        cls = dataclass(cls)

        # ---------------- Registry ----------------
        if key_name:
            # `_factory_key` is the discriminant field name (e.g. "type").
            cls._factory_key = key_name
            # Registry maps key -> subclass.
            cls._registry: dict[str, type] = {}

            @classmethod
            def __init_subclass__(sub_cls, **kwargs):
                """
                Register subclasses declared with e.g. `type_value="foo"`.

                The value is taken from kwargs using `<key_name>_value`.
                """
                key_value = kwargs.pop(f"{key_name}_value", None)
                super(cls, sub_cls).__init_subclass__(**kwargs)
                if key_value:
                    cls._registry[key_value] = sub_cls

            @classmethod
            def _dispatch(sub_cls, cfg_or_key):
                """
                Resolve a subclass from either:
                  - a config dict containing `key_name`, or
                  - a literal registry key.
                """
                key_value = cfg_or_key.get(key_name) if isinstance(cfg_or_key, dict) else cfg_or_key
                if key_value not in sub_cls._registry:
                    raise ValueError(
                        f"Unknown {key_name!r} value {key_value!r}. "
                        f"Valid: {sorted(sub_cls._registry.keys())}"
                    )
                return sub_cls._registry[key_value]

            cls.__init_subclass__ = __init_subclass__
            cls._dispatch = _dispatch

        # ---------------- from_config ----------------
        @classmethod
        def from_config(sub_cls, cfg: dict):
            """
            Construct an instance from a python dict.

            Steps:
              1) If registry-enabled, dispatch to the concrete subclass using cfg[key_name].
              2) If cfg lacks the discriminant key, synthesize it based on registry mapping
                 so the resulting dataclass instance still has the type populated.
              3) If `strict`, reject keys not present as dataclass fields.
              4) For each field:
                 - Use cfg value if provided, else use dataclass default/default_factory.
                 - If the field's type supports `from_config`, recurse into it.
              5) Instantiate the dataclass.

            Required fields (no default) are left absent; dataclass __init__ will raise.
            """
            # dispatch if registry-enabled
            if key_name and hasattr(sub_cls, "_dispatch"):
                sub_cls = sub_cls._dispatch(cfg)

            # add back missing type key if necessary (keeps .type populated)
            if key_name and key_name not in cfg:
                for k, v in getattr(cls, "_registry", {}).items():
                    if v is sub_cls:
                        cfg = {**cfg, key_name: k}
                        break

            # strict key validation
            field_names = {f.name for f in fields(sub_cls)}
            unexpected = set(cfg) - field_names
            if strict and unexpected:
                raise TypeError(f"Unexpected config keys: {sorted(unexpected)}")

            # field parsing + nested from_config
            init_kwargs = {}
            for f in fields(sub_cls):
                default = _default_for_dc_field(f)
                val = cfg.get(f.name, default)
                if val is MISSING:
                    # required dataclass arg (no default) must be provided
                    # let dataclass constructor raise its normal TypeError
                    continue
                init_kwargs[f.name] = _coerce_value(f.type, val)

            return sub_cls(**init_kwargs)

        # ---------------- create ----------------
        @classmethod
        def create(sub_cls, key_value=None, **kwargs):
            """
            Convenience constructor for programmatic creation.

            If registry-enabled:
              - `key_value` is treated as the registry key and used to dispatch.
              - The discriminant field is injected into kwargs if not provided.
            """
            if key_name and hasattr(sub_cls, "_dispatch"):
                sub_cls = sub_cls._dispatch(key_value)
            if key_name and key_name not in kwargs:
                kwargs[key_name] = key_value
            return sub_cls(**kwargs)

        # ---------------- bind to class ----------------
        cls.from_config = from_config
        cls.create = create
        cls.print_expected_args = classmethod(_print_expected_args)
        if buildable:
            # Plain dataclass factories have no positional context args.
            cls.build = lambda self, **context: _build_instance(self, (), context)
        return cls

    return decorator


# ============================================================
# 3) Print utilities
# ============================================================

def _print_expected_args(cls, *, indent=0, visited=None):
    """
    Recursively print "expected arguments" for a factory dataclass.

    This is aimed at introspection/debugging for config-driven systems.

    Output includes:
      - Context args (from `_context_arg_names`, if present)
      - Field names, annotated types, and defaults
      - For registry-enabled nested fields, it enumerates registered concrete variants
        and prints their expected args recursively.
      - For nested dataclass factories, it recurses to print their expected args.

    Parameters
    ----------
    cls:
        A factory dataclass type.
    indent:
        Left padding in spaces for nested output.
    visited:
        Cycle-breaker for recursive structures.
    """
    if visited is None:
        visited = set()
    if cls in visited:
        return
    visited.add(cls)

    prefix = " " * indent
    print(f"{prefix}Expected arguments for {cls.__name__}:")
    context_args = set(getattr(cls, "_context_arg_names", ()))

    print(f"{prefix}  Context Args: {context_args}")
    for f in fields(cls):
        typ = f.type
        default = (
            f.default if f.default is not MISSING
            else (f.default_factory() if f.default_factory is not MISSING else "<required>")
        )
        mark = " [context]" if f.name in context_args else ""
        print(f"{prefix}  {f.name}: {typ} = {default}{mark}")

        # If the field type owns a registry, enumerate variants.
        reg = getattr(typ, "_registry", None)
        owns_registry = isinstance(reg, dict) and ("_registry" in getattr(typ, "__dict__", {}))
        if owns_registry:
            for key, subcls in reg.items():
                print(f"{prefix}    - {key}:")
                subcls.print_expected_args(indent=indent + 6, visited=visited)

        # Otherwise, recurse into nested dataclass factories (if any).
        elif (
            is_dataclass(typ)
            and typ is not cls
            and hasattr(typ, "print_expected_args")
        ):
            typ.print_expected_args(indent=indent + 4, visited=visited)


def _print_class_signature(cls, *, indent=0, visited=None):
    """
    Print the __init__ signature for a class-backed factory.

    Class-backed factories are produced by `make_dataclass_from_callable` when `obj`
    is a class. In that mode, the implementation class receives a proxy `.build(...)`
    and `.from_config(...)`, but it is sometimes more useful to print the actual
    __init__ parameters rather than the generated factory dataclass fields.

    Context args are derived from `cls._factory_dataclass._context_arg_names`.
    """
    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)
    prefix = " " * indent
    print(f"{prefix}Expected arguments for {cls.__name__}:")
    ctx = set(getattr(cls, "_factory_dataclass", None)._context_arg_names or ())

    for name, p in sig.parameters.items():
        if name == "self":
            continue
        typ = hints.get(name, Any)
        default = p.default if p.default is not inspect.Parameter.empty else "<required>"
        mark = " [context]" if name in ctx else ""
        print(f"{prefix}  {name}: {typ} = {default}{mark}")


# ============================================================
# 4) make_dataclass_from_callable
# ============================================================

def make_dataclass_from_callable(base_cls_or_name, name=None, obj=None, *, type_key="type", suffix="ARGS", n_context_args=0):
    """
    Adapter that turns a function or class into a factory dataclass subclass.

    It generates a dataclass that:
      - Inherits from `base_cls` (which is typically decorated with `dataclass_factory`)
      - Registers itself in `base_cls` registry under `name` using `type_key`
      - Has fields corresponding to the callable's parameters *excluding*:
          * "self" (for class __init__)
          * the first `n_context_args` non-self parameters (treated as build-time context)

    Three usage patterns:
      - Direct: make_dataclass_from_callable(Base, "foo", func, n_context_args=1)
      - Decorator with base class: @make_dataclass_from_callable(Base, "foo", n_context_args=1)
      - Decorator standalone (no base class): @make_dataclass_from_callable("TileCoder", n_context_args=1)

    Parameters
    ----------
    base_cls_or_name:
        Either a registry base factory dataclass, or a string name for standalone mode.
        If a string is passed and `name` is None, creates a standalone factory dataclass
        without registry registration.
    name:
        Registry key to register this callable/class under. Required when `base_cls_or_name`
        is a class. Ignored when `base_cls_or_name` is a string (standalone mode).
    obj:
        Callable or class. If None, returns a decorator.
    type_key:
        Discriminant field name used by the base registry (default "type").
    suffix:
        Naming suffix for the generated dataclass.
    n_context_args:
        Number of leading non-self parameters treated as positional context args during build.
    """
    # Standalone mode: just a name string, no base class
    if isinstance(base_cls_or_name, str) and name is None:
        standalone_name = base_cls_or_name
        def decorator(func_or_cls):
            return _make_dataclass_standalone(standalone_name, func_or_cls, suffix, n_context_args)
        return decorator

    # Original behavior with base class
    base_cls = base_cls_or_name
    if obj is not None:
        return _make_dataclass_core(base_cls, name, obj, type_key, suffix, n_context_args)

    def decorator(func_or_cls):
        return _make_dataclass_core(base_cls, name, func_or_cls, type_key, suffix, n_context_args)

    return decorator


def _param_to_dc_field(pname: str, param: inspect.Parameter, hints: dict[str, Any]):
    """
    Convert a callable parameter into a `make_dataclass` field tuple.

    Rules:
      - Missing annotation -> Any
      - No default -> required dataclass field (no default set)
      - dict/list defaults -> use default_factory to avoid shared mutable defaults
      - Everything else -> use the parameter default
    """
    typ = hints.get(pname, Any)

    if param.default is inspect.Parameter.empty:
        return (pname, typ)

    origin = typing.get_origin(typ)
    if origin is dict:
        return (pname, typ, field(default_factory=dict))
    if origin is list:
        return (pname, typ, field(default_factory=list))
    return (pname, typ, field(default=param.default))


def _make_dataclass_core(base_cls, name, obj, type_key, suffix, n_context_args):
    """
    Internal implementation for `make_dataclass_from_callable`.

    Produces a generated dataclass subclass with:
      - `_construct_fn` set for function-backed factories
      - `_impl_cls` set for class-backed factories
      - `_context_arg_names` capturing the first `n_context_args` parameter names
      - `.build(*args, **context)` delegating to `_build_instance`

    Additionally:
      - In class-backed mode, the original class is patched with factory hooks:
          * from_config/create/build/print_expected_args
    """
    is_class = inspect.isclass(obj)
    func = obj.__init__ if is_class else obj
    impl_cls = obj if is_class else None

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # Build dataclass fields, skipping self and the first n_context_args non-self parameters.
    flds = []
    nonself_seen = 0
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if nonself_seen < n_context_args:
            nonself_seen += 1
            continue
        nonself_seen += 1
        flds.append(_param_to_dc_field(pname, param, hints))

    # Create a generated dataclass subclass of base_cls and register it.
    subclass = make_dataclass(f"_{base_cls.__name__}_{name}_{suffix}", flds, bases=(base_cls,))
    subclass.__init_subclass__(**{f"{type_key}_value": name})

    subclass._construct_fn = staticmethod(None if is_class else func)
    subclass._impl_cls = impl_cls
    subclass._n_context_args = n_context_args
    subclass._context_arg_names = tuple(
        [n for n in sig.parameters.keys() if n != "self"][:n_context_args]
    )

    # One build implementation for both function/class modes.
    def _build_for_subclass(self, *args, **context):
        return _build_instance(self, args, context)

    subclass.build = _build_for_subclass
    globals()[subclass.__name__] = subclass

    # --- Class Mode ---
    if is_class:
        # Attach factory metadata and proxy methods to the implementation class.
        obj._factory_dataclass = subclass
        obj._factory_name = name
        obj.from_config = classmethod(lambda cls_, cfg: subclass.from_config(cfg))
        obj.create = classmethod(lambda cls_, **kw: subclass.create(name, **kw))
        obj.build = subclass.build
        obj.print_expected_args = classmethod(_print_class_signature)

        # If the class is not itself a dataclass, provide a print_expected_args proxy.
        if not hasattr(obj, "__dataclass_fields__"):
            @classmethod
            def _proxy_print_expected_args(cls, **kw):
                return cls._factory_dataclass.print_expected_args(**kw)
            obj.print_expected_args = _proxy_print_expected_args

        return obj

    # --- Function Mode ---
    def proxy(*args, **kwargs):
        """
        Function wrapper that:
          - Accepts context args positionally (first n_context_args) or by keyword
          - Binds remaining args/kwargs to the original callable signature
          - Builds the generated dataclass instance and immediately calls `.build(...)`
        """
        arg_names = list(sig.parameters.keys())
        ctx_names = [n for i, n in enumerate(arg_names) if i < n_context_args and n != "self"]

        # accept context args either positionally or by keyword
        ctx_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in ctx_names}
        ctx_args, cfg_args = args[:n_context_args], args[n_context_args:]

        bound = sig.bind_partial(*cfg_args, **kwargs)
        bound.apply_defaults()

        inst = subclass(**{type_key: name, **bound.arguments})
        return inst.build(*ctx_args, **ctx_kwargs)

    proxy.print_expected_args = classmethod(_print_expected_args)
    proxy._factory_dataclass = subclass
    return proxy


def _make_dataclass_standalone(name, obj, suffix, n_context_args):
    """
    Internal implementation for standalone `make_dataclass_from_callable` (no base class).

    Produces a generated dataclass with:
      - `_construct_fn` set for function-backed factories
      - `_impl_cls` set for class-backed factories
      - `_context_arg_names` capturing the first `n_context_args` parameter names
      - `.build(*args, **context)` delegating to `_build_instance`
      - `.from_config(cfg)` for parsing config dicts
      - `.create(**kwargs)` for programmatic creation

    This mode does not register the dataclass in any registry.
    """
    is_class = inspect.isclass(obj)
    func = obj.__init__ if is_class else obj
    impl_cls = obj if is_class else None

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # Build dataclass fields, skipping self and the first n_context_args non-self parameters.
    flds = []
    nonself_seen = 0
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if nonself_seen < n_context_args:
            nonself_seen += 1
            continue
        nonself_seen += 1
        flds.append(_param_to_dc_field(pname, param, hints))

    # Create a standalone dataclass (no base class inheritance).
    dc_cls = make_dataclass(f"_{name}_{suffix}", flds)

    dc_cls._construct_fn = staticmethod(None if is_class else func)
    dc_cls._impl_cls = impl_cls
    dc_cls._n_context_args = n_context_args
    dc_cls._context_arg_names = tuple(
        [n for n in sig.parameters.keys() if n != "self"][:n_context_args]
    )

    # Build method for both function/class modes.
    def _build_for_dc(self, *args, **context):
        return _build_instance(self, args, context)

    dc_cls.build = _build_for_dc

    # from_config for standalone dataclass
    @classmethod
    def from_config(cls, cfg: dict):
        """
        Construct an instance from a python dict.
        """
        field_names = {f.name for f in fields(cls)}
        init_kwargs = {}
        for f in fields(cls):
            default = _default_for_dc_field(f)
            val = cfg.get(f.name, default)
            if val is MISSING:
                continue
            init_kwargs[f.name] = _coerce_value(f.type, val)
        return cls(**init_kwargs)

    # create for standalone dataclass
    @classmethod
    def create(cls, **kwargs):
        """
        Convenience constructor for programmatic creation.
        """
        return cls(**kwargs)

    dc_cls.from_config = from_config
    dc_cls.create = create
    dc_cls.print_expected_args = classmethod(_print_expected_args)

    globals()[dc_cls.__name__] = dc_cls

    # --- Class Mode ---
    if is_class:
        obj._factory_dataclass = dc_cls
        obj._factory_name = name
        obj.from_config = classmethod(lambda cls_, cfg: dc_cls.from_config(cfg))
        obj.create = classmethod(lambda cls_, **kw: dc_cls.create(**kw))
        obj.build = dc_cls.build
        obj.print_expected_args = classmethod(_print_class_signature)

        if not hasattr(obj, "__dataclass_fields__"):
            @classmethod
            def _proxy_print_expected_args(cls, **kw):
                return cls._factory_dataclass.print_expected_args(**kw)
            obj.print_expected_args = _proxy_print_expected_args

        return obj

    # --- Function Mode ---
    def proxy(*args, **kwargs):
        """
        Function wrapper that:
          - Accepts context args positionally (first n_context_args) or by keyword
          - Binds remaining args/kwargs to the original callable signature
          - Builds the generated dataclass instance and immediately calls `.build(...)`
        """
        arg_names = list(sig.parameters.keys())
        ctx_names = [n for i, n in enumerate(arg_names) if i < n_context_args and n != "self"]

        ctx_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in ctx_names}
        ctx_args, cfg_args = args[:n_context_args], args[n_context_args:]

        bound = sig.bind_partial(*cfg_args, **kwargs)
        bound.apply_defaults()

        inst = dc_cls(**bound.arguments)
        return inst.build(*ctx_args, **ctx_kwargs)

    proxy.print_expected_args = classmethod(_print_expected_args)
    proxy._factory_dataclass = dc_cls
    return proxy


# ============================================================
# 5) FixedSubclass
# ============================================================

def FixedSubclass(base_cls, key: str):
    """
    Create a "locked" subclass of a registry-enabled factory base.

    The returned class behaves like the registered subclass `base_cls._registry[key]`,
    but prevents callers from specifying a different registry key in either:
      - from_config(cfg)
      - create(**kw)

    Use case:
      - Expose a specific concrete variant as a stable public type without letting
        configs override the discriminant field.

    Parameters
    ----------
    base_cls:
        A registry-enabled base factory dataclass (created via dataclass_factory(key_name=...)).
    key:
        The registry key to lock to.

    Returns
    -------
    Locked:
        A dynamically created subclass of the target registered subclass that:
          - Overrides from_config/create to enforce the fixed key
          - Removes any inherited `_registry` attribute (so it does not act as a base registry)
          - Preserves print_expected_args for introspection
    """
    if not hasattr(base_cls, "_registry") or key not in base_cls._registry:
        raise ValueError(f"{key!r} not a valid subclass of {base_cls.__name__}")

    sub = base_cls._registry[key]
    key_name = getattr(base_cls, "_factory_key", "type")

    Locked = type(f"{sub.__name__}_LOCKED_{key}", (sub,), {})
    # Ensure the locked type does not itself own a registry.
    if "_registry" in Locked.__dict__:
        delattr(Locked, "_registry")

    Locked._locked_factory_key_name = key_name
    Locked._locked_factory_value = key

    @classmethod
    def from_config(cls, cfg):
        """
        Parse config dict, enforcing that cfg[key_name] (if present) matches the lock.
        """
        if key_name in cfg and cfg[key_name] != key:
            raise ValueError(f"{cls.__name__} locked to {key!r}, not {cfg[key_name]!r}")
        return sub.from_config({**cfg, key_name: key})

    @classmethod
    def create(cls, **kw):
        """
        Programmatic creation, enforcing that kw[key_name] (if present) matches the lock.
        """
        if key_name in kw and kw[key_name] != key:
            raise ValueError(f"{cls.__name__} locked to {key!r}, not {kw[key_name]!r}")
        kw[key_name] = key
        return sub.create(key, **kw)

    Locked.from_config = from_config
    Locked.create = create
    Locked.print_expected_args = classmethod(_print_expected_args)
    return Locked


# ============================================================
# 5. Auto register constructors
# ============================================================
def auto_register_constructors(base_cls, constructors: dict[str, Any]):
    for name, obj in constructors.items():
        args = (0,) if not isinstance(obj, tuple) else (obj[1],)
        make_dataclass_from_callable(base_cls, name, obj[0] if isinstance(obj, tuple) else obj, n_context_args=args[0])



def create_sweep_args(d):
    """
    Generate all combinations of parameter sweeps from a nested dictionary definition.

    This function interprets a configuration dictionary where each key maps to a list
    of values to sweep over. Keys that contain a "+" indicate that multiple parameters
    should be swept together — i.e., their values are grouped and varied jointly
    across corresponding indices.

    The output is a list of dictionaries, each representing one unique combination
    of arguments to be used in an experimental sweep or grid search.

    Parameters
    ----------
    d : dict
        Dictionary defining the parameter sweep. Each entry should be one of:

        - `key: list` — a standard sweep over all combinations of these values.
        - `"a+b": { "a": list, "b": list }` — a grouped sweep where parameters
          `a` and `b` are varied together elementwise.

        Example:
        >>> d = {
        ...     "lr": [0.01, 0.1],
        ...     "optimizer+momentum": {"optimizer": ["sgd", "adam"], "momentum": [0.9, 0.95]},
        ... }

    Returns
    -------
    list of dict
        Each dictionary contains one complete configuration of parameters
        for a single experiment or run. For the example above:

        >>> create_sweep_args(d)
        [
            {"lr": 0.01, "optimizer": "sgd", "momentum": 0.9},
            {"lr": 0.01, "optimizer": "adam", "momentum": 0.95},
            {"lr": 0.1, "optimizer": "sgd", "momentum": 0.9},
            {"lr": 0.1, "optimizer": "adam", "momentum": 0.95},
        ]

    Notes
    -----
    - The grouped keys (with "+") ensure elementwise pairing across lists, not Cartesian products.
    - All lists used for grouped parameters must be of the same length.
    - The function expands grouped keys back into individual arguments in the output.

    See Also
    --------
    itertools.product : Used internally to compute the Cartesian product.
    """"""Takes a diction of args to sweep over and creates the sweep.
    """
    args_to_sweep = {}
    for k in d:
        if "+" in k:
            inner_keys = list(d[k].keys())
            n_values = len(d[k][inner_keys[0]])
            args_to_sweep["+".join(inner_keys)] = \
                [tuple(d[k][ik][i] for ik in inner_keys) for i in range(n_values)]
        else:
            args_to_sweep[k] = d[k]

    _sweep_args = [{k: v[idx] for idx, k in enumerate(args_to_sweep.keys())}
                   for v in itertools.product(*args_to_sweep.values())]

    def parse_stuff(md):
        ret_args = {}
        for k in md:
            if "+" in k:
                inner_keys = k.split("+")
                for i, ik in enumerate(inner_keys):
                    ret_args[ik] = md[k][i]
            else:
                ret_args[k] = md[k]
        return ret_args
    return [parse_stuff(md) for md in _sweep_args]


def sweep_length(cfg_file):
    """Parse a YAML config file and determine the length.
    """
    with open(cfg_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    if "sweep" not in cfg_dict:
        return 1
    return len(create_sweep_args(cfg_dict["sweep"]))


def set_value_from_name(dict, name, value):
    """
    Set a value in a nested dictionary using a dotted key name.

    This function allows setting values deep inside a nested dictionary
    by providing a dotted key path (e.g., `"a.b.c"`). Intermediate
    dictionaries are automatically created if they do not exist.

    Args:
        dict (dict): The dictionary to modify.
        name (str): The key name or dotted key path specifying where to set the value.
            For example, `"a.b.c"` sets `dict["a"]["b"]["c"] = value`.
        value (Any): The value to assign at the specified key path.

    Example:
        >>> d = {}
        >>> set_value_from_name(d, "a.b.c", 42)
        >>> print(d)
        {'a': {'b': {'c': 42}}}

    """
    if "." in name:
        keys = name.split(".")
        d = dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    else:
        dict[name] = value


@dataclass
class SetupArgs():
    experiment: str
    githash: str | None = None


def parse_config(cfg_file, id, EXPERIMENTS):
    """
    Parse a YAML configuration file and initialize a configuration object.

    This function loads a YAML configuration file, expands parameter sweeps,
    and returns a constructed argument object for the specified sweep index.

    Parameters
    ----------
    cfg_file : str
        Path to the YAML configuration file containing base and sweep definitions.
    id : int
        Index of the parameter combination to select from the sweep.
    EXPERIMENTS: dict
        A dictionary containing available experiments.

    Returns
    -------
    Any
        Configuration object initialized with parameters corresponding to `id`.

    Notes
    -----
    - The YAML file should define a top-level key `"sweep"`.
    - Each sweep entry is processed via `create_sweep_args`.
    """
    with open(cfg_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    setup_args = SetupArgs(**cfg_dict["setup"])

    if setup_args.experiment not in EXPERIMENTS:
        raise ValueError(f"{setup_args.experiment} not available. "
                         f"Available Experiments are {list(EXPERIMENTS.keys())}")

    new_cfg_dict = deepcopy(cfg_dict)
    del new_cfg_dict["setup"]
    exp_ns = EXPERIMENTS[setup_args.experiment]

    if hasattr(exp_ns, "get_args_class"):
        return parse_config_dict(new_cfg_dict, id, exp_ns.get_args_class()), setup_args, exp_ns
    else:
        return parse_config_dict(new_cfg_dict, id), setup_args, exp_ns


def parse_config_dict(cfg_dict, id, arg_class=None):
    if "sweep" in cfg_dict:
        _sweep_args = create_sweep_args(cfg_dict["sweep"])
    else:
        _sweep_args = None
    args_dict = {k: v for k, v in cfg_dict.items()
                 if k not in ("sweep")}
    if _sweep_args is not None:
        for k, v in _sweep_args[id].items():
            set_value_from_name(args_dict, k, v)
    if arg_class is not None:
        args = arg_class.from_config(args_dict)
    else:
        args = args_dict
    return args


import hashlib


def get_dirs(exp_args, setup_args, save_dir, base_dir="", post_args=[]):
    base_dir = os.path.abspath(
        os.path.join(
            base_dir,
            save_dir))
    hasher = hashlib.sha1()
    args_dict = asdict(exp_args)
    for k in post_args:
        del args_dict[k]
    hasher.update(str(args_dict).encode())
    a_id = hasher.hexdigest()

    _save_dir = os.path.join(base_dir, a_id)

    args_dict = asdict(exp_args)
    exp_save_dir = os.path.join(
        _save_dir, *[f"{k}-{args_dict[k]}" for k in post_args])

    return exp_save_dir


def get_run_dir(exp_args, setup_args, save_dir, base_dir="", post_args=[]):
    base_dir = os.path.abspath(
        os.path.join(
            base_dir,
            save_dir,
            "runs"))
    hasher = hashlib.sha1()
    args_dict = asdict(exp_args)
    for k in post_args:
        del args_dict[k]
    hasher.update(str(args_dict).encode())
    a_id = hasher.hexdigest()

    _save_dir = os.path.join(base_dir, a_id)

    args_dict = asdict(exp_args)
    exp_save_dir = os.path.join(
        _save_dir, *[f"{k}-{args_dict[k]}" for k in post_args])

    return exp_save_dir


def setup_run_dir(exp_args, setup_args, save_dir, base_dir="", post_args=[]):
    # base directory:
    base_dir = os.path.abspath(
        os.path.join(
            base_dir,
            save_dir,
            "runs"))

    hasher = hashlib.sha1()
    args_dict = asdict(exp_args)
    for k in post_args:
        del args_dict[k]
    hasher.update(str(args_dict).encode())
    a_id = hasher.hexdigest()

    _save_dir = os.path.join(base_dir, a_id)
    os.makedirs(_save_dir, exist_ok=True)

    args_dict = asdict(exp_args)
    exp_save_dir = os.path.join(
        _save_dir, *[f"{k}-{args_dict[k]}" for k in post_args])
    config_path = os.path.join(exp_save_dir, "config.yaml")

    os.makedirs(exp_save_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        d = asdict(exp_args)
        d["setup"] = asdict(setup_args)
        d["setup"]["githash"] = os.popen("git rev-parse HEAD").read().strip()
        yaml.dump(d, f)

    return exp_save_dir


def store_exp_details(config, save_dir, base_dir):

    with open(config, 'r') as f:
        config_str = f.read()

    base_dir = os.path.abspath(
        os.path.join(
            base_dir,
            save_dir,
            "exp_configs"))

    if os.path.isdir(base_dir):
        config_files = os.listdir(base_dir)
    else:
        os.makedirs(base_dir, exist_ok=True)
        config_files = []

    exists_in_folder = False
    for fn in config_files:
        with open(os.path.join(base_dir, fn), 'r') as f:
            compare_str = f.read()
        exists_in_folder = compare_str == config_str
        if exists_in_folder:
            break
    if not exists_in_folder:
        idx = len(config_files)
        shutil.copy2(config, os.path.join(
            base_dir, f"config_{idx}{os.path.splitext(config)[1]}"))

