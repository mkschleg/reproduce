from dataclasses import (
    dataclass,
    fields,
    field,
    MISSING
)


# ============================================================
# Core Factory Decorator
# ============================================================
def dataclass_factory(key_name: str | None = None, *, strict: bool = True, buildable: bool = True):
    """Turns a dataclass into a registry-driven config factory."""
    def decorator(cls):
        cls = dataclass(cls)

        # ---------------- Registry ----------------
        if key_name:
            cls._factory_key = key_name
            cls._registry = {}

            @classmethod
            def __init_subclass__(sub_cls, **kwargs):
                key_value = kwargs.pop(f"{key_name}_value", None)
                super(cls, sub_cls).__init_subclass__(**kwargs)
                if key_value:
                    cls._registry[key_value] = sub_cls

            @classmethod
            def _dispatch(sub_cls, cfg_or_key):
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
            if key_name and hasattr(sub_cls, "_dispatch"):
                sub_cls = sub_cls._dispatch(cfg)

            # add back missing type key if necessary
            if key_name and key_name not in cfg:
                for k, v in getattr(cls, "_registry", {}).items():
                    if v is sub_cls:
                        cfg = {**cfg, key_name: k}
                        break

            field_names = {f.name for f in fields(sub_cls)}
            unexpected = set(cfg) - field_names
            if strict and unexpected:
                raise TypeError(f"Unexpected config keys: {sorted(unexpected)}")

            init_kwargs = {}
            for f in fields(sub_cls):
                val = cfg.get(f.name, f.default)
                if val is not None and hasattr(f.type, "from_config"):
                    val = f.type.from_config(val)
                init_kwargs[f.name] = val
            return sub_cls(**init_kwargs)

        # ---------------- create ----------------
        @classmethod
        def create(sub_cls, key_value=None, **kwargs):
            if key_name and hasattr(sub_cls, "_dispatch"):
                sub_cls = sub_cls._dispatch(key_value)
            if key_name and key_name not in kwargs:
                kwargs[key_name] = key_value
            return sub_cls(**kwargs)

        # ---------------- Bind to class ----------------
        cls.from_config = from_config
        cls.create = create
        if buildable:
            cls.build = _build_method
        return cls

    return decorator


# ============================================================
# Build utility
# ============================================================
def _build_method(self, **context):
    """Recursively build nested factory dataclasses."""
    built = {}
    for f in fields(self):
        val = getattr(self, f.name)
        subctx = {**context, **context.get(f.name, {})} if isinstance(context.get(f.name, {}), dict) else context
        if hasattr(val, "build"):
            val = val.build(**subctx)
        built[f.name] = val

    merged = {**context, **built}
    merged.pop(getattr(self, "_factory_key", None), None)

    if getattr(self, "_impl_cls", None):
        cls = self._impl_cls
        return cls(**_filter_args(cls.__init__, merged))
    if getattr(self, "_construct_fn", None):
        return self._construct_fn(**_filter_args(self._construct_fn, merged))
    return type(self)(**built)


def _filter_args(func, kwargs):
    import inspect
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters and k != "self"}
