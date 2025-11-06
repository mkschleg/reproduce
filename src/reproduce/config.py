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
import inspect
from copy import deepcopy

# ============================================================
# 1. Core Factory Decorator
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
        cls.print_expected_args = classmethod(_print_expected_args)
        if buildable:
            cls.build = _build_method
        return cls

    return decorator


# ============================================================
# 2. Build / Print utilities
# ============================================================
def _filter_args(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters and k != "self"}


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


def _print_expected_args(cls, *, indent=0, visited=None):
    """Recursively display constructor args (dataclass + context-aware)."""
    if visited is None: visited = set()
    if cls in visited: return
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

        reg = getattr(typ, "_registry", None)
        owns_registry = isinstance(reg, dict) and ("_registry" in getattr(typ, "__dict__", {}))
        if owns_registry:
            for key, subcls in reg.items():
                print(f"{prefix}    - {key}:")
                subcls.print_expected_args(indent=indent + 6, visited=visited)
        elif (
                is_dataclass(typ)
                and typ is not cls
                and hasattr(typ, "print_expected_args")  # only recurse if it's a config-type dataclass
        ):
            typ.print_expected_args(indent=indent + 4, visited=visited)


def _print_class_signature(cls, *, indent=0, visited=None):
    """Print the __init__ signature for class-based factories."""
    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)
    prefix = " " * indent
    print(f"{prefix}Expected arguments for {cls.__name__}:")
    ctx = set(getattr(cls, "_factory_dataclass", None)._context_arg_names or ())

    for name, p in sig.parameters.items():
        if name == "self":
            continue
        typ = hints.get(name, Any)
        default = (
            p.default if p.default is not inspect.Parameter.empty else "<required>"
        )
        mark = " [context]" if name in ctx else ""
        print(f"{prefix}  {name}: {typ} = {default}{mark}")

# ============================================================
# 3. make_dataclass_from_callable
# ============================================================
def make_dataclass_from_callable(base_cls, name, obj=None, *, type_key="type", suffix="ARGS", n_context_args=0):
    if obj is not None:
        return _make_dataclass_core(base_cls, name, obj, type_key, suffix, n_context_args)
    def decorator(func_or_cls):
        return _make_dataclass_core(base_cls, name, func_or_cls, type_key, suffix, n_context_args)
    return decorator


def _make_dataclass_core(base_cls, name, obj, type_key, suffix, n_context_args):
    func = obj.__init__ if inspect.isclass(obj) else obj
    impl_cls = obj if inspect.isclass(obj) else None
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # fields
    flds = []
    i = 0
    for pname, param in sig.parameters.items():
        if pname == "self": continue
        if i < n_context_args:
            i += 1
            continue
        i += 1
        typ = hints.get(pname, Any)
        if param.default is inspect.Parameter.empty:
            flds.append((pname, typ))
        else:
            flds.append((pname, typ, field(default=param.default)))

    subclass = make_dataclass(f"_{base_cls.__name__}_{name}_{suffix}", flds, bases=(base_cls,))
    subclass.__init_subclass__(**{f"{type_key}_value": name})
    subclass._construct_fn = staticmethod(func if not inspect.isclass(obj) else None)
    subclass._impl_cls = impl_cls
    subclass._n_context_args = n_context_args
    subclass._context_arg_names = tuple(
        [n for n in sig.parameters.keys() if n != "self"][:n_context_args]
    )

    def _build_for_subclass(self, *args, **context):
        built = {}
        for f in fields(self):
            val = getattr(self, f.name)
            subctx = {**context, **context.get(f.name, {})} if isinstance(context.get(f.name, {}), dict) else context
            if hasattr(val, "build"):
                val = val.build(*args, **subctx)
            built[f.name] = val

        merged = {**context, **built}
        merged.pop(getattr(self, "_factory_key", None), None)

        if getattr(self, "_impl_cls", None):
            cls = self._impl_cls
            context_map = {**context, **{n: a for n, a in zip(self._context_arg_names, args)}}
            vals = [context_map[n] for n in self._context_arg_names if n in context_map]
            filtered = _filter_args(cls.__init__, {**context_map, **merged})
            for n in self._context_arg_names: filtered.pop(n, None)
            return cls(*vals, **filtered)

        if getattr(self, "_construct_fn", None):
            context_map = {**context, **{n: a for n, a in zip(self._context_arg_names, args)}}
            vals = [context_map[n] for n in self._context_arg_names if n in context_map]
            filtered = _filter_args(self._construct_fn, {**context_map, **merged})
            for n in self._context_arg_names: filtered.pop(n, None)
            return self._construct_fn(*vals, **filtered)
        return type(self)(**built)

    subclass.build = _build_for_subclass
    globals()[subclass.__name__] = subclass

    # --- Class Mode ---
    if inspect.isclass(obj):
        obj._factory_dataclass = subclass
        obj._factory_name = name
        obj.from_config = classmethod(lambda cls_, cfg: subclass.from_config(cfg))
        obj.create = classmethod(lambda cls_, **kw: subclass.create(name, **kw))
        obj.build = subclass.build
        obj.print_expected_args = classmethod(_print_class_signature)
        if not hasattr(obj, "__dataclass_fields__"):
            @classmethod
            def _proxy_print_expected_args(cls, **kw):
                return cls._factory_dataclass.print_expected_args(**kw)
            obj.print_expected_args = _proxy_print_expected_args

        return obj

    # --- Function Mode ---
    def proxy(*args, **kwargs):
        arg_names = list(sig.parameters.keys())
        ctx_names = [n for i, n in enumerate(arg_names) if i < n_context_args and n != "self"]
        ctx_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in ctx_names}
        ctx_args, cfg_args = args[:n_context_args], args[n_context_args:]
        bound = sig.bind_partial(*cfg_args, **kwargs)
        bound.apply_defaults()
        inst = subclass(**{type_key: name, **bound.arguments})
        return inst.build(*ctx_args, **ctx_kwargs)

    proxy.print_expected_args = classmethod(_print_expected_args)
    proxy._factory_dataclass = subclass
    return proxy


# ============================================================
# 4. FixedSubclass
# ============================================================
def FixedSubclass(base_cls, key: str):
    """Lock a factory subclass to a specific registered type."""
    if not hasattr(base_cls, "_registry") or key not in base_cls._registry:
        raise ValueError(f"{key!r} not a valid subclass of {base_cls.__name__}")

    sub = base_cls._registry[key]
    key_name = getattr(base_cls, "_factory_key", "type")
    Locked = type(f"{sub.__name__}_LOCKED_{key}", (sub,), {})
    if "_registry" in Locked.__dict__: delattr(Locked, "_registry")

    Locked._locked_factory_key_name = key_name
    Locked._locked_factory_value = key

    @classmethod
    def from_config(cls, cfg):
        if key_name in cfg and cfg[key_name] != key:
            raise ValueError(f"{cls.__name__} locked to {key!r}, not {cfg[key_name]!r}")
        return sub.from_config({**cfg, key_name: key})

    @classmethod
    def create(cls, **kw):
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


def setup_dirs(exp_args, setup_args, save_dir, base_dir="", post_args=[]):
    import hashlib

    # base directory:
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
    os.makedirs(_save_dir, exist_ok=True)

    args_dict = asdict(exp_args)
    exp_save_dir = os.path.join(
        _save_dir, *[f"{k}-{args_dict[k]}" for k in post_args])
    config_path = os.path.join(exp_save_dir, "config.yaml")

    os.makedirs(exp_save_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        d = asdict(exp_args)
        d["setup"] = asdict(setup_args)
        yaml.dump(d, f)

    return exp_save_dir


