from reproduce.config import (
    dataclass_factory,
    make_dataclass_from_callable,
    FixedSubclass
)
import pytest


# ============================================================
# 1. Basic Factory Behavior
# ============================================================
@dataclass_factory("type")
class MyBase:
    type: str


@make_dataclass_from_callable(MyBase, "adder")
def build_adder(x: int, y: int):
    return x + y


@make_dataclass_from_callable(MyBase, "concat")
def build_concat(a: str, b: str):
    return a + b


@make_dataclass_from_callable(MyBase, "mytype")
class MyType:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f"MyType(a={self.a}, b={self.b})"


def test_from_config_dispatch():
    cfg = {"type": "adder", "x": 3, "y": 5}
    obj = MyBase.from_config(cfg)
    assert obj.type == "adder"
    assert obj.build() == 8


def test_create_shortcut_and_direct_build():
    obj = MyBase.create("concat", a="foo", b="bar")
    assert obj.build() == "foobar"


def test_direct_build():
    obj = build_concat(a="foo", b="bar")
    assert obj == "foobar"


def test_class_from_config():
    cfg = {"type": "mytype", "a": "foo", "b": "bar"}
    obj = MyBase.from_config(cfg).build()
    assert isinstance(obj, MyType)
    assert str(obj) == "MyType(a=foo, b=bar)"


def test_class_create_dataclass():
    cfg = {"type": "mytype", "a": "foo", "b": "bar"}
    obj = MyBase.create("mytype", a="foo", b="bar").build()
    assert isinstance(obj, MyType)
    assert str(obj) == "MyType(a=foo, b=bar)"


def test_class_direct_build():
    obj = MyType(a="foo", b="bar")
    assert isinstance(obj, MyType)
    assert str(obj) == "MyType(a=foo, b=bar)"


def test_strict_key_validation():
    cfg = {"type": "adder", "x": 1, "y": 2, "oops": 99}
    with pytest.raises(TypeError):
        MyBase.from_config(cfg)


# ============================================================
# 2. Nested Factories
# ============================================================
@dataclass_factory("type")
class Network:
    type: str


@dataclass_factory("type")
class Optimizer:
    type: str


@make_dataclass_from_callable(Optimizer, "SGD")
class SGD:
    def __init__(self, alpha, momentum=0.0):
        self.alpha = alpha
        self.momentum = momentum

    def __repr__(self):
        return f"SGD(alpha={self.alpha}, momentum={self.momentum})"


@make_dataclass_from_callable(Network, "MLP", n_context_args=2)
def construct_network_MLP(num_in: int, num_out: int, layers: int, hidden: int):
    return f"MLP(in={num_in}, out={num_out}, layers={layers}, hidden={hidden})"


@make_dataclass_from_callable(Network, "CNN")
def construct_network_CNN(layers: int, hidden: int):
    return f"CNN(layers={layers}, hidden={hidden})"


@dataclass_factory()
class Trainer:
    policy_net: Network
    value_net: Network
    opt: Optimizer
    steps: int


def test_nested_factory_and_build_context():
    cfg = {
        "policy_net": {"type": "MLP", "layers": 3, "hidden": 128},
        "value_net": {"type": "CNN", "layers": 2, "hidden": 64},
        "opt": {"type": "SGD", "alpha": 0.1},
        "steps": 1000,
    }

    trainer_cfg = Trainer.from_config(cfg)
    trainer = trainer_cfg.build(
        policy_net={"num_in": 8, "num_out": 4},
    )

    assert "MLP(in=8, out=4, layers=3, hidden=128)" == trainer.policy_net
    assert "CNN(layers=2, hidden=64)" == trainer.value_net
    assert isinstance(trainer.opt, SGD)
    assert trainer.steps == 1000


# ============================================================
# 3. FixedSubclass Behavior
# ============================================================
def test_fixed_subclass_behavior():
    LockedSGD = FixedSubclass(Optimizer, "SGD")
    cfg = {"alpha": 0.1}

    # should succeed
    opt = LockedSGD.from_config(cfg).build()
    assert isinstance(opt, SGD)
    assert opt.alpha == 0.1
    assert opt.momentum == 0.0    

    # wrong type should raise
    bad_cfg = {"type": "NotSGD", "alpha": 0.1}
    with pytest.raises(ValueError):
        LockedSGD.from_config(bad_cfg)


# ============================================================
# 4. Deeply Nested Factories
# ============================================================
@dataclass_factory("type")
class InnerBase:
    type: str


@make_dataclass_from_callable(InnerBase, "inner", n_context_args=1)
def construct_inner(x: int):
    return f"Inner(x={x})"


@make_dataclass_from_callable(MyBase, "outer")
def construct_outer(inner: InnerBase, y: int):
    return f"Outer(inner={inner}, y={y})"


def test_nested_build_propagation_specific():
    cfg = {"type": "outer", "inner": {"type": "inner"}, "y": 42}
    outer_cfg = MyBase.from_config(cfg)
    result = outer_cfg.build(inner={"x": 5})
    assert "Outer(inner=Inner(x=5), y=42)"    


def test_nested_build_propagation_generic():
    cfg = {"type": "outer", "inner": {"type": "inner"}, "y": 42}
    outer_cfg = MyBase.from_config(cfg)
    result = outer_cfg.build(x=5)
    assert "Outer(inner=Inner(x=5), y=42)"


# ============================================================
# 5. Standalone Factory (no base class)
# ============================================================
@make_dataclass_from_callable("TileCoder")
def get_tilecoder(n_tiles: int, n_tilings: int):
    return f"TileCoder(n_tiles={n_tiles}, n_tilings={n_tilings})"


@make_dataclass_from_callable("StandaloneEncoder")
class StandaloneEncoder:
    def __init__(self, input_size: int, hidden_size: int = 64):
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __repr__(self):
        return f"StandaloneEncoder(input={self.input_size}, hidden={self.hidden_size})"


@make_dataclass_from_callable("ContextBuilder", n_context_args=1)
def build_with_context(rng: int, size: int, name: str = "default"):
    return f"Built(rng={rng}, size={size}, name={name})"


def test_standalone_function_direct_call():
    # Direct call now builds immediately (no .build() needed)
    result = get_tilecoder(n_tiles=8, n_tilings=4)
    assert result == "TileCoder(n_tiles=8, n_tilings=4)"


def test_standalone_function_from_config():
    # get_tilecoder IS the dataclass now (not a proxy with _factory_dataclass)
    cfg = {"n_tiles": 16, "n_tilings": 8}
    obj = get_tilecoder.from_config(cfg)
    assert obj.n_tiles == 16
    assert obj.n_tilings == 8
    assert obj.build() == "TileCoder(n_tiles=16, n_tilings=8)"


def test_standalone_function_create():
    # get_tilecoder IS the dataclass now
    obj = get_tilecoder.create(n_tiles=32, n_tilings=16)
    assert obj.build() == "TileCoder(n_tiles=32, n_tilings=16)"


def test_standalone_class_direct_instantiation():
    enc = StandaloneEncoder(input_size=10, hidden_size=32)
    assert str(enc) == "StandaloneEncoder(input=10, hidden=32)"


def test_standalone_class_from_config():
    cfg = {"input_size": 20, "hidden_size": 128}
    enc = StandaloneEncoder.from_config(cfg).build()
    assert isinstance(enc, StandaloneEncoder)
    assert str(enc) == "StandaloneEncoder(input=20, hidden=128)"


def test_standalone_class_create_with_default():
    enc = StandaloneEncoder.create(input_size=30).build()
    assert str(enc) == "StandaloneEncoder(input=30, hidden=64)"


def test_standalone_with_context_args_direct_call():
    # Direct call: first positional arg is context, rest are config
    result = build_with_context(42, size=10, name="test")
    assert result == "Built(rng=42, size=10, name=test)"


def test_standalone_with_context_args_create():
    # create() returns config instance, then .build(context) to construct
    result = build_with_context.create(size=10, name="test").build(42)
    assert result == "Built(rng=42, size=10, name=test)"


def test_standalone_with_context_args_from_config():
    # from_config() returns config instance, then .build(context) to construct
    cfg = {"size": 20, "name": "configured"}
    obj = build_with_context.from_config(cfg)
    result = obj.build(99)
    assert result == "Built(rng=99, size=20, name=configured)"


# ============================================================
# 6. Config Type Separation (config() marker)
# ============================================================
from reproduce.config import config
from dataclasses import fields


# Concrete "static" types (what exists after .build())
class MLP:
    def __init__(self, num_in: int, num_out: int, layers: int, hidden: int):
        self.num_in = num_in
        self.num_out = num_out
        self.layers = layers
        self.hidden = hidden

    def __repr__(self):
        return f"MLP({self.num_in}, {self.hidden}x{self.layers}, {self.num_out})"


class Adam:
    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def __repr__(self):
        return f"Adam(lr={self.lr})"


# Register implementations with registries
@make_dataclass_from_callable(Network, "mlp", n_context_args=2)
def build_mlp(num_in: int, num_out: int, layers: int = 3, hidden: int = 128) -> MLP:
    return MLP(num_in, num_out, layers, hidden)


@make_dataclass_from_callable(Optimizer, "adam")
def build_adam(lr: float = 0.001) -> Adam:
    return Adam(lr)


# Use config() to separate static type from config type
@make_dataclass_from_callable()
def TrainerConfig(
    network: MLP = config(Network),  # MLP is static type, Network is config type
    optimizer: Adam = config(Optimizer, default=None),  # optional
    batch_size: int = 32,
):
    """Build a trainer with network and optimizer."""
    return {"network": network, "optimizer": optimizer, "batch_size": batch_size}


def test_config_marker_field_types():
    """Verify that generated dataclass fields use config types, not static types."""
    field_types = {f.name: f.type for f in fields(TrainerConfig)}

    # network should be typed as Network (config type), not MLP
    assert field_types["network"] is Network
    # optimizer should be typed as Optimizer (config type), not Adam
    assert field_types["optimizer"] is Optimizer
    # batch_size should remain int (no config() marker)
    assert field_types["batch_size"] is int


def test_config_marker_from_config():
    """Verify that from_config works with config() marker fields."""
    cfg = {
        "network": {"type": "mlp", "layers": 4, "hidden": 256},
        "optimizer": {"type": "adam", "lr": 0.01},
        "batch_size": 64,
    }

    trainer_cfg = TrainerConfig.from_config(cfg)

    assert trainer_cfg.batch_size == 64
    assert hasattr(trainer_cfg.network, "build")
    assert hasattr(trainer_cfg.optimizer, "build")


def test_config_marker_build():
    """Verify that build works and produces correct static types."""
    cfg = {
        "network": {"type": "mlp", "layers": 4, "hidden": 256},
        "optimizer": {"type": "adam", "lr": 0.01},
        "batch_size": 64,
    }

    trainer_cfg = TrainerConfig.from_config(cfg)
    result = trainer_cfg.build(network={"num_in": 8, "num_out": 4})

    # After build, we should have the concrete static types
    assert isinstance(result["network"], MLP)
    assert isinstance(result["optimizer"], Adam)
    assert result["batch_size"] == 64


def test_config_marker_optional_field():
    """Verify that optional config() fields work with default=None."""
    cfg = {
        "network": {"type": "mlp", "layers": 2, "hidden": 64},
        # optimizer is omitted - should default to None
    }

    trainer_cfg = TrainerConfig.from_config(cfg)
    assert trainer_cfg.optimizer is None


def test_config_marker_inferred_name():
    """Verify that @make_dataclass_from_callable() infers name from function."""
    # TrainerConfig wrapper should have the inferred name
    assert TrainerConfig.__name__ == "TrainerConfig"
    assert hasattr(TrainerConfig, "from_config")
    assert hasattr(TrainerConfig, "create")
    # Direct call builds immediately, so no .build on the wrapper itself
    # but _config_cls has it
    assert hasattr(TrainerConfig._config_cls, "build")


def test_config_marker_direct_call_builds():
    """Verify that direct call Trainer(...) builds immediately."""
    # Create config instances for the nested fields
    network_cfg = Network.from_config({"type": "mlp", "layers": 2, "hidden": 64})
    optimizer_cfg = Optimizer.from_config({"type": "adam", "lr": 0.005})

    # Direct call should build immediately (no .build() needed)
    # Note: nested configs need to be pre-built or we pass built objects
    # For this test, we'll build them first
    built_network = network_cfg.build(4, 2)  # num_in=4, num_out=2
    built_optimizer = optimizer_cfg.build()

    result = TrainerConfig(
        network=built_network,
        optimizer=built_optimizer,
        batch_size=16
    )

    # Result should be the built dict, not a config instance
    assert isinstance(result, dict)
    assert result["batch_size"] == 16
    assert isinstance(result["network"], MLP)
    assert isinstance(result["optimizer"], Adam)


# ============================================================
# 7. Deferred build (config(..., defer=True))
# ============================================================
from reproduce.config import BoundBuilder


@dataclass_factory("type")
class SubAgent:
    type: str


# key/env are context args; n_actions/agent_i are also context args (supplied per
# deferred call); scale is a plain config field.
@make_dataclass_from_callable(SubAgent, "lin", n_context_args=4)
def make_lin(key, env, n_actions, agent_i, scale: float = 1.0):
    return f"lin(key={key},env={env},n={n_actions},i={agent_i},scale={scale})"


@dataclass_factory("type")
class Combo:
    type: str


@make_dataclass_from_callable(Combo, "decentralized", n_context_args=2)
def make_decentralized(key, env, sub: str = config(SubAgent, defer=True)):
    # `sub` is a BoundBuilder: key/env already bound, build one per action dim.
    return [sub(n_actions=d, agent_i=i) for i, d in enumerate([5, 3])]


@make_dataclass_from_callable(Combo, "centralized", n_context_args=2)
def make_centralized(key, env, sub: str = config(SubAgent, defer=True)):
    return sub(n_actions=8, agent_i=None)


def test_deferred_field_is_bound_builder():
    cfg = {"type": "decentralized", "sub": {"type": "lin", "scale": 2.0}}
    combo = Combo.from_config(cfg)
    # The constructor receives a BoundBuilder, not a built object.
    # Verify by inspecting the generated dataclass field metadata.
    from dataclasses import fields
    sub_field = next(f for f in fields(combo) if f.name == "sub")
    assert sub_field.metadata.get("reproduce_defer") is True


def test_deferred_build_fan_out():
    cfg = {"type": "decentralized", "sub": {"type": "lin", "scale": 2.0}}
    combo = Combo.from_config(cfg)
    out = combo.build("KEY", "ENV")
    assert out == [
        "lin(key=KEY,env=ENV,n=5,i=0,scale=2.0)",
        "lin(key=KEY,env=ENV,n=3,i=1,scale=2.0)",
    ]


def test_deferred_build_single_call():
    cfg = {"type": "centralized", "sub": {"type": "lin"}}  # scale defaults to 1.0
    combo = Combo.from_config(cfg)
    out = combo.build("K", "E")
    assert out == "lin(key=K,env=E,n=8,i=None,scale=1.0)"


def test_bound_builder_exposes_config_and_repr():
    # The BoundBuilder keeps the unbuilt config for introspection.
    sub_cfg = SubAgent.from_config({"type": "lin", "scale": 1.0})
    builder = BoundBuilder(sub_cfg, ("K", "E"), {"key": "K", "env": "E"})
    assert builder.config is sub_cfg
    assert "lin" in repr(builder).lower() or "SubAgent" in repr(builder)
    # Pure context extras (not config fields, not positional ctx) thread through.
    out = builder(n_actions=2, agent_i=0)
    assert out == "lin(key=K,env=E,n=2,i=0,scale=1.0)"


def test_deferred_optional_default_none():
    # A deferred field can be optional; None stays None (no builder, no build).
    @make_dataclass_from_callable(Combo, "maybe", n_context_args=2)
    def make_maybe(key, env, sub: str = config(SubAgent, default=None, defer=True)):
        return "none" if sub is None else sub(n_actions=1, agent_i=0)

    combo = Combo.from_config({"type": "maybe"})
    assert combo.build("K", "E") == "none"


def test_non_deferred_config_still_auto_builds():
    # Regression: config() without defer auto-builds as before.
    @make_dataclass_from_callable(Combo, "eager", n_context_args=4)
    def make_eager(key, env, n_actions, agent_i, sub: str = config(SubAgent)):
        return sub  # already built

    cfg = {"type": "eager", "sub": {"type": "lin", "scale": 3.0}}
    combo = Combo.from_config(cfg)
    out = combo.build("K", "E", 7, 1)
    assert out == "lin(key=K,env=E,n=7,i=1,scale=3.0)"

