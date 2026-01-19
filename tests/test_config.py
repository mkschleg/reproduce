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
    result = get_tilecoder(n_tiles=8, n_tilings=4)
    assert result == "TileCoder(n_tiles=8, n_tilings=4)"


def test_standalone_function_from_config():
    dc = get_tilecoder._factory_dataclass
    cfg = {"n_tiles": 16, "n_tilings": 8}
    obj = dc.from_config(cfg)
    assert obj.n_tiles == 16
    assert obj.n_tilings == 8
    assert obj.build() == "TileCoder(n_tiles=16, n_tilings=8)"


def test_standalone_function_create():
    dc = get_tilecoder._factory_dataclass
    obj = dc.create(n_tiles=32, n_tilings=16)
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


def test_standalone_with_context_args():
    result = build_with_context(42, size=10, name="test")
    assert result == "Built(rng=42, size=10, name=test)"


def test_standalone_with_context_args_from_config():
    dc = build_with_context._factory_dataclass
    cfg = {"size": 20, "name": "configured"}
    obj = dc.from_config(cfg)
    result = obj.build(99)
    assert result == "Built(rng=99, size=20, name=configured)"

