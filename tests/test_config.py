
from reproduce.config import (
    dataclass_factory,
    make_dataclass_from_callable,
    FixedSubclass
)
    

@dataclass_factory("type")
class MyBase:
    type: str


@make_dataclass_from_callable(MyBase, "adder")
def build_adder(x: int, y: int):
    return x + y


@make_dataclass_from_callable(MyBase, "concat")
def build_concat(a: str, b: str):
    return a + b


cfg = {"type": "adder", "x": 3, "y": 5}
obj1 = MyBase.from_config(cfg)
print(obj1.build())   # 8

obj2 = build_concat(a="foo", b="bar")
print(obj2)   # "foobar"


@dataclass_factory("type")
class Network:
    type: str


@dataclass_factory("type")
class Optimizer:
    type: str


@make_dataclass_from_callable(Optimizer, "SGD")
class SGD:
    def __init__(self, alpha, momentum=0.0): self.alpha, self.momentum = alpha, momentum
    def __repr__(self): return f"SGD(alpha={self.alpha}, momentum={self.momentum})"


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


print("\n--- Building TrainerConfig from dict ---")
cfg = {
    "policy_net": {"type": "MLP", "layers": 3, "hidden": 128},
    "value_net": {"type": "MLP", "layers": 2, "hidden": 64},
    "opt": {"type": "SGD", "alpha": 0.1},
    "steps": 1000,
}
trainer_cfg = Trainer.from_config(cfg)
print(trainer_cfg)

print("\n--- Building runtime Trainer with context ---")
trainer = trainer_cfg.build(
    policy_net={"num_in": 8, "num_out": 4},
    value_net={"num_in": 8, "num_out": 1},
)
print(trainer)

# ============================================================
# 7. Nested Example
# ============================================================
@dataclass_factory("type")
class MyBase: type: str

@dataclass_factory("type")
class InnerBase: type: str

@make_dataclass_from_callable(InnerBase, "another_class", n_context_args=1)
def construct_another_class(x: int): return f"Inner(x={x})"

@make_dataclass_from_callable(MyBase, "main_class")
def construct_main_class(inner: InnerBase, y: int): return f"Main(inner={inner}, y={y})"

cfg = {
    "type": "main_class",
    "inner": {"type": "another_class"},
    "y": 9,
}
cfg_obj = MyBase.from_config(cfg)
print(cfg_obj)
print(cfg_obj.build(
    inner=
    {"x": 3}))

# ============================================================
# 8. Complex Nested Config
# ============================================================
@dataclass_factory()
class ARGS:
    MB: MyBase
    policy_net: Network
    value_net: Network
    opt: FixedSubclass(Optimizer, "SGD")
    steps: int = 1000
    runs: int = 1

cfg = {
    "MB": {"type": "main_class", "inner": {"type": "another_class"}, "y": 9},
    "policy_net": {"type": "MLP", "layers": 3, "hidden": 128},
    "value_net": {"type": "MLP", "layers": 2, "hidden": 64},
    "opt": {"alpha": 0.1},
    "runs": 100
}
cfg_obj = ARGS.from_config(cfg)
print("\n--- Building ARGS from dict ---")
print(cfg_obj)
print("\n--- Building sub component with context ---")
print(cfg_obj.MB.build(x=3))
print("\n--- Building runtime Trainer with context ---")
print(cfg_obj.build(
    x=3,
    num_in=8,
    policy_net={"num_out": 4},
    value_net={"num_out": 1}))


@dataclass_factory(buildable=False)
class ARGS2:
    MB: MyBase
    policy_net: Network
    value_net: Network
    opt: Optimizer
    steps: int = 1000

cfg = {
    "MB": {
        "type": "main_class",
        "inner": {"type": "another_class"},
        "y": 9},
    "policy_net": {"type": "MLP", "layers": 3, "hidden": 128},
    "value_net": {"type": "MLP", "layers": 2, "hidden": 64},
    "opt": {"type": "SGD", "alpha": 0.1},
}
cfg_obj = ARGS2.from_config(cfg)
print("\n--- Building ARGS from dict ---")
try:
    print(cfg_obj.build(
        x=3,
        num_in=8,
        policy_net={"num_out": 4},
        value_net={"num_out": 1}))
except Exception as e:
    print(e)



