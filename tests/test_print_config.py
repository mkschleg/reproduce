
from reproduce.config import (
    dataclass_factory,
    make_dataclass_from_callable,
    FixedSubclass
)


@dataclass_factory("type")
class Network:
    type: str

@dataclass_factory("type")
class Optimizer:
    type: str

@make_dataclass_from_callable(Optimizer, "SGD", n_context_args=1)
class SGD:
    def __init__(self, x, alpha, momentum=0.0): self.alpha, self.momentum = alpha, momentum
    def __repr__(self): return f"SGD(alpha={self.alpha}, momentum={self.momentum})"

@make_dataclass_from_callable(Optimizer, "ADAM", n_context_args=1)
class ADAM:
    def __init__(self, x, alpha, momentum=0.0): self.alpha, self.momentum = alpha, momentum
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
    opt: FixedSubclass(Optimizer, "SGD")
    steps: int



# Trainer.print_expected_args()
