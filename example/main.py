from reproduce.config import (
    dataclass_factory,
    make_dataclass_from_callable,
    make_dataclass,
    parse_config,
    sweep_length
)
from reproduce.save_utils import setup_run_dir, get_run_dir, store_exp_details
from dataclasses import dataclass, asdict
import tyro
import yaml
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import optax
import torch
import torchvision


@dataclass_factory("type")
class Network:
    type: str

class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


# Register CNN as a Network type
make_dataclass_from_callable(Network, "cnn", CNN, n_context_args=1)


@dataclass_factory("type")
class Optimizer:
    type: str


# Register optax optimizers
make_dataclass_from_callable(Optimizer, "adam", optax.adam)
make_dataclass_from_callable(Optimizer, "sgd", optax.sgd)


@dataclass_factory(key_name=None, buildable=False)
class Args:
    network: Network
    optimizer: Optimizer
    num_steps: int
    print_every: int = 100
    batch_size: int = 64


def loss(model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]) -> Float[Array, ""]:
    """Cross-entropy loss for MNIST classification."""
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


def get_mnist_data(batch_size: int):
    """Load MNIST train and test data."""
    normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=normalize
    )
    test_dataset = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=normalize
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return trainloader, testloader


def evaluate(model: CNN, testloader: torch.utils.data.DataLoader):
    """Evaluate model on test set."""
    losses = []
    accuracies = []

    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        pred_y = jax.vmap(model)(x)
        loss_val = -jnp.mean(jax.nn.one_hot(y, 10) * pred_y)
        accuracy = jnp.mean(jnp.argmax(pred_y, axis=1) == y)
        losses.append(loss_val)
        accuracies.append(accuracy)

    return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))


def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model
    
def run_experiment(args: Args, save_dir: str):
    """
    Run a single experiment with the given args and save results to save_dir.
    """
    print("="*60)
    print("Building objects from config")
    print("="*60)

    # Get random key for initialization
    key = jax.random.PRNGKey(42)

    # Build network - pass key as context arg (n_context_args=1)
    model = args.network.build(key)
    print(f"Built model: {type(model).__name__}")

    # Build optimizer - no context args needed
    optim = args.optimizer.build()
    print(f"Built optimizer: {type(optim).__name__}")
    print()

    # ============================================================
    # Train the model
    # ============================================================
    print("="*60)
    print("Training the model")
    print("="*60)

    # Load data
    print("Loading MNIST data...")
    trainloader, testloader = get_mnist_data(args.batch_size)

    # Train
    print(f"Training for {args.num_steps} steps...")
    trained_model = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optim=optim,
        steps=args.num_steps,
        print_every=args.print_every
    )

    # ============================================================
    # Save results
    # ============================================================
    print("="*60)
    print("Saving results")
    print("="*60)

    # Evaluate final performance
    final_loss, final_accuracy = evaluate(trained_model, testloader)

    # Save results to YAML
    results = {
        "final_test_loss": float(final_loss),
        "final_test_accuracy": float(final_accuracy),
    }

    results_path = f"{save_dir}/results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

    print(f"Results saved to: {results_path}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    print(f"Final test loss: {final_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    @dataclass
    class RunArgs:
        """Run a single experiment from a config file with parameter sweeps."""
        config: str
        """Path to the YAML configuration file."""
        id: int
        """Sweep ID (0-indexed) to run from the configuration."""
        exp_name: str = "mnist_sweep"
        """Name for the experiment directory."""
        base_save_dir: str = "./"
        """Base directory for saving results."""
        sweep_length: bool = False

    # Parse command line args with tyro
    run_args = tyro.cli(RunArgs)

    # Define experiment module
    class ExpModule:
        @staticmethod
        def get_args_class():
            return Args

        main = staticmethod(run_experiment)

    EXPERIMENTS = {"mnist": ExpModule()}

    if run_args.sweep_length:
        print(sweep_length(run_args.config))
        exit(0)

    # Parse config file and get args for this sweep id
    exp_args, setup_args, exp_ns = parse_config(
        run_args.config, run_args.id, EXPERIMENTS)

    # Setup run directory
    exp_save_dir = setup_run_dir(
        exp_args=exp_args,
        setup_args=setup_args,
        save_dir=run_args.exp_name,
        base_dir=run_args.base_save_dir
    )

    # Store experiment config (only for first run)
    if run_args.id == 0:
        store_exp_details(
            config=run_args.config,
            save_dir=run_args.exp_name,
            base_dir=run_args.base_save_dir)

    # Run the experiment
    exp_ns.main(
        args=exp_args,
        save_dir=exp_save_dir
    )
