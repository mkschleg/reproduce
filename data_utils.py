import numpy as np
import glob
import os
import yaml
import cloudpickle
import tqdm
import pandas as pd
from functools import partial
from collections import defaultdict


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dictionary with string keys.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): Internal key prefix used during recursion.
        sep (str, optional): Separator between nested keys. Defaults to '.'.

    Returns:
        dict: A flattened dictionary with concatenated keys.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """
    Reconstructs a nested dictionary from a flattened dictionary.

    Args:
        d (dict): Flattened dictionary (e.g. {"a.b.c": 1, "a.b.d": 2}).
        sep (str, optional): Separator used in flattened keys. Defaults to '.'.

    Returns:
        dict: A nested dictionary reconstructed from the flattened one.
    """
    result = {}
    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        current = result
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    return result


def to_python_type(obj):
    """
    Recursively converts NumPy scalar types and arrays to native Python types.
    """
    if isinstance(obj, np.generic):  # scalar (np.int64, np.float32, etc.)
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_python_type(v) for v in obj)
    else:
        return obj


def aggregate(func, x, *, include_final=False):
    used_x = x if include_final else x[:-1]
    return func(used_x)


def agg_last_n(arr: np.ndarray, *, n: int, agg=np.mean) -> float:
    """
    Compute the average of the last `n` elements in a NumPy array.

    Args:
        arr (np.ndarray): Input array of numeric values.
        n (int): Number of final elements to include in the average.

    Returns:
        float: The mean of the last `n` elements.

    Raises:
        ValueError: If `n` is not positive or greater than the array length.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n > arr.size:
        float(agg(arr))
    #     raise ValueError(f"n ({n}) cannot exceed array length ({arr.size}).")

    return float(agg(arr[-n:]))


def build_aggregate_func(name: str, **kwargs):
    if name == "agg_last_n":
        return partial(agg_last_n, **kwargs)
    else:
        raise ValueError("Agg Function not Found")


def prep_include_last(row):
    if row["last_episode_finished"]:
        return "returns", row["returns"]
    else:
        return "returns", row["returns"][:-1]


def create_summary(agg_funcs, row):
    res = {}
    for k in agg_funcs:
        x_name_or_func, agg_func = agg_funcs[k]
        if isinstance(x_name_or_func, str):
            x = row[x_name_or_func]
            x_name = x_name_or_func
        else:
            x_name, x = x_name_or_func(row)
        res[f"{x_name}:{k}"] = agg_func(x)
    return res


def load_exp_dir(exp_dir: str, agg_funcs: dict, *, wildcard="*", include_res=False, progbar=False):
    if "runs" in os.listdir(exp_dir):
        exp_dir = os.path.join(exp_dir, "runs")
    files = glob.glob(os.path.join(exp_dir, wildcard, "config.yaml"))
    specs = []
    if progbar:
        l_files = tqdm.tqdm(files)
    else:
        l_files = files
    for fn in l_files:
        with open(fn, 'r') as f:
            d = flatten_dict(yaml.safe_load(f))
        r_file = os.path.join(os.path.dirname(fn), "results.pkl")
        with open(r_file, 'rb') as f:
            res = cloudpickle.load(f)
        d["config_file"] = fn

        summary = create_summary(agg_funcs, res)
        if include_res:
            summary = {**summary, **res}
        specs.append({**d, **summary})
    df = pd.DataFrame(specs)
    for k in df.keys():
        if isinstance(df[k].iloc[0], list):
            df[k] = df[k].apply(tuple)
    return df


def get_group_by(df, *, ignore_githash=False):
    result_cols = [c for c in df.columns if "returns" in c or ":" in c or c.startswith("metric_")]
    ignore_cols = ["seed", "config_file"]
    if ignore_githash:
        ignore_cols.append("setup.githash")
    hyperparam_cols = [c for c in df.columns if c not in result_cols + ignore_cols ]
    valid_hparams = [
        c for c in hyperparam_cols
        if df[c].notna().any() and df[c].nunique() > 0
    ]
    agg_df = (
        df.groupby(valid_hparams, as_index=False)
            .agg({col: list for col in result_cols + ["seed", "config_file"]})
    )

    return agg_df


def get_best_hypers(df, *, sort_key: str = "returns:avg_end_mean", ascending=False, best_over=[]):
    result_cols = [c for c in df.columns if "returns" in c or ":" in c or c.startswith("metric_")]
    hyperparam_cols = [c for c in df.columns if c not in result_cols + ["seed", "config_file"]]
    valid_hparams = [
        c for c in hyperparam_cols
        if df[c].notna().any() and df[c].astype(str).nunique() > 0
    ]
    agg_df = (
        df.groupby(valid_hparams, as_index=False, dropna=False)
            .agg({col: list for col in result_cols + ["seed", "config_file"]})
    )
    for k in result_cols:
        agg_df[f"{k}_mean"] = agg_df[k].map(np.mean)
        agg_df[f"{k}_std"] = agg_df[k].map(np.std)


    return agg_df.loc[agg_df.groupby(best_over)[sort_key].idxmax()]


def create_params_for_final_sweep(df, save_file=None):
    result_cols = [c for c in df.columns if "returns" in c or ":" in c or c.startswith("metric_")]
    hyperparam_cols = [c for c in df.columns if c not in result_cols + ["seed", "config_file"]]
    sweep_keys = []
    base_keys = []
    for column in hyperparam_cols:
        # print(column)
        if df[column].nunique() == 1:
            base_keys.append(column)
        else:
            sweep_keys.append(column)
    base = {}
    sweep = defaultdict(lambda: [])
    for k in base_keys:
        base[k] = df[k].iloc[0]
    for row in df.iterrows():
        for k in sweep_keys:
            sweep[k].append(row[1][k])

    new_sweep = {"+params": sweep}
    sweep = to_python_type(new_sweep)
    base = to_python_type(unflatten_dict(base))

    if save_file is not None:
        with open(save_file, 'w') as f:
            yaml.dump({**base, "sweep": sweep}, f, default_flow_style=None, sort_keys=False)
    return base, sweep
