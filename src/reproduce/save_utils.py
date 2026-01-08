import hashlib
import os
import yaml
import shutil
from dataclasses import (
    asdict
)


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
