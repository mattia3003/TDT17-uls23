"""Layered YAML config loading with dotted ``key=value`` overrides.

Configs are merged left to right (later files win), then command-line overrides
are applied, then environment variables (``$VAR`` / ``${VAR}``) and ``~`` in
string values are expanded. The result is a plain nested ``dict``.
"""

from __future__ import annotations

import argparse
import copy
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``update`` recursively merged into ``base``."""
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_override(cfg: dict[str, Any], override: str) -> None:
    """Apply one ``a.b.c=value`` override in place. ``value`` is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, raw_value = override.split("=", 1)
    *parents, leaf = key.strip().split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
    node[leaf] = yaml.safe_load(raw_value)


def expand_env(obj: Any) -> Any:
    """Recursively expand environment variables and ``~`` in string values."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, Mapping):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file that must contain a mapping (an empty file gives ``{}``)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {path} must be a mapping")
    return data


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Merge YAML files in order, apply overrides, then expand environment variables."""
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_merge(cfg, load_yaml(path))
    for override in overrides:
        apply_override(cfg, override)
    return expand_env(cfg)


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Write a resolved config to YAML, keeping key order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--config``, ``--machine`` and override arguments to a CLI."""
    parser.add_argument("--config", nargs="+", default=[], help="Experiment YAML files, merged in order.")
    parser.add_argument("--machine", required=True, help="Machine config, e.g. configs/machines/idun.yaml.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. train.epochs=5.")


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the config from parsed CLI args; the machine config is merged last."""
    return load_config([*args.config, args.machine], args.overrides)
