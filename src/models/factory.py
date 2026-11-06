"""Model factory: a model is chosen by ``model.name`` in the config; other keys are its arguments.

Adding a model means adding one builder to ``BUILDERS``; training and evaluation code stay unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from monai.networks.nets import DynUNet
from torch import nn


def dynunet(spatial_dims: int, kernel_size: Sequence[Any], strides: Sequence[Any], **kwargs: Any) -> DynUNet:
    """MONAI ``DynUNet`` (nnU-Net style U-Net), 2-D or 3-D by ``spatial_dims``.

    The decoder upsamples by the encoder strides, so ``upsample_kernel_size`` is ``strides[1:]``.
    """
    return DynUNet(spatial_dims=spatial_dims, kernel_size=kernel_size, strides=strides,
                   upsample_kernel_size=strides[1:], **kwargs)


BUILDERS: dict[str, Callable[..., nn.Module]] = {"dynunet": dynunet}


def build_model(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Build the model named ``model_cfg["name"]`` with the remaining keys as arguments."""
    args = dict(model_cfg)
    name = args.pop("name")
    if name not in BUILDERS:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[name](**args)
