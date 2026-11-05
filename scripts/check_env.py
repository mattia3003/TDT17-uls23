"""Phase 0 environment check.

Prints library and GPU info, shows the top of the data tree, loads ONE image/label
VOI with MONAI, and runs ONE mixed-precision DynUNet training step. Deliberately
light: it reads a single VOI and never iterates over the dataset.

Examples:
    python scripts/check_env.py --machine configs/machines/home_pc.yaml --no-data
    python scripts/check_env.py --machine configs/machines/idun.yaml
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import cast

import monai
import numpy as np
import torch
import torch.version
from monai.data import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ResizeWithPadOrCrop
from monai.utils import convert_to_tensor
from torch.amp.grad_scaler import GradScaler

from src.data.discovery import describe_tree, find_first_pair
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", required=True, help="Machine config, e.g. configs/machines/idun.yaml.")
    parser.add_argument("--image", type=Path, help="Explicit image path (skips auto-discovery).")
    parser.add_argument("--label", type=Path, help="Explicit label path (used with --image).")
    parser.add_argument("--no-data", action="store_true", help="Skip data checks; use a random tensor.")
    parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 64], help="Crop size for the GPU step.")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def report_versions() -> None:
    print(f"python   {sys.version.split()[0]}  ({platform.platform()})")
    print(f"torch    {torch.__version__}  (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})")
    print(f"monai    {monai.__version__}")
    print(f"numpy    {np.__version__}")


def report_gpu() -> torch.device:
    """Print GPU info and return the device to use."""
    if not torch.cuda.is_available():
        print("GPU      CUDA not available")
        return torch.device("cpu")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    arch = f"sm_{props.major}{props.minor}"
    print(f"GPU      {props.name}, {props.total_memory / 1024**3:.1f} GB, {arch}, count={torch.cuda.device_count()}")
    if arch not in torch.cuda.get_arch_list():
        print(f"WARNING  {arch} not in this PyTorch build's arch list {torch.cuda.get_arch_list()}; "
              "a newer PyTorch wheel may be needed.")
    return device


def load_voi(image_path: Path, label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one image/label pair with MONAI and print its key properties."""
    keys = ["image", "label"]
    loader = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    start = time.perf_counter()
    sample = cast(dict[str, MetaTensor], loader({"image": str(image_path), "label": str(label_path)}))
    elapsed = time.perf_counter() - start

    image, label = sample["image"], sample["label"]
    spacing = affine_to_spacing(image.affine, r=3).tolist()
    voxel_mm3 = float(np.prod(spacing))
    label_values = torch.unique(label).tolist()
    foreground = int((label > 0).sum())
    pad_fraction = float((image == image.min()).float().mean())

    print(f"image    {image_path}")
    print(f"label    {label_path}")
    print(f"load     {elapsed:.2f} s")
    print(f"shape    image {tuple(image.shape)}, label {tuple(label.shape)}, dtype {image.dtype}")
    print(f"spacing  {[round(s, 3) for s in spacing]} mm")
    print(f"HU       min {float(image.min()):.0f}, max {float(image.max()):.0f}, "
          f"fraction at min (padding?) {pad_fraction:.3f}")
    print(f"label    values {label_values[:10]}{' ...' if len(label_values) > 10 else ''}, "
          f"foreground {foreground} voxels = {foreground * voxel_mm3:.0f} mm^3")
    return image.as_tensor(), label.as_tensor()


def gpu_step(device: torch.device, image: torch.Tensor, label: torch.Tensor, batch_size: int) -> None:
    """Run one AMP training step of a small 3D DynUNet and report time and peak memory."""
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        norm_name="instance",
    ).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    x = ((image - image.mean()) / (image.std() + 1e-6)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    y = (label > 0).float().unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)

    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if use_amp:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak = f", peak memory {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB" if use_amp else ""
    print(f"step     input {tuple(x.shape)}, loss {loss.item():.4f}, {elapsed:.2f} s (includes warm-up){peak}")


def main() -> None:
    args = parse_args()
    cfg = load_config([args.machine])
    print(f"machine  {cfg['machine']['name']}")
    report_versions()
    device = report_gpu()

    roi = tuple(args.roi)
    if args.no_data:
        image = torch.randn(1, *roi)
        label = (torch.rand(1, *roi) > 0.95).float()
    else:
        if args.image:
            pair = (args.image, args.label)
        else:
            data_root = Path(cfg["paths"]["data_root"])
            print(f"\nData tree (top levels) under {data_root}:")
            print("\n".join(describe_tree(data_root, max_depth=3)))
            pair = find_first_pair(data_root)
            if pair is None:
                sys.exit("\nNo images/ + labels/ pair found; share the tree above so the layout can be mapped.")
        print()
        image, label = load_voi(*pair)
        crop = ResizeWithPadOrCrop(roi)
        image = convert_to_tensor(crop(image), track_meta=False)
        label = convert_to_tensor(crop(label), track_meta=False)
    print()
    gpu_step(device, image, label, args.batch_size)
    print("\nOK: environment check finished.")


if __name__ == "__main__":
    main()
