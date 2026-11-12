"""Tests for the transform chains, case lists, caching and model factory, on tiny synthetic VOIs."""

import warnings
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from monai.data import MetaTensor

from src.data.datasets import (build_dataset, build_loader, cache_dir, case_table, check_files_exist, data_dicts,
                               silence_cache_warning)
from src.data.transforms import eval_transforms, from_specs, train_transforms
from src.models.factory import build_model
from src.utils.config import load_yaml

REPO = Path(__file__).resolve().parents[1]
SHAPE = (64, 64, 32)
CENTER = (32, 32, 16)
INTENSITY_CFG = {"shifted_if_min_at_least": 0, "shift": 1024, "clip": [-1000, 1500]}
TRANSFORMS_CFG = {"patch_size": [32, 32, 16], "samples_per_volume": 3,
                  "augment": [{"name": "RandFlipd", "keys": ["image", "label"], "prob": 1.0, "spatial_axis": 0}]}


def write_voi(root: Path, name: str, offset: float) -> dict[str, str]:
    """Air (-1000 HU) with a 100 HU cube lesion at the centre, stored as (x, y, z, 1) like ULS23."""
    image = np.full(SHAPE, -1000.0, dtype=np.float32)
    label = np.zeros(SHAPE, dtype=np.float32)
    region = tuple(slice(c - 2, c + 2) for c in CENTER)
    image[region], label[region] = 100.0, 1.0
    paths = {"image": root / f"{name}_img.nii.gz", "label": root / f"{name}_lab.nii.gz"}
    nib.save(nib.Nifti1Image(image[..., None] + offset, np.eye(4)), paths["image"])
    nib.save(nib.Nifti1Image(label[..., None], np.eye(4)), paths["label"])
    return {"image": str(paths["image"]), "label": str(paths["label"]), "case": name, "dataset": "DS"}


@pytest.mark.parametrize("offset", [0.0, 1024.0])
def test_eval_transforms_correct_offset_and_scale(tmp_path: Path, offset: float) -> None:
    out = cast(dict[str, MetaTensor], eval_transforms(INTENSITY_CFG)(write_voi(tmp_path, "v", offset)))
    image, label = out["image"], out["label"]
    assert tuple(image.shape) == (1, *SHAPE) and image.dtype == torch.float16  # compact, as cached
    # -1000 HU -> 0 and 100 HU -> 1100/2500 = 0.44 after correction, whatever the stored offset.
    assert float(image.min()) == pytest.approx(0.0)
    assert float(image[(0, *CENTER)]) == pytest.approx(0.44, abs=1e-3)  # float16 step near 0.44
    assert set(torch.unique(label).tolist()) == {0.0, 1.0}
    assert image.affine.shape == (4, 4)  # metadata kept for saving predictions


def test_train_transforms_give_patches_that_contain_the_lesion(tmp_path: Path) -> None:
    transform = train_transforms(INTENSITY_CFG, TRANSFORMS_CFG)
    transform.set_random_state(seed=0)
    samples = cast(list[dict[str, torch.Tensor]], transform(write_voi(tmp_path, "v", 1024.0)))
    assert len(samples) == 3
    for sample in samples:
        assert tuple(sample["image"].shape) == (1, 32, 32, 16) and sample["image"].dtype == torch.float32
        # A 32x32x16 patch inside a 64x64x32 VOI nearly always covers the VOI centre (D19).
        assert sample["label"].sum() > 0


def test_from_specs_builds_monai_transforms() -> None:
    (flip,) = from_specs([{"name": "RandFlipd", "keys": ["image"], "prob": 0.5, "spatial_axis": 1}])
    assert type(flip).__name__ == "RandFlipd" and getattr(flip, "prob") == 0.5


def test_persistent_cache_gives_same_result(tmp_path: Path) -> None:
    item = write_voi(tmp_path, "v", 1024.0)
    cache = cache_dir(tmp_path / "work", INTENSITY_CFG)
    plain = cast(dict[str, torch.Tensor], build_dataset([item], eval_transforms(INTENSITY_CFG))[0])
    cached = build_dataset([item], eval_transforms(INTENSITY_CFG), cache=cache)
    first, second = (cast(dict[str, torch.Tensor], cached[0]) for _ in range(2))  # 2nd read is from the cache
    assert any(cache.iterdir())
    for key in ("image", "label"):
        assert torch.equal(plain[key], first[key]) and torch.equal(first[key], second[key])
    assert cache_dir(tmp_path, {**INTENSITY_CFG, "clip": [-500, 500]}) != cache_dir(tmp_path, INTENSITY_CFG)


@pytest.mark.parametrize("eval_first", [True, False])
def test_train_and_eval_share_one_cache_consistently(tmp_path: Path, eval_first: bool) -> None:
    # Both chains hash to the same cache file per VOI, so its content must suit both, whichever writes it.
    item = write_voi(tmp_path, "v", 1024.0)
    cache = cache_dir(tmp_path / "work", INTENSITY_CFG)
    reference = cast(dict[str, torch.Tensor], eval_transforms(INTENSITY_CFG)(item))
    eval_ds = build_dataset([item], eval_transforms(INTENSITY_CFG), cache=cache)
    train_ds = build_dataset([item], train_transforms(INTENSITY_CFG, TRANSFORMS_CFG), cache=cache)
    if eval_first:
        _ = eval_ds[0]
    samples = cast(list[dict[str, torch.Tensor]], train_ds[0])
    evaluated = cast(dict[str, torch.Tensor], eval_ds[0])
    assert len(list(cache.iterdir())) == 1
    assert all(s["image"].dtype == torch.float32 and tuple(s["image"].shape) == (1, 32, 32, 16) for s in samples)
    for key in ("image", "label"):
        assert evaluated[key].dtype == reference[key].dtype and torch.equal(evaluated[key], reference[key])


def test_cache_reads_do_not_warn_in_main_process_or_workers(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    item = write_voi(tmp_path, "v", 1024.0)
    dataset = build_dataset([item], eval_transforms(INTENSITY_CFG), cache=cache_dir(tmp_path / "work", INTENSITY_CFG))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_cache_warning()  # re-add inside this context, as build_dataset does in a real run
        _ = dataset[0], dataset[0]  # write, then read the cache
    assert not [w for w in caught if "torch.load" in str(w.message)]
    for _ in build_loader(dataset, batch_size=1, shuffle=False, num_workers=1):  # a fresh worker process
        pass
    assert "torch.load" not in capfd.readouterr().err


def test_case_table_and_data_dicts(tmp_path: Path) -> None:
    cases = ["a1", "a2", "a3", "b1"]
    pd.DataFrame({"dataset": ["A", "A", "A", "B"], "case": cases,
                  "patient_id": ["A:1", "A:1", "A:2", "B:1"], "split": ["train", "train", "val", "train"],
                  "image": [f"img/{c}.nii.gz" for c in cases], "label": [f"lab/{c}.nii.gz" for c in cases]},
                 ).to_csv(tmp_path / "splits.csv", index=False)
    pd.DataFrame({"dataset": ["A", "A", "A", "B"], "case": cases, "patient_id": ["x"] * 4,
                  "image": ["old/layout.nii.gz"] * 4, "label": ["old/layout.nii.gz"] * 4,
                  "spacing_z": [1.0, 5.0, 1.0, 0.8]}).to_csv(tmp_path / "manifest.csv", index=False)
    table = case_table(tmp_path / "manifest.csv", tmp_path / "splits.csv")
    assert table["patient_id"].tolist() == ["A:1", "A:1", "A:2", "B:1"]  # from the split file
    assert table["image"].tolist() == [f"img/{c}.nii.gz" for c in cases]  # so are the paths
    assert "spacing_z" in table
    assert [i["case"] for i in data_dicts(table, tmp_path, ["train"])] == ["a1", "a2", "b1"]
    items = data_dicts(table, tmp_path, ["train"], cases=["b1", "a1"])
    assert [i["case"] for i in items] == ["a1", "b1"]
    assert items[0]["image"] == str(tmp_path / "img/a1.nii.gz")
    with pytest.raises(ValueError, match="not in splits"):
        data_dicts(table, tmp_path, ["train"], cases=["a3"])  # a3 is a validation case


def test_check_files_exist_reports_missing_files(tmp_path: Path) -> None:
    item = write_voi(tmp_path, "v", 0.0)
    check_files_exist([item])
    Path(item["label"]).unlink()
    with pytest.raises(FileNotFoundError, match="1 of 2 image/label files"):
        check_files_exist([item])


def test_repo_dynunet_config_builds_and_runs() -> None:
    model = build_model(load_yaml(REPO / "configs/model/dynunet3d.yaml")["model"]).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 64, 64, 32))  # smallest size with a bottleneck above 1 voxel
    assert tuple(out.shape) == (1, 2, 64, 64, 32)


def test_factory_builds_2d_model_and_rejects_unknown_names() -> None:
    model = build_model({"name": "dynunet", "spatial_dims": 2, "in_channels": 1, "out_channels": 2,
                         "kernel_size": [3, 3, 3], "strides": [1, 2, 2], "filters": [8, 16, 32]}).eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 1, 16, 16)).shape) == (1, 2, 16, 16)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "nope"})
