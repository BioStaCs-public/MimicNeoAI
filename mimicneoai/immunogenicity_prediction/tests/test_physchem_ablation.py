import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_immunogenicity.py"
SPEC = importlib.util.spec_from_file_location("train_immunogenicity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_mode_returns_original_tensor():
    features = torch.arange(80, dtype=torch.float32).reshape(2, 40)
    output = MODULE.apply_physchem_mode(features, "full")
    assert output.data_ptr() == features.data_ptr()
    assert torch.equal(output, features)


def test_zero_mode_only_ablates_physicochemical_block():
    features = torch.arange(80, dtype=torch.float32).reshape(2, 40)
    original = features.clone()
    output = MODULE.apply_physchem_mode(features, "zero")
    assert torch.count_nonzero(output[:, : MODULE.X2_DIM]).item() == 0
    assert torch.equal(output[:, MODULE.X2_DIM :], original[:, MODULE.X2_DIM :])
    assert torch.equal(features, original)


def test_training_only_standardization():
    generator = torch.Generator().manual_seed(7)
    x2 = torch.randn(100, MODULE.X2_DIM, generator=generator) * 3.0 + 4.0
    sequence = torch.zeros(100, 12)
    features = torch.cat([x2, sequence], dim=1)
    scaler = MODULE.fit_physchem_scaler(features)
    transformed = MODULE.transform_physchem_features(features, scaler, clip=0.0)
    assert torch.allclose(transformed[:, : MODULE.X2_DIM].mean(dim=0), torch.zeros(MODULE.X2_DIM), atol=1e-6)
    assert torch.allclose(
        transformed[:, : MODULE.X2_DIM].std(dim=0, unbiased=False),
        torch.ones(MODULE.X2_DIM),
        atol=1e-6,
    )
    assert torch.equal(transformed[:, MODULE.X2_DIM :], sequence)
