import importlib.util
from pathlib import Path

import torch


MODEL_PATH = Path(__file__).resolve().parents[1] / "model.py"
TRAIN_PATH = Path(__file__).resolve().parents[1] / "train_immunogenicity.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODEL = load_module("mimicneoai_model_for_test", MODEL_PATH)
TRAIN = load_module("mimicneoai_train_for_test", TRAIN_PATH)


def build_model(fusion):
    return MODEL.NewBiLSTM(
        vocab_size=22,
        embedding_dim=12,
        hidden_dim_x1=8,
        hidden_dim_x2=4,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.0,
        pad_idx=0,
        x2_dim=25,
        physchem_fusion=fusion,
        physchem_adapter_dim=6,
        physchem_dropout=0.0,
    )


def test_zero_initialized_residual_starts_as_sequence_only():
    model = build_model("residual").eval()
    x1 = torch.randint(0, 22, (5, 10))
    x2 = torch.randn(5, 25)
    with torch.no_grad():
        full_logits = model(x1, x2)
        zero_logits = model(x1, torch.zeros_like(x2))
    assert torch.equal(full_logits, zero_logits)


def test_zero_input_remains_zero_through_bias_free_adapter():
    model = build_model("residual").eval()
    with torch.no_grad():
        model.physchem_adapter[-1].weight.normal_()
        residual = model.physchem_adapter(torch.zeros(4, 25))
    assert torch.count_nonzero(residual).item() == 0


def test_legacy_checkpoint_migration_keeps_sequence_classifier():
    legacy = build_model("legacy_concat")
    residual = build_model("residual")
    legacy_state = legacy.state_dict()
    loaded = TRAIN._migrate_legacy_phla_state(residual, legacy_state)
    assert "fc.weight" in loaded
    assert torch.equal(residual.fc.weight, legacy.fc.weight[:, : residual.fc.weight.shape[1]])
    assert torch.equal(residual.fc.bias, legacy.fc.bias)
    assert torch.count_nonzero(residual.physchem_adapter[-1].weight).item() == 0
