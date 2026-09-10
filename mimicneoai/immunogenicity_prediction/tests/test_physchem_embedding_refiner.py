import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODEL = load_module("mimicneoai_refiner_model_test", ROOT / "model.py")
TRAIN = load_module("mimicneoai_refiner_train_test", ROOT / "train_immunogenicity.py")


def build_model(dropout=0.0):
    torch.manual_seed(17)
    return MODEL.NewBiLSTM(
        vocab_size=22,
        embedding_dim=12,
        hidden_dim_x1=8,
        hidden_dim_x2=4,
        output_dim=2,
        n_layers=1,
        bidirectional=True,
        dropout=dropout,
        pad_idx=0,
        x2_dim=25,
        physchem_fusion="embedding_refine",
        refiner_embedding_dim=16,
        refiner_composition_dim=18,
        refiner_composition_hidden_dim=12,
        refiner_scalar_hidden_dim=8,
        refiner_fusion_hidden_dim=20,
        refiner_gate_hidden_dim=10,
        refiner_dropout=0.0,
        refiner_layer_scale_init=0.1,
    )


def build_source_aware_model():
    return MODEL.SourceAwareBiLSTM(
        vocab_size=22,
        embedding_dim=12,
        hidden_dim_x1=8,
        hidden_dim_x2=4,
        output_dim=2,
        n_layers=1,
        bidirectional=True,
        dropout=0.0,
        pad_idx=0,
        x2_dim=25,
        source_count=3,
        source_embedding_dim=4,
        physchem_fusion="embedding_refine",
        refiner_embedding_dim=16,
        refiner_composition_dim=18,
        refiner_composition_hidden_dim=12,
        refiner_scalar_hidden_dim=8,
        refiner_fusion_hidden_dim=20,
        refiner_gate_hidden_dim=10,
        refiner_dropout=0.0,
        refiner_layer_scale_init=0.1,
    )


def inputs(batch=4):
    lengths = torch.tensor(([4, 5, 6, 7] * ((batch + 3) // 4))[:batch], dtype=torch.long)
    x1 = torch.zeros(batch, 30, dtype=torch.long)
    for row, length in enumerate(lengths.tolist()):
        x1[row, :length] = torch.arange(1, length + 1) % 21 + 1
        hla_start = length + 12
        x1[row, hla_start : hla_start + 6] = torch.tensor([2, 4, 6, 8, 10, 12])
    x2 = torch.randn(batch, 25)
    return x1, x2, lengths


def test_zero_initialization_full_and_zero_logits_are_identical():
    model = build_model().eval()
    x1, x2, lengths = inputs()
    with torch.no_grad():
        full = model(x1, x2, lengths, torch.ones(len(x1)))
        zero = model(x1, torch.zeros_like(x2), lengths, torch.zeros(len(x1)))
    assert torch.equal(full, zero)


def test_source_aware_and_phla_models_use_the_common_encoder():
    phla = build_model().eval()
    source_aware = build_source_aware_model().eval()
    assert isinstance(phla.encoder, MODEL.PHLASequenceEncoder)
    assert isinstance(source_aware.encoder, MODEL.PHLASequenceEncoder)
    x1, x2, lengths = inputs()
    with torch.no_grad():
        logits = source_aware(
            x1,
            x2,
            torch.tensor([0, 1, 2, 0]),
            peptide_lengths=lengths,
            physchem_present=torch.ones(len(x1)),
        )
    assert logits.shape == (len(x1), 2)


def test_presence_zero_blocks_arbitrary_physchem_values():
    model = build_model().eval()
    x1, x2, lengths = inputs()
    refiner = model.encoder.physchem_refiner
    with torch.no_grad():
        refiner.gamma.weight.normal_(0, 0.2)
        refiner.beta.weight.normal_(0, 0.2)
        a = model(x1, x2, lengths, torch.zeros(len(x1)))
        b = model(x1, x2 * 1000 + 99, lengths, torch.zeros(len(x1)))
    assert torch.equal(a, b)


def test_only_peptide_tokens_change_and_hla_separator_padding_are_invariant():
    model = build_model().eval()
    x1, x2, lengths = inputs()
    refiner = model.encoder.physchem_refiner
    with torch.no_grad():
        refiner.gamma.bias.fill_(0.25)
        refiner.beta.bias.fill_(0.15)
        _, full_diag = model(
            x1, x2, lengths, torch.ones(len(x1)), return_diagnostics=True
        )
        _, zero_diag = model(
            x1, x2, lengths, torch.zeros(len(x1)), return_diagnostics=True
        )
    full_emb = full_diag["encoder_embeddings"]
    zero_emb = zero_diag["encoder_embeddings"]
    peptide_mask = full_diag["peptide_mask"]
    assert torch.count_nonzero(full_emb[peptide_mask] - zero_emb[peptide_mask]).item() > 0
    assert torch.equal(full_emb[~peptide_mask], zero_emb[~peptide_mask])
    for row, length in enumerate(lengths.tolist()):
        assert torch.count_nonzero(full_emb[row, length : length + 12]).item() == 0
        hla_end = length + 12 + 6
        assert torch.count_nonzero(full_emb[row, hla_end:]).item() == 0
        assert torch.equal(
            full_emb[row, length + 12 : hla_end],
            zero_emb[row, length + 12 : hla_end],
        )


def test_all_physchem_inputs_and_refiner_components_receive_updates():
    model = build_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x1, x2, lengths = inputs(batch=8)
    labels = torch.arange(8) % 2
    refiner = model.encoder.physchem_refiner
    before = {
        "gamma": copy.deepcopy(refiner.gamma.state_dict()),
        "beta": copy.deepcopy(refiner.beta.state_dict()),
        "gate": copy.deepcopy(refiner.gate.state_dict()),
        "layer_scale": refiner.layer_scale.detach().clone(),
    }
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x1, x2, lengths, torch.ones(len(x1)))
        nn.functional.cross_entropy(logits, labels).backward()
        optimizer.step()

    x2_grad = x2.detach().clone().requires_grad_(True)
    model(x1, x2_grad, lengths, torch.ones(len(x1))).sum().backward()
    assert x2_grad.grad is not None
    assert torch.all(x2_grad.grad.abs().sum(dim=0) > 0)
    for name in ("gamma", "beta", "gate"):
        current = getattr(refiner, name).state_dict()
        assert any(not torch.equal(current[key], before[name][key]) for key in current)
    assert not torch.equal(refiner.layer_scale.detach(), before["layer_scale"])


def test_checkpoint_roundtrip_is_exact_and_contains_scaler(tmp_path):
    model = build_model().eval()
    x1, x2, lengths = inputs()
    scaler = {"mean": [0.0] * 25, "scale": [1.0] * 25, "n_fit": 4, "fit_source": "train_split"}
    metadata = {
        "model_config": {"physchem_fusion": "embedding_refine"},
        "physchem_scaler": scaler,
    }
    path = tmp_path / "model.pth"
    with torch.no_grad():
        expected = model(x1, x2, lengths, torch.ones(len(x1)))
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)
    restored = build_model().eval()
    TRAIN.load_initial_weights(restored, path, torch.device("cpu"))
    with torch.no_grad():
        actual = restored(x1, x2, lengths, torch.ones(len(x1)))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["physchem_scaler"]["fit_source"] == "train_split"
    assert torch.equal(expected, actual)


def test_legacy_checkpoint_loads_but_cannot_initialize_new_refiner(tmp_path):
    legacy = MODEL.NewBiLSTM(
        vocab_size=22,
        embedding_dim=12,
        hidden_dim_x1=8,
        hidden_dim_x2=4,
        output_dim=2,
        n_layers=1,
        bidirectional=True,
        dropout=0.0,
        pad_idx=0,
        x2_dim=25,
        physchem_fusion="legacy_concat",
    )
    historical_state = {}
    for key, value in legacy.state_dict().items():
        historical_key = key.removeprefix("encoder.") if key.startswith("encoder.") else key
        historical_state[historical_key] = value
    path = tmp_path / "historical_state_dict.pth"
    torch.save(historical_state, path)

    restored_legacy = copy.deepcopy(legacy)
    TRAIN.load_initial_weights(restored_legacy, path, torch.device("cpu"))
    assert all(
        torch.equal(value, restored_legacy.state_dict()[key])
        for key, value in legacy.state_dict().items()
    )
    with pytest.raises(RuntimeError, match="may only initialize"):
        TRAIN.load_initial_weights(build_model(), path, torch.device("cpu"))


def test_scaler_is_fit_from_train_tensor_only():
    train = torch.cat((torch.zeros(6, 25), torch.zeros(6, 20)), dim=1)
    validation = torch.cat((torch.full((6, 25), 1000.0), torch.zeros(6, 20)), dim=1)
    scaler = TRAIN.fit_physchem_scaler(train)
    assert scaler["n_fit"] == len(train)
    assert scaler["mean"] == [0.0] * 25
    transformed_validation = TRAIN.transform_physchem_features(validation, scaler, clip=0.0)
    assert torch.all(transformed_validation[:, :25] == 1000.0)


def test_stage2_can_reuse_checkpoint_bound_stage1_scaler(tmp_path):
    scaler = {
        "mean": list(range(25)),
        "scale": [2.0] * 25,
        "n_fit": 100,
        "fit_source": "train_split",
    }
    path = tmp_path / "stage1.pth"
    torch.save(
        {
            "model_state": build_model().state_dict(),
            "metadata": {"physchem_scaler": scaler},
        },
        path,
    )
    loaded = TRAIN.load_checkpoint_physchem_scaler(path, torch.device("cpu"))
    assert loaded == scaler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP check requires a GPU")
def test_mixed_precision_has_no_nan_or_inf():
    model = build_model().cuda().train()
    x1, x2, lengths = (tensor.cuda() for tensor in inputs(batch=16))
    labels = torch.arange(16, device="cuda") % 2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x1, x2, lengths, torch.ones(len(x1), device="cuda"))
        loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_512_sample_overfit_loss_decreases():
    torch.manual_seed(29)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device).train()
    x1, x2, lengths = inputs(batch=512)
    x1, x2, lengths = x1.to(device), x2.to(device), lengths.to(device)
    labels = (x2[:, 0] + 0.5 * x2[:, 19] > 0).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x1, x2, lengths, torch.ones(len(x1)))
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.75
    with torch.no_grad():
        _, diagnostics = model(
            x1[:32], x2[:32], lengths[:32], torch.ones(32), return_diagnostics=True
        )
    assert torch.isfinite(diagnostics["delta_to_base_ratio"])
    assert diagnostics["delta_to_base_ratio"] > 0
