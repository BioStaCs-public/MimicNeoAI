from pathlib import Path

import torch

from mimicneoai.immunogenicity_prediction.model import NewBiLSTM
from mimicneoai.immunogenicity_prediction.train_immunogenicity import load_initial_weights


def make_model(fusion: str = "gated_late") -> NewBiLSTM:
    torch.manual_seed(7)
    return NewBiLSTM(
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
        refiner_embedding_dim=16,
        late_physchem_dim=8,
        late_group_hidden_dim=4,
        late_gate_hidden_dim=8,
        late_alpha_init=0.1,
        late_alpha_max=0.5,
    )


def inputs(batch: int = 6):
    generator = torch.Generator().manual_seed(11)
    x1 = torch.randint(1, 21, (batch, 20), generator=generator)
    x2 = torch.randn(batch, 25, generator=generator)
    lengths = torch.full((batch,), 9, dtype=torch.long)
    return x1, x2, lengths


def test_zero_initialized_full_and_zero_logits_match():
    model = make_model().eval()
    x1, x2, lengths = inputs()
    with torch.no_grad():
        full = model(x1, x2, lengths, torch.ones(len(x1)))
        zero = model(x1, torch.zeros_like(x2), lengths, torch.zeros(len(x1)))
    torch.testing.assert_close(full, zero, rtol=0.0, atol=0.0)


def test_absent_physchem_cannot_change_logits():
    model = make_model().eval()
    x1, x2, lengths = inputs()
    with torch.no_grad():
        first = model(x1, x2, lengths, torch.zeros(len(x1)))
        second = model(x1, x2 * 1000.0 + 99.0, lengths, torch.zeros(len(x1)))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_physchem_branch_learns_and_all_input_dimensions_receive_gradient():
    model = make_model().train()
    x1, x2, lengths = inputs(batch=16)
    labels = torch.arange(len(x1)) % 2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(3):
        optimizer.zero_grad()
        loss = loss_fn(model(x1, x2, lengths, torch.ones(len(x1))), labels)
        loss.backward()
        optimizer.step()

    x2_grad = x2.detach().clone().requires_grad_(True)
    model(x1, x2_grad, lengths, torch.ones(len(x1))).sum().backward()
    assert x2_grad.grad is not None
    assert torch.all(x2_grad.grad.abs().sum(dim=0) > 0)
    assert 0.0 < float(model.physchem_late_fusion.alpha()) < 0.5


def test_embedding_refine_zero_checkpoint_initializes_sequence_encoder(tmp_path: Path):
    source = make_model("embedding_refine")
    checkpoint = tmp_path / "stage1_zero.pth"
    torch.save(
        {
            "model_state": source.state_dict(),
            "metadata": {
                "physchem_mode": "zero",
                "model_config": {"physchem_fusion": "embedding_refine"},
            },
        },
        checkpoint,
    )
    target = make_model("gated_late")
    load_initial_weights(target, checkpoint, torch.device("cpu"))
    for key, value in source.encoder.state_dict().items():
        if key.startswith("physchem_refiner."):
            continue
        torch.testing.assert_close(value, target.encoder.state_dict()[key], rtol=0.0, atol=0.0)


def test_checkpoint_roundtrip_is_exact(tmp_path: Path):
    model = make_model().eval()
    x1, x2, lengths = inputs()
    with torch.no_grad():
        expected = model(x1, x2, lengths, torch.ones(len(x1)))
    checkpoint = tmp_path / "gated_late.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    with torch.no_grad():
        actual = restored(x1, x2, lengths, torch.ones(len(x1)))
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
