from pathlib import Path

import torch

from mimicneoai.immunogenicity_prediction.model import NewBiLSTM
from mimicneoai.immunogenicity_prediction.train_immunogenicity import (
    film_aux_multitask_loss,
    load_initial_weights,
)


def make_model(fusion: str = "film_aux") -> NewBiLSTM:
    torch.manual_seed(17)
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
        film_physchem_dim=8,
        film_group_hidden_dim=4,
        film_scale=0.1,
    )


def inputs(batch: int = 12):
    generator = torch.Generator().manual_seed(23)
    x1 = torch.randint(1, 21, (batch, 20), generator=generator)
    x2 = torch.randn(batch, 25, generator=generator)
    lengths = torch.full((batch,), 9, dtype=torch.long)
    labels = torch.arange(batch) % 2
    return x1, x2, lengths, labels


def test_zero_initialization_preserves_sequence_logits_exactly():
    model = make_model().eval()
    x1, x2, lengths, _ = inputs()
    with torch.no_grad():
        components = model(
            x1,
            x2,
            lengths,
            torch.ones(len(x1)),
            return_components=True,
        )
    torch.testing.assert_close(
        components["fused_logits"],
        components["sequence_logits"],
        rtol=0.0,
        atol=0.0,
    )


def test_absent_physchem_cannot_change_fused_logits():
    model = make_model().eval()
    x1, x2, lengths, _ = inputs()
    absent = torch.zeros(len(x1))
    with torch.no_grad():
        first = model(x1, x2, lengths, absent)
        second = model(x1, x2 * 1000.0 + 77.0, lengths, absent)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_auxiliary_loss_reaches_all_features_and_film_projection():
    model = make_model().train()
    x1, x2, lengths, labels = inputs()
    x2 = x2.requires_grad_(True)
    components = model(
        x1,
        x2,
        lengths,
        torch.ones(len(x1)),
        return_components=True,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(components["fused_logits"], labels) + 0.2 * loss_fn(
        components["physchem_logits"], labels
    )
    loss.backward()

    assert x2.grad is not None
    assert torch.all(x2.grad.abs().sum(dim=0) > 0)
    projection = model.physchem_film.film_projection
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    aux_classifier = model.physchem_film.physchem_classifier
    assert aux_classifier.weight.grad is not None
    assert aux_classifier.weight.grad.abs().sum() > 0


def test_film_modulation_learns_from_zero_initialization():
    model = make_model().train()
    x1, x2, lengths, labels = inputs(batch=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(3):
        optimizer.zero_grad()
        components = model(
            x1,
            x2,
            lengths,
            torch.ones(len(x1)),
            return_components=True,
        )
        loss = loss_fn(components["fused_logits"], labels) + 0.2 * loss_fn(
            components["physchem_logits"], labels
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, diagnostics = model(
            x1,
            x2,
            lengths,
            torch.ones(len(x1)),
            return_diagnostics=True,
        )
    assert float(diagnostics["modulation_norm"]) > 0.0
    assert float(diagnostics["modulation_to_base_ratio"]) > 0.0


def test_full_and_zero_have_identical_initial_sequence_encoder_gradients():
    full = make_model().train()
    zero = make_model().train()
    zero.load_state_dict(full.state_dict())
    x1, x2, lengths, labels = inputs()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    full_output = full(
        x1,
        x2,
        lengths,
        torch.ones(len(x1)),
        return_components=True,
    )
    full_loss, *_ = film_aux_multitask_loss(
        full_output,
        labels,
        torch.ones(len(x1)),
        loss_fn,
        aux_physchem_loss_weight=0.2,
        aux_sequence_loss_weight=0.1,
    )
    full_loss.mean().backward()

    zero_output = zero(
        x1,
        torch.zeros_like(x2),
        lengths,
        torch.zeros(len(x1)),
        return_components=True,
    )
    zero_loss, *_ = film_aux_multitask_loss(
        zero_output,
        labels,
        torch.zeros(len(x1)),
        loss_fn,
        aux_physchem_loss_weight=0.2,
        aux_sequence_loss_weight=0.1,
    )
    zero_loss.mean().backward()

    for (full_name, full_parameter), (zero_name, zero_parameter) in zip(
        full.encoder.named_parameters(), zero.encoder.named_parameters()
    ):
        assert full_name == zero_name
        if full_parameter.grad is None or zero_parameter.grad is None:
            assert full_parameter.grad is None and zero_parameter.grad is None
            continue
        torch.testing.assert_close(
            full_parameter.grad,
            zero_parameter.grad,
            rtol=0.0,
            atol=0.0,
        )


def test_embedding_refine_zero_checkpoint_initializes_film_sequence_encoder(tmp_path: Path):
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
    target = make_model("film_aux")
    load_initial_weights(target, checkpoint, torch.device("cpu"))
    for key, value in source.encoder.state_dict().items():
        if key.startswith("physchem_refiner."):
            continue
        torch.testing.assert_close(value, target.encoder.state_dict()[key], rtol=0.0, atol=0.0)


def test_checkpoint_roundtrip_preserves_all_heads(tmp_path: Path):
    model = make_model().eval()
    x1, x2, lengths, _ = inputs()
    with torch.no_grad():
        expected = model(
            x1,
            x2,
            lengths,
            torch.ones(len(x1)),
            return_components=True,
        )
    checkpoint = tmp_path / "film_aux.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    with torch.no_grad():
        actual = restored(
            x1,
            x2,
            lengths,
            torch.ones(len(x1)),
            return_components=True,
        )
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0.0, atol=0.0)
