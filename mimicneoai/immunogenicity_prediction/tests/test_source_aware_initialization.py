from pathlib import Path

import pytest
import torch

from mimicneoai.immunogenicity_prediction.model import SourceAwareBiLSTM
from mimicneoai.immunogenicity_prediction.train_immunogenicity import load_initial_weights


def make_model(fusion: str) -> SourceAwareBiLSTM:
    torch.manual_seed(101)
    return SourceAwareBiLSTM(
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
        source_count=3,
        source_embedding_dim=5,
        physchem_fusion=fusion,
    )


def test_encoder_source_transfer_preserves_encoder_and_source_embedding(tmp_path: Path):
    source = make_model("sequence_only")
    with torch.no_grad():
        for parameter in source.encoder.parameters():
            parameter.add_(0.25)
        source.source_embedding.weight.add_(0.5)
    checkpoint = tmp_path / "source_aware_stage1.pth"
    torch.save({"model_state": source.state_dict()}, checkpoint)

    target = make_model("legacy_concat")
    target_head_before = target.fc.weight.detach().clone()
    target_physchem_before = target.fc_x2.weight.detach().clone()
    load_initial_weights(
        target,
        checkpoint,
        torch.device("cpu"),
        scope="encoder_source",
    )

    for key, value in source.encoder.state_dict().items():
        torch.testing.assert_close(value, target.encoder.state_dict()[key], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        source.source_embedding.weight,
        target.source_embedding.weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(target.fc.weight, target_head_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target.fc_x2.weight, target_physchem_before, rtol=0.0, atol=0.0)


def test_encoder_source_transfer_rejects_non_source_aware_target(tmp_path: Path):
    source = make_model("sequence_only")
    checkpoint = tmp_path / "source_aware_stage1.pth"
    torch.save({"model_state": source.state_dict()}, checkpoint)
    del source.source_embedding
    with pytest.raises(RuntimeError, match="requires a source-aware model"):
        load_initial_weights(
            source.encoder,
            checkpoint,
            torch.device("cpu"),
            scope="encoder_source",
        )
