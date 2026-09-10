from pathlib import Path

import pandas as pd
import torch

from mimicneoai.immunogenicity_prediction.model import NewBiLSTM
from mimicneoai.immunogenicity_prediction.train_immunogenicity import (
    add_zero_peptide_features,
    extract_feature_vector,
    load_initial_weights,
)


def make_model(fusion: str) -> NewBiLSTM:
    torch.manual_seed(31)
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
    )


def test_sequence_only_logits_are_independent_of_physchem_features():
    model = make_model("sequence_only").eval()
    generator = torch.Generator().manual_seed(37)
    x1 = torch.randint(1, 21, (8, 20), generator=generator)
    x2 = torch.randn(8, 25, generator=generator)
    with torch.no_grad():
        first = model(x1, x2)
        second = model(x1, x2 * 1000.0 + 17.0)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_sequence_only_zero_feature_rows_have_25_values():
    frame = add_zero_peptide_features(pd.DataFrame({"peptide": ["SIINFEKL"]}))
    features = extract_feature_vector(frame.iloc[0])
    assert len(features) == 25
    assert features == [0.0] * 25


def test_encoder_only_checkpoint_transfer_skips_temporary_head(tmp_path: Path):
    source = make_model("sequence_only")
    with torch.no_grad():
        for parameter in source.encoder.parameters():
            parameter.add_(0.25)
        source.fc.weight.fill_(9.0)
        source.fc.bias.fill_(9.0)
    checkpoint = tmp_path / "stage1_sequence_only.pth"
    torch.save(
        {
            "model_state": source.state_dict(),
            "metadata": {"model_config": {"physchem_fusion": "sequence_only"}},
        },
        checkpoint,
    )

    target = make_model("legacy_concat")
    target_head_before = target.fc.weight.detach().clone()
    target_x2_before = target.fc_x2.weight.detach().clone()
    load_initial_weights(target, checkpoint, torch.device("cpu"), scope="encoder")

    for key, value in source.encoder.state_dict().items():
        torch.testing.assert_close(value, target.encoder.state_dict()[key], rtol=0.0, atol=0.0)
    torch.testing.assert_close(target.fc.weight, target_head_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target.fc_x2.weight, target_x2_before, rtol=0.0, atol=0.0)

