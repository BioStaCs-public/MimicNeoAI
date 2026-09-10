# Immunogenicity benchmark helpers

This directory contains generic benchmark utilities for MimicNeoAI
immunogenicity models. The scripts do not contain project-specific paths,
private datasets, or local tool installation paths.

## Evaluate prediction files

Input prediction files must include a binary label column and a score column.

```bash
python -m mimicneoai.immunogenicity_prediction.benchmark.evaluate_predictions \
  --predictions predictions.tsv \
  --label-col label \
  --score-col score \
  --method MimicNeoAI \
  --benchmark microbial_internal_heldout \
  --outdir benchmark/microbial/MimicNeoAI
```

Outputs include:

- `metrics.tsv`
- `metrics.json`
- `reliability_bins.tsv`
- `precision_recall_curve.tsv`
- optional `group_metrics.tsv`

## Lightweight internal baselines

Three baseline models are provided:

- `aa_composition`: peptide length plus amino-acid composition
- `peptide_only`: peptide character n-gram logistic regression
- `hla_only`: HLA character n-gram logistic regression

```bash
python -m mimicneoai.immunogenicity_prediction.benchmark.train_internal_baseline \
  --baseline peptide_only \
  --train-csv train.model_input.tsv \
  --val-csv validation.model_input.tsv \
  --test-csv test.model_input.tsv \
  --benchmark microbial_internal_heldout \
  --outdir benchmark/microbial/peptide_only
```

