# Immunogenicity Prediction

MimicNeoAI provides source-specific models for scoring peptide-HLA pairs from
microbial, mutation-derived, and cryptic antigen workflows. The same runtime can
be called from a pipeline, from a YAML-configured command, or through a Python
API.

Immunogenicity prediction is an optional downstream evidence layer. It does not
replace peptide-HLA binding prediction and does not establish natural
presentation or T-cell recognition.

## Model Scope

| Antigen class | Runtime model | Aggregation |
|---|---|---|
| `microbial` | One source-specific checkpoint | Single-model score |
| `mutation_derived` | One source-specific checkpoint | Single-model score |
| `cryptic` | Ten independently trained checkpoints | Mean of ten member scores |

The three model families were trained independently. Their raw probabilities
are suitable for ranking or thresholding within an antigen source, but they are
not a shared calibrated scale for comparing microbial, mutation-derived, and
cryptic candidates directly.

The 25-dimensional amino-acid physicochemical feature block is transformed with
the scaler stored in the corresponding checkpoint. The HLA representation must
come from the NetMHC-derived pseudosequence resources expected by that
checkpoint. These are part of the model contract, not optional preprocessing
choices.

## Runtime Environment

Install the Python dependencies from the repository root:

```bash
python -m pip install -e '.[immunogenicity]'
```

PyTorch must be compatible with the deployment's CPU or CUDA runtime. To avoid
coupling the sequencing environment to a particular PyTorch build, the helper
script can create a dedicated interpreter:

```bash
# CUDA 11.8 runtime
scripts/install_immunogenicity_runtime.sh \
  gpu-cu118 /path/to/MimicNeoAI_immunogenicity_gpu

# CPU-only runtime
scripts/install_immunogenicity_runtime.sh \
  cpu /path/to/MimicNeoAI_immunogenicity_cpu
```

Pipeline entry points resolve the interpreter in this order:

1. `others.immunogenicity_python_bin` in the workflow YAML;
2. `MIMICNEOAI_IMMUNOGENICITY_PYTHON_BIN` in the environment;
3. `path.common.IMMUNOGENICITY.PYTHON_BIN` in `paths.yaml`;
4. the Python interpreter running the pipeline.

Model-root resolution follows the same pattern through
`others.immunogenicity_model_root`,
`MIMICNEOAI_IMMUNOGENICITY_MODEL_ROOT`, and
`path.common.IMMUNOGENICITY.MODEL_ROOT`.

## Runtime Assets

Model checkpoints are recovered separately and are not committed to Git:

```text
mimicneoai/immunogenicity_prediction/models/default/
├── microbial/model.pth
├── mutation_derived/model.pth
└── cryptic/
    ├── member_01.pth
    ├── ...
    └── member_10.pth
```

The expected checkpoint hashes are recorded in the
[model payload README](models/README.md).

NetMHC-derived pseudosequence resources are stored under the ignored `local/`
directory described in the [HLA resource README](resources/hla_pseudoseq/README.md).
They are not redistributed because the source NetMHCpan and NetMHCIIpan files
are governed by their original download terms.

## Input Contract

The minimal input table contains one peptide-HLA pair per row:

```csv
peptide,hla
ILDAIELAV,HLA-A*02:01
MLAAKTTVPV,HLA-A*02:01
```

Required columns:

- `peptide`: amino-acid sequence;
- `hla`: normalized HLA allele or HLA-II heterodimer.

HLA-II DP and DQ inputs must be represented as the actual alpha-beta
heterodimer used by the binding prediction, not as independent alpha and beta
chains. Additional provenance columns are preserved in the returned table.

With input QC enabled, the runtime appends:

- `input_qc_peptide_length`;
- `input_qc_normalized_hla`;
- `input_qc_flags`;
- `input_qc_status`;
- `input_qc_hla_reference_status`;
- `immunogenicity_score`;
- `immunogenicity_status`.

Missing pseudosequences, malformed HLA values, missing peptides, and
noncanonical residues are reported explicitly. The runtime does not silently
rewrite or discard those rows. Duplicate peptide-HLA pairs may be evaluated
once internally and then merged back to the original records.

## Command-Line Use

The standalone CLI accepts a YAML configuration:

```bash
mimicneoai immunogenicity-prediction \
  -c mimicneoai/example/immunogenicity_prediction/config/microbial.yaml
```

The equivalent module entry point is:

```bash
python -m mimicneoai.immunogenicity_prediction.immunogenicity_prediction \
  -c mimicneoai/example/immunogenicity_prediction/config/microbial.yaml
```

The generic explicit-model template is
[`configures/immunogenicity_prediction_configure.yaml`](../configures/immunogenicity_prediction_configure.yaml).
It requires:

```yaml
path:
  input_csv: /path/to/peptide_hla.csv
  output_csv: /path/to/predictions.csv
  model_path: /path/to/model.pth
  hla_fasta: /path/to/hla_sequences.fasta

args:
  batch_size: 512
  num_processes: 8
  device: auto  # auto | cpu | cuda | cuda:<index>

io:
  peptide_col: peptide
  hla_col: hla
  score_col: immunogenicity_score
```

For the recovered source-specific models, use the example configurations under
`mimicneoai/example/immunogenicity_prediction/config/`. The cryptic example uses
the bundled ensemble runner:

```bash
python mimicneoai/example/immunogenicity_prediction/run_cryptic_ensemble.py \
  -c mimicneoai/example/immunogenicity_prediction/config/cryptic.yaml
```

## Python API

New pipeline code should use the stable source-aware helpers:

```python
import pandas as pd

from mimicneoai.functions.immunogenicity_runner import (
    predict_default_immunogenicity_df,
)

candidates = pd.DataFrame(
    {
        "peptide": ["ILDAIELAV"],
        "hla": ["HLA-A*02:01"],
    }
)

predictions = predict_default_immunogenicity_df(
    candidates,
    antigen_class="microbial",
    device="auto",
    include_input_qc=True,
)
```

Available source-aware entry points are:

- `predict_default_immunogenicity_df`;
- `predict_default_immunogenicity_csv`.

Lower-level interfaces in `immunogenicity_prediction.api` support explicit
model paths. `core.py`, `default_models.py`, and
`immunogenicity_prediction.py` remain compatibility surfaces; new pipeline code
should not depend on their internal implementation directly.

## Pipeline Integration

All three antigen workflows keep immunogenicity disabled by default. Enable it
only after a valid binding stage:

```yaml
others:
  run_immunogenicity_prediction: true
  immunogenicity_device: auto
  immunogenicity_batch_size: 512
  immunogenicity_workers: 8
```

The pipeline passes the correct antigen class automatically. A binding sample
that is skipped by a scale guard, or a peptide-HLA record that cannot be
evaluated against the required HLA resource, is reported as skipped or not
evaluable rather than as immunogenicity-negative.

For mutation-derived antigens, MT and matched-WT rows are both scored and
preserved. MT remains the primary candidate for downstream prioritization; WT
is a control and must not be counted as a mutation-derived candidate.

## Reproduction Check

Recovered model payloads can be compared with a locked prediction table:

```bash
python -m mimicneoai.immunogenicity_prediction.tests.verify_locked_prediction_reproduction \
  --reference-tsv /path/to/locked_predictions.tsv \
  --antigen-class microbial \
  --device cuda
```

The verifier uses an absolute score tolerance of `1e-3` by default to
accommodate the small floating-point and feature-recalculation differences
observed across supported runtime environments.

## Training and Benchmark Code

`train_immunogenicity.py` and `benchmark/` are research and model-development
utilities. They are not required for formal inference, and running them does not
replace or update the frozen default checkpoints. Benchmark helper usage is
documented in [`benchmark/README.md`](benchmark/README.md).

## Interpretation

`immunogenicity_score` is a model output conditioned on the peptide, HLA
representation, antigen source, and frozen checkpoint. It should be interpreted
with its `immunogenicity_status` and input-QC fields. A high score is not direct
evidence of endogenous processing, HLA presentation, T-cell activation, or
clinical response.
