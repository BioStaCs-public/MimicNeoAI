# Physicochemical embedding refinement

`embedding_refine` is the revised physicochemical fusion path for MimicNeoAI
immunogenicity models. It keeps the pHLA sequence encoder common across the
antigen-specific, pooled, and source-aware architectures.

## Data contract

- The encoded tensor contains 25 physicochemical features followed by peptide
  tokens, 12 zero separator tokens, HLA pseudo-sequence tokens, and padding.
- Peptide length is stored explicitly in encoded-cache version
  `peptide_hla_tensor_v2_explicit_lengths`.
- `physchem_present=1` enables refinement; `physchem_present=0` guarantees that
  physicochemical values cannot affect logits.
- Only peptide tokens are refined. HLA, separator, and padding embeddings are
  unchanged, and separator/padding embeddings remain exactly zero.

## Refiner

The first 18 composition features and the remaining 7 scalar properties are
encoded separately and fused into a 64-dimensional conditioning vector. This
vector produces feature-wise gamma and beta modulation. A position-aware gate
uses the peptide-token embedding and normalized N/C-terminal positions. The
result is added through a learnable layer-scale residual before the shared
BiLSTM pHLA encoder.

Gamma and beta output layers are initialized to zero. Consequently, full and
no-physchem models have identical logits before optimization.

## Scaling and checkpoints

Standardization is mandatory for `embedding_refine`. Mean and scale are fitted
from the training split only and embedded in every checkpoint with the complete
model configuration. Stage 2 automatically reuses the scaler bound to its
Stage 1 initialization checkpoint. Legacy checkpoints remain loadable by the
legacy architecture, but they cannot initialize an `embedding_refine` model.

## Required checks

Run:

```bash
python -m pytest -q mimicneoai/immunogenicity_prediction/tests
```

The test suite covers masking invariants, gradients for all 25 features,
refiner parameter updates, checkpoint roundtrips, training-only scaling, CUDA
mixed precision, a 512-sample overfit test, source-aware encoder reuse, and
legacy checkpoint behavior.
