# Binding Prediction Tests

Run the dependency-light contract and golden-fixture suite from the repository root:

```bash
PYTHONPATH=. python -m unittest discover -v \
  mimicneoai/functions/binding_prediction/tests
```

The suite does not invoke external binding predictors. It covers all three
pipeline backend branches, mutation peptide windows, cryptic and microbial FASTA
fixtures, WT/MT handling, HLA-II pairing, normalized status fields, EL versus
IC50 summary semantics, input-aware resume behavior, and the oversized-task
scale gate.
