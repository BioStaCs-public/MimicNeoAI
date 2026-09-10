# Native Binding Prediction Backend

MimicNeoAI provides a local binding-prediction backend shared by the microbial,
cryptic, and mutation-derived antigen pipelines. All three packaged workflow
templates use the native `mimicneoai` backend with the `fast` preset by default.
The legacy `pvactools` backend remains available only when selected explicitly.

## Pipeline Boundary

- Mutation-derived antigens use external pVACtools only for VCF conversion and
  WT/MT protein FASTA generation. MimicNeoAI builds mutation-covering peptide
  windows, predicts binding, and writes a pVACseq-compatible merged table.
- Cryptic and microbial antigens read their peptide FASTA directly, build
  de-duplicated peptide-HLA-algorithm tasks, and write a pVACbind-compatible
  merged table.
- Matched-normal microbial analyses first build a tumor-only peptide Core.
  Their FASTA input is already tiled; it is passed with
  `--input-mode peptide-core` so the common workflow writes one epitope-window
  row per FASTA record instead of sliding windows across it again.
- The common runner does not perform antigen-specific biological filtering.

## Configuration

Common native-backend options:

```yaml
others:
  binding_prediction_backend: "mimicneoai"
  binding_prediction_preset: "fast"
  binding_prediction_workers: 8
```

`binding_prediction_preset` is optional. If it is omitted, the backend keeps
using the explicitly configured lengths and algorithm list.

Available presets:

- `full`: one-stage multialgorithm prediction. Uses HLA-I 8-11 aa and HLA-II
  13-17 aa. HLA-I predictors are MHCflurry, MHCflurryEL, MHCnuggetsI,
  NetMHCpan, and NetMHCpanEL. HLA-II predictors are MHCnuggetsII, NNalign,
  NetMHCIIpan, and NetMHCIIpanEL.
- `fast`: two-stage prediction for large peptide sets. Stage 1 routes by
  NetMHCpanEL/NetMHCIIpanEL `%Rank < 10` only. Stage 2 runs the same HLA-I
  predictors as `full` and HLA-II predictors MHCnuggetsII, NetMHCIIpan, and
  NetMHCIIpanEL. NNalign is intentionally omitted in this preset.

Stage 1 is a routing filter, not a final binding or immunogenicity tier. It
records `stage1_screen_pass`, `screened_out_stage1`, unsupported HLA alleles,
and missing/failed predictions separately. For mutation-derived antigens,
Stage 1 is evaluated on MT peptides; when an MT peptide passes, the matched WT
peptide is carried into Stage 2 for the existing WT/MT fold-change calculation.

Predictor installation paths are deployment settings under
`path.common.BINDING_PREDICTORS` in `configures/paths.yaml`. They are passed
through the pipeline entry point to the common runner, including the
MHCflurry model directory and the MHCnuggets/IEDB working directories. The
matching `MIMICNEOAI_*` environment variables remain available for standalone
runner use.

For mutation-derived antigens, the native backend still calls pVACtools for VCF
conversion and WT/MT protein source generation. Configure
`path.common.APPTAINER_BIN`, `path.common.BCFTOOLS_BIN`, and
`path.common.TABIX_BIN` in `configures/paths.yaml` when deployment should use
absolute tool paths rather than relying on the shell `PATH`.

Cryptic and microbial pipelines also support:

```yaml
  binding_prediction_max_task_rows: 5000000
  binding_prediction_force_large_samples: false
```

The task limit is evaluated after peptide-window construction but before
`binding_tasks.tsv` is materialized. Oversized samples retain epitope windows,
the task-count estimate, and a manifest. Prediction runs only when
`binding_prediction_force_large_samples` is explicitly enabled.

## Output Layout

Mutation-derived output:

```text
07.binding_prediction_mimicneoai/
├── 00_input_vcf
├── 01_pvactools_sources
├── 02_epitope_tasks
├── 03_binding_predictions
├── 04_merged_epitopes
└── archive
```

Cryptic and microbial native output:

```text
<binding_step>_mimicneoai/
├── mimicneoai_epitope_tasks
├── mimicneoai_binding_predictions
├── combined
└── <sample>.mimicneoai_binding.summary.json
```

## Resume Rules

- Mutation pVACtools source files record the VCF, flank length, pass-only mode,
  and pVACtools image identity. A later mismatch stops with an explicit request
  to use a new output directory instead of silently reusing converter/FASTA
  files.
- Non-mutation epitope windows are reused only when the peptide FASTA identity
  and requested peptide lengths match their manifest.
- Binding tasks are reused only when the epitope-window identity, HLA file, and
  algorithm sets match their manifest.
- Predictor chunks are reused only when normalized rows exactly match the
  current peptide, HLA, algorithm, MHC class, and peptide length requests and
  contain no `status=error` rows.
- Changed inputs rebuild the affected stage instead of silently accepting stale
  output.

File identity uses resolved path, size, and nanosecond modification time. Use a
new output directory when replacing an input while preserving all three values.

## Output Semantics

- `IC50_SUMMARY_ALGORITHMS` contains binding-affinity algorithms only.
- EL/presentation algorithms contribute their score and percentile fields but
  never contribute to Best/Median IC50 or fold-change summaries.
- Mutation missense and in-frame events retain corresponding WT predictions.
- Frameshift WT peptide, WT IC50, WT percentile, and fold-change fields remain
  blank.
- Unsupported predictor-HLA pairs remain in the normalized long table with
  `status=skipped` and `error=unsupported_allele_by_predictor`.
- Predictor failures remain in the long table with `status=error`.
- Partial predictor failures remain informational. A run with runnable tasks
  but no usable prediction rows exits nonzero and stops the parent pipeline.

## HLA Support Discovery

The runner builds support catalogs from the installed predictor resources:

- MHCflurry model `allele_sequences.csv` or `--list-supported-alleles`;
- NetMHCpan `-listMHC`;
- NetMHCIIpan `-list`;
- MHCnuggets model and training-resource intersection;
- IEDB MHC-I/MHC-II allele-info pickle files.

`binding_predictions.summary.json` records `allele_support_matrix` and
`predictor_runtime`, including catalog sources, allele counts, configured
executables, scripts, and environments. Catalog discovery failure is fail-open:
the task is attempted and the adapter records the real execution result.

## IEDB MHC-II and NNalign Runtime

IEDB MHC-II 3.1.11 imports the legacy `pkg_resources` API. Use a dedicated
Python 3.10 environment and pin setuptools below version 81:

```bash
python3.10 -m venv /path/to/IEDB/.venv
/path/to/IEDB/.venv/bin/python -m pip install 'setuptools<81'
```

Configure `IEDB_MHCII_PYTHON_BIN` with
`/path/to/IEDB/.venv/bin/python`. The binding runner executes an NNalign
runtime preflight before scheduling supported NNalign tasks, so a missing or
incompatible IEDB environment fails before prediction starts.

## Regression Tests

```bash
PYTHONPATH=. python -m unittest discover -v \
  mimicneoai/functions/binding_prediction/tests
```

The dependency-light suite does not invoke external predictors. It covers all
three pipeline backend branches, mutation event classes, cryptic and microbial
FASTA fixtures, HLA-II pairing, unsupported alleles, scale gating, resume input
signatures, normalized error states, and merged-table summary semantics.

## Interpretation

The native backend reports algorithm-level predictions and aggregation fields;
it does not perform antigen-source discovery QC. A Stage 1 routing failure,
unsupported predictor-allele combination, scale-gated sample, or execution
error is distinct from a completed non-binding prediction and remains explicit
in the output status fields.
