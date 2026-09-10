# Molecular Mimicry

MimicNeoAI connects candidate peptide repertoires from different antigen sources
through within-patient sequence comparison. The module evaluates unique 8–11-aa
HLA-I candidates retained after source-specific quality control, first identifying
sequence-similar pairs and then integrating patient-specific HLA-binding predictions
to identify a more focused set of predicted mimicry candidates.

## Cross-source Peptide Sequence Matching

Comparisons are performed within each patient between peptides from different
antigen sources. Only peptides of the same length are compared, and all sequences
must contain the 20 standard amino acids.

For an 8-mer peptide, the central region comprises P4–P7; for 9–11-mer peptides,
it comprises P4–P8. A peptide pair is classified as a sequence-similar pair when
it meets all three criteria:

1. at least 60% positional identity within the central region;
2. at least three consecutive identical positions within this region; and
3. at least four identical positions across the full peptide.

These criteria combine central-region similarity with a minimum level of
full-peptide identity. The module also records P2 and PΩ identity, BLOSUM62
scores and additional sequence metrics for each retained pair.

The supported comparisons are Microbial–Mutation, Microbial–Cryptic and
Cryptic–Mutation. Microbial–Mutation and Microbial–Cryptic comprise the
microbial–host analyses, whereas Cryptic–Mutation is reported separately as
cross-source tumour-antigen sequence similarity.

For each patient and source pair, MimicNeoAI reports the number of eligible
same-length comparisons, the number of sequence-similar pairs and the normalized
frequency per 10^6 comparisons. Peptide, parent-sequence and mutation-event
provenance is retained for downstream interpretation.

## Mutation-specific Sequence Similarity

For sequence-similar pairs involving a mutation-derived peptide, the same criteria
are applied to the corresponding event-specific matched-WT peptide. Evaluable
pairs are classified as:

- **MT only**, when the MT peptide forms a sequence-similar pair but its matched
  WT peptide does not; or
- **MT and matched WT**, when both peptides meet the sequence criteria.

Frameshift contexts and events without a conventional matched-WT peptide are
recorded separately. This comparison identifies the subset of cross-source
sequence similarity attributable to the mutant peptide while retaining events
shared with the matched-WT context.

## Predicted Mimicry Candidates with Shared HLA-I Binding

When MimicNeoAI binding results are provided, each sequence-similar pair is
evaluated against the patient’s HLA-A, HLA-B and HLA-C alleles. A peptide is
considered binding supported when both the best and median predicted IC50 values
are below 500 nM and both the best and median percentile ranks are below 2.

A sequence-similar pair is classified as a predicted mimicry candidate when both
peptides meet these criteria for the same patient HLA-I allele. The output records
the supporting allele or alleles and the corresponding binding metrics, providing
a focused set of peptide pairs for subsequent immunopeptidomics, TCR and
experimental evaluation.

Enable this assessment and provide the corresponding MimicNeoAI merged binding
tables through `binding_path`:

```yaml
binding_support:
  enabled: true
```

## Command Line

Create a compact run configuration:

```yaml
method: "sequence_mimicry_v1.2"
input_manifest: "/analysis/mimicry_inputs.tsv"
output_dir: "/analysis/mimicry"

source_pairs:
  - "microbial--mutation-derived"
  - "microbial--cryptic"
  - "cryptic--mutation-derived"

workers: 8
shard_count: 2
```

Then run:

```bash
mimicneoai mimicry -c mimicry.yaml
```

The method identifier records the implemented eligibility criteria and thresholds.
Alternative threshold sets should be assigned a distinct method identifier.

`workers` limits concurrent search processes. `shard_count` divides each
patient, source-pair, and peptide-length comparison into deterministic,
resumable units. More shards may improve utilization for a single large
comparison, but each shard rebuilds its sequence index. A value of 1 or 2 is
therefore preferable for multi-patient analyses; increase it only when one
comparison does not provide enough parallel work.

## Input Manifest

The tab-delimited input manifest has one row per patient and antigen source:

| patient_id | antigen_source | input_format | peptide_path | provenance_path | run_manifest_path | binding_path |
|---|---|---|---|---|---|---|
| Patient_1 | microbial | microbial_core_v1 | `<microbial-core.tsv>` | `<parent-map.tsv>` | `<run-manifest.json>` | `<binding.tsv>` |
| Patient_1 | cryptic | cryptic_final_core_v1_1 | `<cryptic-core.tsv>` | `<final-sidecar.tsv>` | `<run-manifest.json>` | `<binding.tsv>` |
| Patient_1 | mutation-derived | mutation_epitope_windows_v1 | `<epitope-windows.tsv>` | | | `<binding.tsv>` |

Relative paths are resolved from the manifest directory. Supported formats
are:

- `microbial_core_v1`: paired microbial Core records with
  `peptide_qc_status=core`;
- `cryptic_final_core_v1_1`: final external-normal-QC records with
  `peptide_core_status=core`;
- `mutation_epitope_windows_v1`: mutation-covering MT windows and event-level
  WT controls;
- `generic_peptide_table_v1`: a portable table containing `peptide`,
  `peptide_id`, `mhc_class`, and `qc_status=core`.

For mutation input, the generic format may additionally contain `event_id`,
`wt_peptide`, and `wt_control_status`. Native mutation output uses
`mt_epitope_seq`, `wt_epitope_seq`, and `covers_mutation` directly.

When a provenance table is supplied, every eligible peptide must map to at
least one provenance row. Incomplete upstream manifests and Core manifests with
`binding_eligible=false` are reported as input-validation errors.

## Outputs

```text
mimicry/
├── mimicry_pairs.tsv.gz
├── mimicry_mutation_wt_evidence.tsv.gz
├── mimicry_member_provenance.tsv.gz
├── mimicry_hla_support.tsv.gz
├── mimicry_input_qc.tsv
├── mimicry_stagewise_qc.tsv
├── run_manifest.json
└── .parts/
```

`mimicry_pairs.tsv.gz` contains the sequence-similar pairs identified by the
analysis. The `primary_call` field records these pairs as
`sequence_based_mimicry_v12_candidate`. When binding support is enabled, their
shared HLA-I binding status and predicted mimicry classification are recorded
in `mimicry_hla_support.tsv.gz`. The stagewise table records the number of
evaluable same-length comparisons and the counts retained at each search stage.
Member provenance is written once for each participating source occurrence and
retains the original source row as JSON.

For source pairs containing mutation-derived peptides, **MT only** and
**MT and matched WT** are recorded as
`MT_specific_sequence_mimic` and `WT_compatible_sequence_mimic`, respectively.
Non-evaluable events are recorded as `WT_not_evaluable` or
`WT_not_evaluable_for_neo_specificity`.

The manifest records resolved policy values, source file identities, code
identities, output hashes, worker settings, and elapsed time. Existing results
are reused when the complete input and output contracts match.

The result is written to `mimicry_hla_support.tsv.gz`. Pairs lacking complete
binding metrics receive the status `binding_not_evaluable`. Differences between
the HLA-A/B/C allele sets represented in the two source results are recorded as
`binding_not_evaluable_hla_contract_mismatch`. The output retains the original
sequence classification alongside the shared-binding annotation.

## Python API

```python
from mimicneoai.mimicry import compute_pair_metrics, is_sequence_mimic

is_sequence_mimic("AAAAAAAA", "CAAAAACC")
metrics = compute_pair_metrics("AAAAAAAA", "CAAAAACC")
```

The stable public API is exported from `mimicneoai.mimicry`. Modules beginning
with an underscore are implementation details.
