# Binding Backend Fast Formalization Validation

Date: 2026-07-13

## Scope

This validation covers pipeline entry integration, output contracts, oversized
non-mutation task handling, dependency-light regression fixtures, and runtime
documentation. Full-size end-to-end prediction and performance benchmarking are
outside this fast validation scope.

## Acceptance results

| Area | Result | Evidence |
|---|---|---|
| Default backend | Pass | All three packaged YAML files select `pvactools`; tests also omit the key and verify legacy dispatch. |
| Native entry dispatch | Pass | Mutation, cryptic, and microbial tests verify script, output directory, worker, and scale-control arguments. |
| Mutation fixture | Pass | SNV, in-frame insertion, in-frame deletion, frameshift, and N-terminal mutation windows are covered. |
| Cryptic fixture | Pass | Short ORF handling, source preservation, duplicate-sequence task de-duplication, and I/II lengths are covered. |
| Microbial fixture | Pass | Multiple proteins, duplicate sequences, source preservation, and task de-duplication are covered. |
| Resume behavior | Pass | Unchanged intermediates are reused; changed FASTA or algorithms invalidate only dependent stages; predictor chunks require exact task identity. |
| Oversized task gate | Pass | Estimate is written before task materialization; ordinary task TSV and prediction are skipped unless force mode is enabled. |
| HLA support | Pass | Runtime predictor catalogs replace specific-allele hardcoding; unsupported tasks are retained as skipped rows. |
| MT/WT contract | Pass | Missense/in-frame WT pairing is retained; insertion windows without same-position WT are marked `WT_not_evaluable`; frameshift WT and dependent metrics are blank, with event-level WT context written to a sidecar. |
| IC50 and EL separation | Pass | EL/presentation results enter percentile fields only and cannot alter Best/Median IC50 or fold change. |
| Error visibility | Pass | Skipped and failed tasks remain explicit in normalized outputs and summary status counts. |
| Output schema | Pass | Mutation uses the established 111-column profile; non-mutation uses the common 45-column profile. |

Static inspection of the current downstream `afterNeoantigen_v2.3` notebook
found 99 direct references to pVACseq input columns. All 99 are present in the
native 111-column reference merged table. Other candidate strings identified by the
scan are fields created later by population-frequency, expression, or
immunogenicity annotation steps rather than missing binding-input columns.

## Existing smoke evidence

The previously completed non-mutation smoke tests remain valid after the
shared-adapter changes:

- cryptic: 150 rows, all MHCnuggets IC50 and percentile fields populated;
- microbial: 78 rows, all MHCnuggets IC50 and percentile fields populated;
- cryptic schema matches its historical pVACbind table;
- microbial retains all historical columns in order and adds the common
  MHCflurry fields.

Smoke artifacts are maintained outside the repository.

## Deferred validation

- full mutation, cryptic, and microbial end-to-end prediction;
- multi-sample throughput, memory, and worker-count benchmarking;
- forced full prediction for microbial samples exceeding the scale threshold.
