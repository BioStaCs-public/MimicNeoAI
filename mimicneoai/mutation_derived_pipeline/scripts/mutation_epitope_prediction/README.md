# Mutation Epitope Prediction

Experimental workflow for mutation-derived epitope generation and binding
prediction.

The current production pipeline runs pVACseq end to end under
`07.binding_prediction`. This package is a side-channel implementation used to
test a MimicNeoAI-native prediction workflow:

1. Use external pVACtools only for VCF-derived mutation annotation and WT/MT
   protein sequence generation.
2. Build stable variant and transcript identifiers independent of pVACseq
   chunk-local `Index` values.
3. Generate mutation-covering epitope windows, extended peptides, and
   de-duplicated peptide-HLA-algorithm prediction tasks.
4. Run binding predictors through MimicNeoAI adapters and merge prediction
   outputs back to variant, transcript, and expression
   evidence.

This package must not import or vendor pVACtools source code. pVACtools should
be invoked as an external tool, typically through the validated Apptainer image.

Development and validation output directory layout:

```text
07.binding_prediction_mimicneoai/
  00_input_vcf/
  01_pvactools_sources/
  02_epitope_tasks/
  03_binding_predictions/
  04_merged_epitopes/
  archive/
```

CLI steps:

```text
00_prepare_pvacseq_sources.py
01_build_epitope_tasks.py
02_split_binding_tasks.py
run_mimicneoai_binding_prediction.py
02_merge_binding_predictions.py
```

The one-command wrapper is the preferred entry point during validation:

```text
run_mimicneoai_binding_prediction.py \
  -s <sample> \
  --input-vcf <vep.vcf.gz> \
  --hla-file <HLA-HD result.txt> \
  --pvactools-sif <pvactools.sif> \
  -o <sample>/07.binding_prediction_mimicneoai
```

When `01_pvactools_sources/` and `02_epitope_tasks/` have already been
reviewed and frozen, resume directly from binding with:

```text
run_mimicneoai_binding_prediction.py \
  -s <sample> \
  --input-vcf <vep.vcf.gz> \
  --hla-file <HLA-HD result.txt> \
  --pvactools-sif <pvactools.sif> \
  -o <sample>/07.binding_prediction_mimicneoai \
  --preset fast \
  --start-from epitope_tasks
```

The wrapper records pVACtools and predictor metadata, peptide lengths, worker
counts, MT/WT task counts, merged row counts, and the frameshift WT-field rule
in `<sample>.binding_prediction_mimicneoai.summary.json`.

`02_epitope_tasks/epitope_windows.tsv` includes explicit WT-control status:

- `matched_WT`: same-position, same-length WT epitope is available and can be
  used for WT prediction and fold-change.
- `WT_not_evaluable`: an MT window is valid but has no same-position,
  same-length WT epitope, for example an insertion-derived window that falls
  outside the WT fragment coordinate range.
- `WT_context_control`: frameshift windows have no conventional matched-WT
  epitope; their event-level WT context is written to
  `02_epitope_tasks/wt_context_controls.tsv` for provenance and control
  reporting, but is not scheduled as a WT binding task.

Default predictors intentionally exclude legacy-only default runs for
SMM/SMMPMBEC/PickPocket/NetMHC. pVACseq-compatible output columns for those
methods remain available in the merged table schema and are left blank unless
the corresponding algorithms are explicitly added and supported.
