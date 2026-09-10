# Cryptic Antigen Pipeline

The cryptic pipeline discovers short open reading frames (sORFs) from known
noncoding and novel transcripts, evaluates tumor-associated expression and
genomic evidence, and constructs a traceable peptide Core for HLA binding
prediction.

## Workflow

```text
Tumor RNA FASTQ + optional matched-normal RNA FASTQ
  -> read QC and STAR alignment
  -> known-noncoding and novel transcript discovery
  -> Salmon tumor/control quantification
  -> aberrantly expressed sORF extraction
  -> ORF-to-genome annotation and ORF filtering
  -> Cryptic Core parent QC
       source and expression
       coordinate, mapping, and reference translation
       optional tumor/normal junction support
  -> HLA-I 8-11mer and HLA-II 13-17mer generation
  -> canonical human proteome exact-match QC
  -> deterministic candidate selection with optional external-normal QC/refill
  -> optional binding prediction
  -> optional immunogenicity prediction
```

The primary output is a peptide Core with parent, expression, coordinate,
junction, human-reference, and external-normal evidence. Candidates removed by
QC and candidates deferred only by a computational cap are reported separately.

## Requirements

Install MimicNeoAI from the repository root:

```bash
python -m pip install -e .
```

Configure executables, containers, and references in
[`configures/paths.yaml`](../configures/paths.yaml).

| Tool | Validated version | Role |
|---|---:|---|
| `fastp` | 0.22.0 | FASTQ quality control |
| STAR | 2.5.3a | Tumor and optional control RNA alignment |
| `samtools` | 1.5 | Alignment processing |
| StringTie | 3.0.1 | Novel transcript assembly |
| GffCompare | 0.12.10 | Transcript classification |
| GffRead | 0.12.7 | Transcript sequence extraction |
| Minimap2 | 2.30-r1287 | ORF/contig genomic alignment |
| BCFtools | 1.11 | Known-branch RNA variant processing |
| TransDecoder | 5.5.0 | ORF discovery |
| Salmon | 1.10.0 | Tumor/control expression quantification |
| `bowtie2` | 2.4.1 | HLA-HD preprocessing |
| HLA-HD | 1.7.0 | HLA typing |
| Apptainer | 1.4.2 | Container execution |
| Trinity | 2.15.2 container or compatible native install | Novel transcript assembly |
| Native binding predictors | see `paths.yaml` | Peptide-HLA binding prediction |
| pVACtools | 4.2.1 container | Legacy pVACbind backend |

Production Core QC also requires the configured GRCh38 reference, GENCODE
annotation, reviewed canonical human proteome, and any enabled external-normal
resources. See the [database guide](../configures/Database_and_Paths.md).

## Input Layout

FASTQ files are grouped by sample:

```text
<input_dir>/
├── Tumor_1/
│   ├── Tumor_1.R1.fq.gz
│   └── Tumor_1.R2.fq.gz
└── Normal_1/
    ├── Normal_1.R1.fq.gz
    └── Normal_1.R2.fq.gz
```

Tumor/control analysis uses:

```yaml
samples:
  - Tumor_1,Normal_1
```

The tumor drives cryptic discovery, HLA typing, binding, and immunogenicity.
The matched normal contributes control expression and, when enabled, an
independent STAR alignment for junction annotation. It does not enter the
normal sample into downstream cryptic discovery.

## Configuration

Start from the canonical template:

```bash
cp mimicneoai/configures/cryptic_configure.yaml cryptic.run.yaml
```

Review the input paths, resource allocation, stage switches, expression policy,
and Core resources:

```yaml
path:
  tmp_dir: /path/to/tmp
  input_dir: /path/to/fastq
  output_dir: /path/to/results

args:
  threads: 30
  pool_size: 1

others:
  alignment_control: false
  cryptic_core_qc: true
  cryptic_external_normal_qc: false
  min_tpm_tumor: 5.0
  max_tpm_ctrl: 0.5
  min_log2fc: 4.0
  human_reference_proteome_fasta: /path/to/canonical_human_proteome.fasta
  allow_missing_human_reference: false
  binding_prediction_backend: mimicneoai
  binding_prediction_preset: fast
  run_immunogenicity_prediction: false

candidate_selection:
  mode: all

samples:
  - Tumor_1,Normal_1
```

`args.threads` is the thread budget per sample task, and `args.pool_size`
controls sample-level concurrency. Account for memory-intensive Trinity and
alignment stages when setting both values.

### Cryptic Core Policies

The generic template enables Core QC and defaults to
`cryptic_core_qc_v1.0`. The v1.1 policy adds production coordinate, translation,
and junction contracts. The Core accepts only configured novel/noncoding
sources, re-evaluates the expression thresholds inside the Core stage, records
excluded parents, and generates already tiled peptide FASTA files.

A missing formal human reference fails closed. Setting
`allow_missing_human_reference: true` creates an exploratory result and must not
be routed into a formal binding analysis.

### Junction QC

Junction QC is optional in the generic template. With
`junction_qc.enabled: true`, policy `junction_qc_v1.0` requires every required
parent junction to have at least two unique tumor split reads. It also reports
sensitivity at 1, 2, 3, and 5 reads. Intronless parents are retained as
`not_applicable`; matched-normal junction observations are annotations, not hard
exclusions.

Junction analysis consumes a frozen tumor/control STAR pair table. Provide
`junction_qc.star_pair_inputs`, or enable both control alignment and automatic
provenance freezing:

```yaml
others:
  alignment_control: true

junction_qc:
  enabled: true
  auto_freeze_star_provenance: true
```

STAR completion requires the BAM, `SJ.out.tab`, `Log.final.out`, and `Log.out`.
Resume additionally validates FASTQ, index, critical STAR parameters, role, and
output identities. A partial or incompatible STAR directory fails closed rather
than being overwritten.

### External-Normal QC

`others.cryptic_external_normal_qc` is disabled by default because its resource
bundle is project-specific. Policy v1.0 evaluates exact peptide matches in
normal smORF and HLA-ligand resources. Policy v1.1 adds strand-aware genomic
coordinate and reading-frame evidence for trusted parent alignments.

When enabled, resource files and hashes must match the frozen external-normal
manifest. The final binding input is then produced by step 08c. Binding cannot
fall back to an earlier FASTA if the 08c contract is missing or invalid.

### Candidate Selection

`candidate_selection.mode: all` retains all QC-passed unique peptides. In
`ranked_cap` mode, all parents first receive the same source, expression,
coordinate, translation, and junction QC. Eligible parents are then ranked
deterministically; generated peptides receive the same human-reference and
external-normal checks before unique caps are applied separately to HLA-I and
HLA-II.

If external-normal QC removes an initially selected peptide, step 08c continues
through the ranked parent stream and applies the complete evidence contract to
each refill candidate. Cap-external candidates are `deferred` or
`not_selected_due_to_analysis_cap`; these are computational states, not QC
failures or binding-negative calls.

### RNA Variant Boundary

`rna_variant_editing_qc.enabled` remains `false` in the current strict main
configuration. Reference-translation mismatches and RNA-dependent parents are
retained in provisional or excluded sidecars and do not enter the strict
primary Core.

The optional RNA variant helper can assess normalized VCF and read-level
evidence for exploratory sequence reconstruction. REDIportal is not a required
dependency of the main pipeline, and RNA-only differences must not be described
as somatic, non-editing, or tumor-specific mutations.

## Run

```bash
mimicneoai cryptic \
  -c cryptic.run.yaml \
  -p mimicneoai/configures/paths.yaml
```

The equivalent module entry point is:

```bash
python -m mimicneoai.cryptic_pipeline.cryptic \
  -c cryptic.run.yaml \
  -p mimicneoai/configures/paths.yaml
```

## Outputs

```text
<output_dir>/Cryptic/<tumor_sample>/
├── 00-clean
├── 01-star
├── 02-known
├── 03-novel
├── 023-shared
├── 04-salmon_quant
├── 05-hla_typing
├── 06-aeSEPs
├── 07-orf_genome_annotation
├── 08-orf_filter
├── 08b-cryptic_core_qc
├── 08c-external_normal_qc
├── 09-hla_binding_pred_mimicneoai
└── 10-immunogenicity_prediction_mimicneoai
```

Key stages are:

- `06-aeSEPs`: aberrantly expressed sORF proteins before formal Core QC;
- `07-orf_genome_annotation`: ORF/CDS genomic alignments and annotations;
- `08-orf_filter`: parent proteins retained by the ORF-level policy;
- `08b-cryptic_core_qc`: parent Core, ranked parent stream, unique HLA-I/HLA-II
  peptide Core, exclusions, evidence sidecars, stagewise counts, and manifest;
- `08c-external_normal_qc`: tumor-restricted final Core, refill audit, final
  binding FASTA, and `cryptic_final_peptide_parent_sidecar.tsv`;
- `09-hla_binding_pred_mimicneoai`: native peptide-Core binding results;
- `10-immunogenicity_prediction_mimicneoai`: optional cryptic ensemble scores.

The final peptide-parent sidecar is a pre-binding evidence table. It records MHC
class and all supporting parent occurrences but does not invent an HLA allele.
Allele-specific evidence is joined after binding.

## Binding and Immunogenicity

The default native backend uses the `fast` preset. Use `full` for one-stage
multi-algorithm prediction. The strict Core is accepted only in
`input_mode=peptide-core`, which prevents a second round of peptide tiling. The
legacy pVACbind route must be selected explicitly and is not interchangeable
with the strict peptide-Core contract.

When 08c is enabled, binding requires a formal complete manifest,
`binding_eligible=true`, and an exact size/SHA256 match for the final FASTA. A
sample skipped by the binding task-scale guard is labeled
`skipped_due_to_scale`, not non-binding.

Immunogenicity is disabled by default. The cryptic model is a fixed ten-member
ensemble, and the formal score is the member mean. Install the model payload and
HLA pseudosequence resources described in the
[immunogenicity guide](../immunogenicity_prediction/README.md) before enabling
this stage.

## Resume and Provenance

Core, junction, external-normal, binding, and immunogenicity stages validate
their relevant input, code, configuration, resource, and output signatures.
Compatible outputs, including valid zero-candidate FASTA files, can be resumed.
An incompatible manifest or a changed output fails closed.

Older discovery stages retain stage-specific completion checks. For a policy or
reference change, use a new output directory or archive the prior stage rather
than editing an existing formal result in place.

## Interpretation

A cryptic Core peptide is supported by the configured RNA expression and
sequence-QC policy. Binding, immunogenicity, natural HLA presentation, and
T-cell recognition remain separate evidence layers. Public RNA-seq results
support RNA-level candidate discovery only unless independent genomic,
proteomic, or functional evidence is available.
