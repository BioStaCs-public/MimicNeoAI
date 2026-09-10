# Mutation-Derived Antigen Pipeline

The mutation-derived pipeline identifies somatic protein-altering events from
matched tumor-normal WES, reconstructs event-level mutant and wild-type protein
contexts, and predicts HLA binding for mutation-covering peptides.

## Workflow

```text
Matched tumor-normal WES FASTQ
  -> read QC
  -> BWA alignment, duplicate marking, and BQSR
  -> Mutect2, Strelka2, and VarDict somatic calling
  -> caller reconciliation and PASS filtering
  -> VEP annotation and transcript mismatch filtering
  -> HLA typing
  -> event-level MT/WT protein reconstruction
  -> mutation-covering HLA-I 8-11mer and HLA-II 13-17mer tasks
  -> native binding prediction
  -> optional source-specific immunogenicity prediction
```

The workflow preserves each variant event, transcript consequence, mutant (MT)
context, and its matched wild-type (WT) control. Frameshift events retain an
explicit WT context control rather than being assigned an artificial
position-matched WT epitope.

## Requirements

Install MimicNeoAI from the repository root:

```bash
python -m pip install -e .
```

Configure tool and reference paths in
[`configures/paths.yaml`](../configures/paths.yaml).

| Tool | Validated version | Role |
|---|---:|---|
| `fastp` | 0.22.0 | FASTQ quality control |
| `bwa` | 0.7.17 | WES alignment |
| `samtools` | 1.5 | Alignment processing and indexing |
| Java | 17 | GATK execution |
| GATK | 4.6.0.0 | MarkDuplicates, BQSR, and Mutect2 |
| BCFtools / HTSlib | 1.11+ | VCF normalization, compression, and indexing |
| Manta / Strelka2 | deployment-specific | Somatic calling workflow |
| VarDict Java and R helpers | deployment-specific | Somatic calling workflow |
| VEP and plugins | release 110 deployment | Variant consequence annotation |
| pVACtools | 4.2.1 container | Protein source reconstruction and legacy backend |
| `bowtie2` | 2.4.1 | HLA-HD preprocessing |
| HLA-HD | 1.7.0 | HLA typing |
| Native binding predictors | see `paths.yaml` | Peptide-HLA binding prediction |

Production use also requires a GRCh38 reference, exome target intervals,
known-sites resources, germline population resource, panel of normals, and VEP
cache/plugins. See the [database guide](../configures/Database_and_Paths.md).

## Input Layout

Each tumor and normal sample has its own FASTQ directory:

```text
<input_dir>/
├── Tumor_1/
│   ├── Tumor_1.R1.fq.gz
│   └── Tumor_1.R2.fq.gz
└── Normal_1/
    ├── Normal_1.R1.fq.gz
    └── Normal_1.R2.fq.gz
```

The current pipeline requires matched pairs:

```yaml
others:
  tumor_with_matched_normal: true

samples:
  - Tumor_1,Normal_1
```

## Configuration

Start from the canonical template:

```bash
cp mimicneoai/configures/mutation_derived_configure.yaml mutation.run.yaml
```

Review the project paths, resources, target BED, concurrency, and prediction
policy:

```yaml
path:
  tmp_dir: /path/to/tmp
  input_dir: /path/to/fastq
  output_dir: /path/to/results

args:
  thread: 30
  pool_size: 1
  mem: 128G

others:
  tumor_with_matched_normal: true
  bed_file: /path/to/exome_targets.bed
  binding_prediction_backend: mimicneoai
  binding_prediction_preset: fast
  binding_prediction_start_from: source_prep
  run_immunogenicity_prediction: false

samples:
  - Tumor_1,Normal_1
```

`args.thread` is allocated per sample pair. `args.pool_size` pairs can run
concurrently, so memory and CPU limits must be evaluated as a product of both
settings.

### Resume from Epitope Tasks

When validated event reconstruction and epitope tasks already exist, the
workflow can resume directly at binding:

```yaml
others:
  binding_prediction_start_from: epitope_tasks
```

This mode consumes the existing `02_epitope_tasks` contract. It does not infer
or repair missing event provenance. Use it only when the task manifest and
event-level outputs correspond to the VCF intended for the run.

## Run

```bash
mimicneoai mutation-derived \
  -c mutation.run.yaml \
  -p mimicneoai/configures/paths.yaml
```

The equivalent module entry point is:

```bash
python -m mimicneoai.mutation_derived_pipeline.mutation_derived \
  -c mutation.run.yaml \
  -p mimicneoai/configures/paths.yaml
```

## Outputs

```text
<output_dir>/Mutation-derived/<tumor_sample>/
├── 00.QC
├── 01.alignment
├── 02.markdup
├── 03.bqsr
├── 04.variants_calling
├── 05.annotation
├── 06.hlatyping
├── 07.binding_prediction_mimicneoai
└── 08.immunogenicity_prediction_mimicneoai
```

The binding stage is divided into explicit evidence layers:

```text
07.binding_prediction_mimicneoai/
├── 01_pvactools_sources
├── 02_epitope_tasks
├── 03_binding_predictions
└── 04_merged_epitopes
```

Important products include:

- `01_pvactools_sources/*.protein.flank25.wt_mt.fasta`: source MT/WT protein
  contexts;
- `01_pvactools_sources/*.source_inputs.manifest.json`: source identities;
- `02_epitope_tasks/epitope_windows.tsv`: mutation-covering MT windows,
  matched-WT sequences, and control-evaluation status without counting WT as a
  mutation candidate;
- `02_epitope_tasks/event_qc.tsv`: the disposition of each annotated event;
- `02_epitope_tasks/excluded_or_failed_events.tsv`: explicit no-window and
  conversion failures;
- `02_epitope_tasks/wt_context_controls.tsv`: frameshift WT context controls;
- `04_merged_epitopes/*.merged.all_epitopes.tsv`: peptide-HLA binding results
  with event metadata.

Events that cannot form a valid mutation-covering 8-11mer or 13-17mer are
reported as `no_window`. They are not silently discarded.

## Binding Policy

The default backend is `mimicneoai` with the `fast` preset. Stage 1 uses the MT
peptide to decide whether an event proceeds. When an MT peptide enters Stage 2,
its matched WT control is carried into Stage 2 regardless of the WT Stage 1
result. This preserves independent MT and WT prediction fields and downstream
fold-change calculations.

The default Stage 2 algorithm set comprises MHCflurry, MHCflurryEL,
MHCnuggetsI/II, NetMHCpan/NetMHCpanEL, and
NetMHCIIpan/NetMHCIIpanEL. NNalign is not part of the packaged `fast` preset.
Select `full` for one-stage multi-algorithm prediction, or explicitly select
the legacy `pvactools` backend when required for comparison with an older run.

Stage 1 is a routing decision. A Stage 1 failure, unsupported allele, missing
prediction, or tool failure must not be interpreted as a final non-binding
classification.

## Immunogenicity

Set `run_immunogenicity_prediction: true` to score MT and matched-WT
peptide-HLA rows after binding. Inference is deduplicated by peptide-HLA key and
merged back to event-level MT/WT records. WT scores are retained for control and
comparison, while formal mutation-derived prioritization is based on MT rows.

Model weights and HLA pseudosequence resources are separate runtime payloads.
See the [immunogenicity guide](../immunogenicity_prediction/README.md).

## Resume and Provenance

The event reconstruction and native prediction layers retain manifests and
event-level QC tables. Resume is valid only when the relevant inputs,
configuration, and outputs match the recorded contract. Existing files alone
do not prove that they were generated from the current PASS VCF or policy.

For a changed VCF, transcript annotation, reference, or binding policy, use a
new output directory or archive the previous stage. Do not merge products from
different source manifests.

## Interpretation

The pipeline reports computationally reconstructed somatic mutation-derived
peptides and their HLA predictions. These results do not establish natural HLA
presentation, T-cell recognition, or clinical immunogenicity without
independent evidence.
