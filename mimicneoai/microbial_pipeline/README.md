# Microbial Antigen Pipeline

The microbial pipeline identifies RNA- or DNA-supported microbial protein
fragments, constructs HLA-I and HLA-II peptide candidates, and optionally runs
binding and source-specific immunogenicity prediction. It supports both
single-sample discovery and matched tumor-normal analysis.

## Workflow

```text
FASTQ
  -> read QC
  -> host depletion (GRCh38 and T2T)
  -> vector decontamination
  -> PathSeq taxonomic profiling
  -> BLASTX or DIAMOND protein-hit identification
  -> protein-hit QC
  -> matched-normal peptide subtraction (paired mode)
  -> contaminant blacklist QC
  -> HLA-I 8-11mer and HLA-II 13-17mer peptide Core
  -> HLA typing
  -> optional binding prediction
  -> optional immunogenicity prediction
```

In paired mode, tumor and normal are processed with the same upstream rules.
Exact peptide sequences observed in the matched normal are removed before the
tumor peptide Core is finalized. Taxon or protein overlap alone is retained as
annotation and is not used as a substitute for exact peptide subtraction.

## Requirements

Install MimicNeoAI from the repository root:

```bash
python -m pip install -e .
```

The following tools and resources must also be available. Paths to executables,
containers, and reference data are configured in
[`configures/paths.yaml`](../configures/paths.yaml).

| Tool | Validated version | Role |
|---|---:|---|
| `fastp` | 0.22.0 | FASTQ quality control |
| `bwa` | 0.7.17 | Host and vector alignment |
| `samtools` | 1.5 | Alignment processing |
| Java | 17 | GATK execution |
| GATK | 4.6.0.0 | PathSeq profiling |
| BLASTX | 2.15.0+ | Protein-fragment search option |
| DIAMOND | deployment-specific | Faster protein-fragment search option |
| `bowtie2` | 2.4.1 | HLA-HD preprocessing |
| HLA-HD | 1.7.0 | HLA typing |
| Native binding predictors | see `paths.yaml` | Peptide-HLA binding prediction |
| Apptainer and pVACtools | 1.4.2 / 4.2.1 | Legacy pVACbind backend |

The microbial reference bundle, PathSeq resources, protein-search database,
protein-to-taxon catalog, and formal contaminant blacklist are required for a
production paired run. See the [database guide](../configures/Database_and_Paths.md).

## Input Layout

Each sample is a directory containing paired FASTQ files:

```text
<input_dir>/
├── Tumor_1/
│   ├── Tumor_1.R1.fq.gz
│   └── Tumor_1.R2.fq.gz
└── Normal_1/
    ├── Normal_1.R1.fq.gz
    └── Normal_1.R2.fq.gz
```

Single-sample mode lists one sample per entry. Matched-normal mode uses exactly
one `Tumor,Normal` pair per entry:

```yaml
others:
  tumor_with_matched_normal: true
  run_paired_core_qc: true
  run_binding_prediction: false

samples:
  - Tumor_1,Normal_1
```

Pair parsing is strict. A pair must contain two different, non-empty sample
identifiers, and a sample cannot have conflicting roles across pairs.

## Configuration

Start from the canonical template:

```bash
cp mimicneoai/configures/microbial_configure.yaml microbial.run.yaml
```

At minimum, review:

```yaml
path:
  tmp_dir: /path/to/tmp
  input_dir: /path/to/fastq
  output_dir: /path/to/results

args:
  thread: 20
  pool_size: 2

others:
  microbial_peptide_search_engine: diamond  # blastx | diamond
  tumor_with_matched_normal: true
  run_paired_core_qc: true
  run_binding_prediction: true
  binding_prediction_backend: mimicneoai
  binding_prediction_preset: fast
  run_immunogenicity_prediction: false

samples:
  - Tumor_1,Normal_1
```

`args.thread` is the thread budget per concurrently processed task, while
`args.pool_size` controls task-level concurrency. Their product should not
exceed the CPUs and memory available to the deployment.

### Protein-hit QC

The normalized protein-hit contract requires:

- percent identity equal to 100;
- E-value at or below `1e-5`;
- query coverage at or above 90%;
- a canonical amino-acid sequence after removal of at most one terminal stop;
- no internal stop, gap, `X`, or other noncanonical residue.

Missing query coverage fails closed. BLASTX and DIAMOND outputs are normalized
to the same `protein_hits.filtered.tsv` schema before Core construction.

### Paired Core

`others.run_paired_core_qc: true` builds the peptide Core independently of
binding. Formal paired runs require the contaminant taxon blacklist configured
at `database.microbial.BLACKLISTS.CONTAMINANT_TAXIDS`. Setting
`allow_missing_blacklist: true` is exploratory and is recorded as such in the
manifest.

The early scale guard
`paired_core_max_estimated_peptide_windows` is a computational protection. A
sample that exceeds it is recorded as `scale_gate_skipped`; it is not classified
as biologically negative. Set the value to `0` only for an intentional oversized
run.

### Candidate Selection

`candidate_selection.mode: all` retains the complete QC-passed peptide Core and
is the generic default. `ranked_cap` ranks microbial source groups and limits
unique peptide sequences independently for HLA-I and HLA-II. Candidates outside
the cap are deferred for computation; they are not failed QC records or
non-binders.

## Run

```bash
mimicneoai microbial \
  -c microbial.run.yaml \
  -p mimicneoai/configures/paths.yaml
```

The equivalent module entry point is:

```bash
python -m mimicneoai.microbial_pipeline.microbial \
  -c microbial.run.yaml \
  -p mimicneoai/configures/paths.yaml
```

## Outputs

Results are sample-centered:

```text
<output_dir>/Microbial/<tumor_sample>/
├── 00.QC
├── 01.HostSequencesRemovingStep1
├── 02.HostSequencesRemovingStep2
├── 03.VectorContaminationRemoving
├── 04.MicrobialTaxaQuantificationStep1
├── 05.MicrobialTaxaQuantificationStep2
├── 06.MicrobialPeptidesIdentification
├── 06b.MicrobialProteinCoreQC_v1.0
├── 07.HlaTyping
├── 08.MicrobialPeptidesBindingPrediction_mimicneoai
└── 09.ImmunogenicityPrediction_mimicneoai
```

### Protein-hit products

`06.MicrobialPeptidesIdentification` retains legacy search outputs and writes a
normalized interface:

- `<sample>.protein_hits.filtered.tsv`: protein hits eligible for downstream QC;
- `<sample>.protein_hits.excluded.tsv`: excluded records with explicit reasons;
- `<sample>.protein_hits.qc_summary.tsv`: stagewise counts;
- `<sample>.peptide.fasta`: legacy parent-fragment FASTA.

The legacy FASTA is not a valid paired-mode binding input.

### Paired peptide Core

`06b.MicrobialProteinCoreQC_v1.0` contains:

- `microbial_parent_core.tsv`: retained parent-level evidence;
- `microbial_parent_excluded.tsv`: parent-level exclusions;
- `microbial_peptide_core.tsv`: final unique tumor peptide Core;
- `microbial_peptide_core_hla_i.fasta`: HLA-I 8-11mer candidates;
- `microbial_peptide_core_hla_ii.fasta`: HLA-II 13-17mer candidates;
- `microbial_peptide_core.fasta`: combined peptide-Core binding input;
- `microbial_peptide_parent_map.tsv`: peptide-to-parent provenance;
- `matched_normal_peptide.tsv`: exact matched-normal exclusions;
- `stagewise_qc.tsv` and `run_manifest.json`: counts and run identity.

The FASTA files already contain tiled peptides. Native binding uses
`input_mode=peptide-core` and must not tile them a second time.

## Binding and Immunogenicity

The default native backend uses `binding_prediction_preset: fast`. Stage 1
routes candidate peptide-HLA pairs by EL rank, and Stage 2 applies the configured
multi-algorithm predictor set. Use `full` to bypass Stage 1 routing. The legacy
pVACbind backend must be selected explicitly.

Immunogenicity is disabled by default. Enable
`run_immunogenicity_prediction` only after binding has produced an eligible
peptide-HLA table and the microbial runtime model and HLA pseudosequence
resources are installed. A scale-gated or failed binding run does not produce a
negative immunogenicity call.

## Resume and Provenance

The paired Core validates input, policy, blacklist, code, and output identities
through `run_manifest.json`. Compatible completed outputs can be resumed;
changed inputs or signatures fail closed. Existing outputs should be archived
or written to a new directory when an intentional policy change requires a
rebuild.

Some older upstream discovery stages use stage-specific completion checks.
Inspect their detailed logs before resuming a partial run. Do not infer success
from the presence of one intermediate file.

## Interpretation

The peptide Core represents sequence candidates supported by the configured
microbial search and QC policy. Taxonomic abundance, peptide-HLA binding, model
immunogenicity, mass-spectrometry detection, and natural HLA presentation are
distinct evidence layers. None should be inferred from another without the
corresponding result.
