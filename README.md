# MimicNeoAI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.9-3776AB.svg)](pyproject.toml)
[![Development status](https://img.shields.io/badge/status-research%20beta-orange.svg)](#project-status)

MimicNeoAI is a research toolkit for constructing and evaluating tumor antigen
candidates from sequencing data. It provides separate workflows for microbial,
sORF-encoded cryptic, and mutation-derived antigens, followed by HLA-binding
prediction, source-specific immunogenicity assessment, and cross-source peptide
sequence matching. Pairs for which both peptides are predicted to bind the same
patient HLA-I allele are reported as predicted mimicry candidates.

The software preserves the evidence and exclusion status of each candidate.
Unsupported HLA alleles, failed predictors, scale-gated samples, and candidates
that do not pass a routing threshold are not silently labeled as non-binders.

## Project Status

MimicNeoAI is research beta software. Install it from source and record the Git
commit used for each analysis. A portable container image is planned but is not
currently distributed as part of the public release.

## Workflows

| Workflow | Primary input | Main output | Documentation |
|---|---|---|---|
| Microbial antigen | Tumor RNA/WGS FASTQ, optionally paired with matched normal | Matched-normal-depleted microbial peptide Core | [Microbial pipeline](mimicneoai/microbial_pipeline/README.md) |
| Cryptic antigen | Tumor RNA FASTQ, preferably with matched-normal RNA | Expression-, ORF-, mapping-, and junction-supported cryptic peptide Core | [Cryptic pipeline](mimicneoai/cryptic_pipeline/README.md) |
| Mutation-derived antigen | Matched tumor-normal WES FASTQ | Event-level mutant peptides with matched-WT controls | [Mutation-derived pipeline](mimicneoai/mutation_derived_pipeline/README.md) |
| Immunogenicity prediction | Peptide-HLA table | Source-specific immunogenicity scores and input QC | [Immunogenicity prediction](mimicneoai/immunogenicity_prediction/README.md) |
| Molecular mimicry | Quality-controlled HLA-I candidate tables from two or more antigen sources, with optional binding results | Within-patient sequence-similar pairs and predicted mimicry candidates | [Molecular mimicry](mimicneoai/mimicry/README.md) |

The antigen workflows use HLA-I peptides of 8–11 amino acids and HLA-II
peptides of 13–17 amino acids in the packaged configuration. Molecular
mimicry analysis is currently restricted to 8–11-aa HLA-I candidates. The native
binding backend is documented separately in the
[binding prediction guide](mimicneoai/functions/binding_prediction/README.md).

## Installation

MimicNeoAI currently supports Linux and is developed and validated primarily on
Ubuntu 22.04 with Python 3.10. The package metadata requires Python 3.9 or
newer. The sequencing workflows also require command-line bioinformatics
software and reference databases that are not installed by `pip`.

```bash
git clone https://github.com/TaoChen-Code/MimicNeoAI.git
cd MimicNeoAI

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

mimicneoai --help
```

Install immunogenicity inference dependencies with:

```bash
python -m pip install -e '.[immunogenicity]'
```

PyTorch must match the host CPU or CUDA runtime. For isolated CPU and CUDA 11.8
environments, see the [immunogenicity runtime guide](mimicneoai/immunogenicity_prediction/README.md#runtime-environment).

## References and External Software

Each workflow calls established third-party tools such as `fastp`, `samtools`,
HLA-HD, GATK, VEP, STAR, Salmon, NetMHCpan, NetMHCIIpan, MHCflurry, and
MHCnuggets. The exact requirements differ by workflow and are listed in the
corresponding pipeline README.

The optional database helper downloads the MimicNeoAI reference bundle:

```bash
mimicneoai download_database --target-dir /path/to/MimicNeoAI_database
```

Review the [database and path guide](mimicneoai/configures/Database_and_Paths.md)
before running an analysis. Some external predictors, HLA pseudosequence files,
reference datasets, and model payloads have their own licenses or access terms
and are therefore not distributed in this Git repository.

## Quick Start

Copy a workflow configuration and replace the example paths and sample names:

```bash
cp mimicneoai/configures/microbial_configure.yaml microbial.run.yaml
cp mimicneoai/configures/cryptic_configure.yaml cryptic.run.yaml
cp mimicneoai/configures/mutation_derived_configure.yaml mutation.run.yaml
cp mimicneoai/configures/mimicry_configure.yaml mimicry.run.yaml
```

Run a workflow with its analysis configuration and the shared tool/reference
configuration:

```bash
mimicneoai microbial \
  -c microbial.run.yaml \
  -p mimicneoai/configures/paths.yaml

mimicneoai cryptic \
  -c cryptic.run.yaml \
  -p mimicneoai/configures/paths.yaml

mimicneoai mutation-derived \
  -c mutation.run.yaml \
  -p mimicneoai/configures/paths.yaml

mimicneoai mimicry \
  -c mimicry.run.yaml
```

The example YAML files are templates, not universal production settings.
Reference paths, memory, concurrency, paired-sample mode, and project-specific
QC resources must be reviewed before execution.

## Binding and Immunogenicity

The packaged workflows use the native `mimicneoai` binding backend with the
`fast` preset by default:

- `fast` first routes peptide-HLA pairs with NetMHCpan EL 4.2 or NetMHCIIpan EL
  4.3, then applies the configured multi-algorithm prediction set to candidates
  that pass Stage 1.
- `full` applies the configured multi-algorithm prediction set without Stage 1
  routing.

Stage 1 is a computational routing step, not a final binding or immunogenicity
classification. See the [binding backend documentation](mimicneoai/functions/binding_prediction/README.md)
for statuses, output fields, supported algorithms, and resume behavior.

Immunogenicity prediction is disabled by default. When enabled, MimicNeoAI uses
independently trained source-specific models: one microbial model, one
mutation-derived model, and a fixed ten-member cryptic ensemble. Scores from
different antigen sources are not calibrated for direct cross-source
comparison.

## Reproducibility

Recent Core, binding, and immunogenicity stages write manifests with input,
configuration, code, resource, and output identities. A result is resumed only
when the relevant signatures match. Missing or incompatible formal resources
fail closed where required by the workflow policy.

Older discovery stages retain stage-specific completion checks. Before
resuming a partially completed run, inspect the stage log and output contract.
Do not overwrite or delete a prior analysis merely to force a rerun; use a new
output directory or archive the previous stage first.

Zero-candidate outputs can be valid biological or QC outcomes. They should be
interpreted from the stage manifest and summary rather than from file size
alone.

## Scientific Scope

MimicNeoAI produces computational antigen candidates and explicit evidence
sidecars. A predicted binder is not, by itself, evidence of natural HLA
presentation, T-cell recognition, or clinical immunogenicity. RNA-only support
for a cryptic or microbial peptide should not be described as DNA-confirmed,
somatic, or naturally presented without independent evidence.

The molecular-mimicry module connects candidate repertoires across antigen
sources through within-patient peptide sequence comparison. It reports
sequence-similar pairs and, when binding results are available, identifies
predicted mimicry candidates supported by binding of both peptides to the same
patient HLA-I allele.

## Repository Layout

```text
mimicneoai/
├── configures/                    # Workflow and shared path templates
├── cryptic_pipeline/              # sORF-encoded cryptic antigens
├── microbial_pipeline/            # Microbial antigens
├── mutation_derived_pipeline/     # Somatic mutation-derived antigens
├── mimicry/                        # Cross-source peptide sequence matching
├── functions/binding_prediction/  # Shared native binding backend
├── immunogenicity_prediction/     # Runtime API, models, and benchmarks
└── example/                       # Small module-level examples
```

## Testing

Focused regression suites can be run with the standard library test runner:

```bash
python -m unittest discover -v mimicneoai/cryptic_pipeline/tests
python -m unittest discover -v mimicneoai/microbial_pipeline/tests
python -m unittest discover -v mimicneoai/functions/binding_prediction/tests
python -m unittest discover -v mimicneoai/mimicry/tests
```

Several end-to-end stages require licensed predictors and large reference
bundles, so unit tests do not replace deployment-specific smoke testing.

## Citation

If you use MimicNeoAI, please cite:

> Chen T, Wang W, Zuo X, et al. MimicNeoAI: An integrated pipeline for
> identifying microbial epitopes and mimicry of tumor neoepitopes. *bioRxiv*.
> 2025. doi: [10.1101/2025.06.13.658292](https://doi.org/10.1101/2025.06.13.658292).

## License

MimicNeoAI source code is released under the [Apache License 2.0](LICENSE).
External tools, databases, pretrained models, and derived resources remain
subject to their respective licenses and terms of use.
