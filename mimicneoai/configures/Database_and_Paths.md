# Database and Path Configuration

All three antigen workflows read executable and reference locations from one
shared `paths.yaml` file. The packaged template is:

```text
mimicneoai/configures/paths.yaml
```

The template documents the expected reference layout. It is not evidence that
every resource or licensed predictor is present on a new system.

## Reference Bundle

The optional download helper retrieves and extracts the MimicNeoAI database
bundle:

```bash
mimicneoai download_database --target-dir /path/to/MimicNeoAI_database
```

Without `--target-dir`, the helper uses `mimicneoai/database` inside the source
tree. With a custom target, it creates a link at the expected package location.
Do not replace a non-empty existing database directory without first auditing
its contents.

The bundle does not override the licenses or access conditions of external
software and data. NetMHCpan, NetMHCIIpan, some IEDB components, model payloads,
HLA pseudosequences, and project-specific normal-reference resources may need
to be obtained or configured separately.

## Path Resolution

Pass a paths file explicitly for a reproducible deployment:

```bash
mimicneoai <pipeline> \
  -c /path/to/analysis.yaml \
  -p /path/to/paths.yaml
```

Relative values inside an explicitly supplied `paths.yaml` are resolved against
the directory containing that YAML file. Absolute paths are accepted and are
preferable for shared or containerized deployments.

When `-p` is omitted, MimicNeoAI attempts to load the packaged
`mimicneoai/configures/paths.yaml`. That default is intended as a template and
usually requires a database layout compatible with the repository.

## Configuration Sections

The shared file contains two main groups:

- `path.common`: executables, containers, binding predictors, and optional
  immunogenicity runtime settings;
- `database`: references and policy resources for HLA typing, microbial,
  mutation-derived, and cryptic workflows.

Keep deployment paths in `paths.yaml` and analysis choices in the workflow
configuration. Do not copy host-specific `/workspace`, home-directory, or
temporary paths into the public template.

## Preflight Checklist

Before a production run, verify:

1. every enabled executable is present and runnable;
2. reference FASTA files and indices correspond to the same genome build;
3. HLA-HD script, dictionary, frequency data, and Bowtie2 index belong to one
   installation;
4. binding-predictor executables and model directories are compatible;
5. enabled formal blacklists or external-normal resources match their recorded
   manifests and hashes;
6. the output and temporary filesystems have enough space for the selected
   concurrency;
7. the resolved `paths.yaml` and Git commit are archived with the run.

Pipeline-specific requirements are documented in the
[microbial](../microbial_pipeline/README.md),
[cryptic](../cryptic_pipeline/README.md), and
[mutation-derived](../mutation_derived_pipeline/README.md) guides.
