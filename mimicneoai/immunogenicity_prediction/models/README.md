# Immunogenicity Model Payloads

This directory defines the on-disk contract for the frozen source-specific
immunogenicity checkpoints. Model payloads are intentionally excluded from Git
and must be restored from the separately distributed MimicNeoAI model bundle.

## Required Layout

```text
models/default/
├── microbial/
│   └── model.pth
├── mutation_derived/
│   └── model.pth
└── cryptic/
    ├── member_01.pth
    ├── member_02.pth
    ├── ...
    └── member_10.pth
```

The microbial and mutation-derived sources each use one checkpoint. Cryptic
inference requires all ten members and reports their mean score. Missing or
inconsistent ensemble members are runtime errors; the ensemble must not be
silently reduced.

## Checkpoint Identities

| Relative path | SHA256 |
|---|---|
| `microbial/model.pth` | `0690b4a3ebf6d79ca7e8186e694b68651e44f1e1ea30d01ae84d00c5807fa9e3` |
| `mutation_derived/model.pth` | `6a55541a327b680013e3afb7cccdb14bdd9a7b0b8d100595492e52267a5ea013` |
| `cryptic/member_01.pth` | `7b9cc914610ca85424ea27ae783a7c1b233506023d8f02d214c5b060b5b41d18` |
| `cryptic/member_02.pth` | `fecc2bfe64f2eb6906cd06e1d61956bd143a91af814014ee17d07248410d96fd` |
| `cryptic/member_03.pth` | `6662dfc5b0c54f9151e83bdd77afbdcb5a410cdb22cc9df9ef846d5c7235b58e` |
| `cryptic/member_04.pth` | `21db89a6d2e4c8e86959222f04a1d574e9024beb9108890c68e894ea2bdde899` |
| `cryptic/member_05.pth` | `7c059e64ba1888409622f9d50f03053be813f484bc93287fedc13725136312dc` |
| `cryptic/member_06.pth` | `bb7a9e313a663087fa49c42e05cc76f969dac4bce84309645e30362707a8301c` |
| `cryptic/member_07.pth` | `8ddbe80dd82c3fa073f4e8214998895d87fbb4f6059d57d58d76b9e2f59b6928` |
| `cryptic/member_08.pth` | `3b6686a6ee60843cbc8a8c48674160aa22f5ce66c20595d50a23fb68e51fd7f4` |
| `cryptic/member_09.pth` | `258cb40fd82da568a18e25a787c9d998123d74867759df452fd36e1f31c114c9` |
| `cryptic/member_10.pth` | `1fceda96ee7146c918ebb855e2134704468d4eb30b8497cda16b785b7588e8b1` |

Verify a restored payload from the model root:

```bash
cd mimicneoai/immunogenicity_prediction/models/default
sha256sum microbial/model.pth mutation_derived/model.pth cryptic/member_*.pth
```

Every digest must match this table before the payload is used for a formal
analysis. A different digest is a different model artifact, even if the file
name is unchanged.

## HLA Resource Contract

The checkpoints require the matching NetMHC-derived HLA pseudosequence CSVs
under:

```text
mimicneoai/immunogenicity_prediction/resources/hla_pseudoseq/local/
```

See the [HLA pseudosequence README](../resources/hla_pseudoseq/README.md) for
the expected layout and preparation command. Missing HLA alleles are reported
as not evaluable; they must not be silently replaced with a nearby allele.

## Runtime Selection

The default runtime resolves this directory automatically. A deployment may
store the payload elsewhere by setting one of:

1. `others.immunogenicity_model_root` in the pipeline configuration;
2. `MIMICNEOAI_IMMUNOGENICITY_MODEL_ROOT` in the environment;
3. `path.common.IMMUNOGENICITY.MODEL_ROOT` in `paths.yaml`.

The configured root must contain the `microbial`, `mutation_derived`, and
`cryptic` subdirectories shown above.

## Interpretation and Redistribution

These checkpoints are frozen inference artifacts. Do not retrain, replace, or
combine members during a production run. Scores from the three antigen-source
models are not calibrated for direct cross-source comparison.

Model redistribution and the HLA resource files are governed separately from
the Apache-licensed source code. Confirm the terms of the supplied model and
NetMHC resource bundles before redistribution.
