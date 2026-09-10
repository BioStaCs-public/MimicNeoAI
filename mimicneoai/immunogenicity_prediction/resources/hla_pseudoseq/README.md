# HLA Pseudosequence Resources

The frozen MimicNeoAI immunogenicity checkpoints use HLA pseudosequences
derived from the NetMHCpan and NetMHCIIpan resource bundles. These files are a
required part of the default-model inference contract and are also used during
model development.

The source resources are distributed by DTU under their own access terms. They
are therefore stored under the Git-ignored `local/` directory and are not
redistributed with the MimicNeoAI source code.

## Expected Layout

```text
local/
├── raw/
│   ├── netmhcpan/
│   │   └── MHC_pseudo.dat
│   └── netmhciipan/
│       └── pseudosequence.2023.dat
├── netmhcpan_class1_allele_to_pseudoseq.csv
└── netmhciipan_class2_allele_to_pseudoseq.csv
```

Class I pseudosequences are derived from `data/MHC_pseudo.dat` in the official
NetMHCpan bundle. Class II pseudosequences are derived from the official
NetMHCIIpan `pseudosequence.2023.dat` resource.

## Preparation

After placing the two licensed source files in `local/raw/`, generate the
runtime CSV files from the repository root:

```bash
python \
  mimicneoai/immunogenicity_prediction/resources/hla_pseudoseq/prepare_netmhc_pseudoseq_csv.py
```

Each generated row contains:

```text
allele,pseudo_sequence,mhc_class,raw_allele,source_file
```

HLA-I alleles and HLA-II alpha-beta heterodimers are normalized by the
preparation script. DP and DQ chains must not be supplied independently to the
immunogenicity runtime.

## Inference Behavior

The runtime validates HLA inputs against these resources and reports the result
in `input_qc_hla_reference_status` and `immunogenicity_status`. An allele that
is absent from the frozen resource is not evaluable. Do not replace it with a
more common or nearby allele unless an independent, explicitly documented
analysis policy has been established.

Keep the original licensed files, generated CSV files, software version, and
file hashes with the analysis provenance. A resource change can alter the model
input and must not be treated as an interchangeable cache update.
