#!/usr/bin/env python3
"""Run MimicNeoAI local binding prediction for cryptic aeSEP peptides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from mimicneoai.functions.binding_prediction import PREDICTOR_PATH_ARGUMENTS
from mimicneoai.functions.binding_prediction.nonmutation_workflow import main as run_nonmutation_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--pep-fasta", required=True)
    parser.add_argument(
        "--input-mode",
        choices=("parent-fasta", "peptide-core"),
        default="parent-fasta",
        help="parent-fasta tiles ORF sequences; peptide-core treats FASTA records as pre-tiled peptides.",
    )
    parser.add_argument("--hla-file", required=True)
    parser.add_argument("-o", "--outdir", required=True, help="Cryptic 07-hla_binding_pred_mimicneoai directory.")
    parser.add_argument("-t", "--threads", type=int, default=10, help="Local predictor workers.")
    parser.add_argument("--mhc-i-lengths", default="8,9,10")
    parser.add_argument("--mhc-ii-lengths", default="15")
    parser.add_argument("--preset", default="")
    parser.add_argument(
        "--algorithms",
        default="MHCflurry,MHCflurryEL,MHCnuggetsI,MHCnuggetsII,NNalign,NetMHCpan,NetMHCpanEL,NetMHCIIpan,NetMHCIIpanEL",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-task-rows", type=int, default=5000000)
    parser.add_argument("--force-large-samples", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--command-timeout", type=int, default=None)
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--windows-only", action="store_true")
    for option in PREDICTOR_PATH_ARGUMENTS.values():
        parser.add_argument(option, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir) / args.sample
    command = [
        "-s",
        args.sample,
        "--pep-fasta",
        args.pep_fasta,
        "--input-mode",
        args.input_mode,
        "--hla-file",
        args.hla_file,
        "-o",
        str(outdir),
        "--mhc-i-lengths",
        args.mhc_i_lengths,
        "--mhc-ii-lengths",
        args.mhc_ii_lengths,
        "--algorithms",
        args.algorithms,
        "--workers",
        str(args.threads),
        "--max-task-rows",
        str(args.max_task_rows),
        "--device",
        args.device,
        "--gpu-id",
        args.gpu_id,
    ]
    if args.preset:
        command.extend(["--preset", args.preset])
    if args.chunk_size:
        command.extend(["--chunk-size", str(args.chunk_size)])
    if args.command_timeout:
        command.extend(["--command-timeout", str(args.command_timeout)])
    if args.force_large_samples:
        command.append("--force-large-samples")
    if args.skip_prediction:
        command.append("--skip-prediction")
    if args.windows_only:
        command.append("--windows-only")
    for option in PREDICTOR_PATH_ARGUMENTS.values():
        value = getattr(args, option[2:].replace("-", "_"))
        if value:
            command.extend([option, value])
    return run_nonmutation_binding(command)


if __name__ == "__main__":
    raise SystemExit(main())
