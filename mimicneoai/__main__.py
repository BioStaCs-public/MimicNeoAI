# coding=utf-8
from __future__ import annotations
import argparse
import sys
from typing import List, Callable

def _call_main(func: Callable, rest: List[str]) -> int:
    """
    Always patch sys.argv to only the child-args before calling child's main.
    Works whether the child main is main(argv=None) or main().
    """
    # strip a leading '--' if present
    rest = list(rest or [])
    if rest and rest[0] == "--":
        rest = rest[1:]

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + rest
        try:
            return int(func(rest) or 0)   # child supports argv
        except TypeError:
            return int(func() or 0)       # child uses sys.argv
    finally:
        sys.argv = old_argv


def _run_download_database(rest: List[str]) -> int:
    from mimicneoai import download_database
    return _call_main(download_database.main, rest)

def _run_cryptic(rest: List[str]) -> int:
    from mimicneoai.cryptic_pipeline.cryptic import main as cryptic_main
    return _call_main(cryptic_main, rest)

def _run_microbial(rest: List[str]) -> int:
    from mimicneoai.microbial_pipeline.microbial import main as microbial_main
    return _call_main(microbial_main, rest)

def _run_mutation_derived(rest: List[str]) -> int:
    from mimicneoai.mutation_derived_pipeline.mutation_derived import main as md_main
    return _call_main(md_main, rest)

def _run_immunogenicity_prediction(rest: List[str]) -> int:
    from mimicneoai.immunogenicity_prediction.cli import main as pred_main
    return _call_main(pred_main, rest)

def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(prog="mimicneoai", description="MimicNeoAI unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 注意：这里不再为各子命令添加 "rest = REMAINDER"
    subparsers.add_parser("download_database", help="Download and extract the reference database")
    subparsers.add_parser("cryptic",            help="Run the cryptic pipeline")
    subparsers.add_parser("microbial",          help="Run the microbial pipeline")
    subparsers.add_parser("mutation-derived",   help="Run the mutation-derived pipeline")
    subparsers.add_parser("immunogenicity-prediction", help="Run immunogenicity prediction subtool")

    # 关键：用 parse_known_args，保留未知参数给子命令
    args, rest = parser.parse_known_args(argv)

    # rest 里仍然包含子命令自身后面的全部参数（例如 -c ...）
    if args.command == "download_database":
        return _run_download_database(rest)
    if args.command == "cryptic":
        return _run_cryptic(rest)
    if args.command == "microbial":
        return _run_microbial(rest)
    if args.command == "mutation-derived":
        return _run_mutation_derived(rest)
    if args.command == "immunogenicity-prediction":
        return _run_immunogenicity_prediction(rest)

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
