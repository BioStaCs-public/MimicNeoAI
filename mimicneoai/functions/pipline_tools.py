# coding=utf-8
import os
import yaml
import logging
import re
import shlex
import subprocess
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union
from pathlib import Path


def raise_for_failed_samples(async_results) -> None:
    """Raise after all submitted sample jobs have finished if any job failed."""

    failures = []
    for sample, result in async_results:
        try:
            result.get()
        except Exception as exc:
            failures.append((str(sample), exc))

    if failures:
        details = "; ".join(
            f"{sample}: {type(exc).__name__}: {exc}"
            for sample, exc in failures
        )
        raise RuntimeError(
            f"{len(failures)} sample pipeline job(s) failed: {details}"
        )


def sanitize_command_label(label: str) -> str:
    """Return a short filesystem-safe label for per-command log files."""

    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(label).strip())
    cleaned = cleaned.strip("_")
    return (cleaned or "cmd")[:120]


def _command_tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except ValueError:
        return cmd.strip().split()


def _script_label(path: str) -> str:
    name = Path(path).name
    for suffix in (".py", ".sh", ".pl", ".R"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name or "cmd"


def infer_command_display_name(cmd: str, pipline: Optional[str] = None) -> str:
    """Infer a user-facing command label when the caller did not provide one."""

    tokens = _command_tokens(cmd)
    if not tokens:
        return "cmd"

    lower_cmd = cmd.lower()
    if "bwa mem" in lower_cmd and "samtools sort" in lower_cmd:
        return "BWA alignment and sort"
    if "vardict-java" in lower_cmd:
        return "VarDict variant calling"
    if "configmanta" in lower_cmd:
        return "Manta configuration"
    if "configurestrelkasomaticworkflow" in lower_cmd:
        return "Strelka2 configuration"
    if "runworkflow.py" in lower_cmd:
        return "Workflow execution"
    if "hlahd.sh" in lower_cmd:
        return "HLA-HD typing"
    if "annotatepeaks.pl" in lower_cmd:
        return "HOMER annotatePeaks"
    if "diamond" in lower_cmd and "blastx" in lower_cmd:
        return "DIAMOND blastx"
    if "blastx" in lower_cmd:
        return "BLASTX peptide annotation"
    if "minimap2" in lower_cmd:
        return "minimap2 ORF genome mapping"
    if "pvacseq" in lower_cmd:
        return "pVACtools pvacseq"
    if "pvacbind" in lower_cmd:
        return "pVACtools pvacbind"

    first = Path(tokens[0]).name
    if first in {"python", "python3", "python3.10", "python3.9", "python2"}:
        for token in tokens[1:]:
            if token.endswith(".py"):
                return _script_label(token)
        return first
    if first in {"bash", "sh"}:
        for token in tokens[1:]:
            if token.endswith((".py", ".sh")):
                return _script_label(token)
        return first
    if first == "java":
        gatk_tools = {
            "MarkDuplicatesSpark",
            "BaseRecalibratorSpark",
            "ApplyBQSRSpark",
            "Mutect2",
            "FilterMutectCalls",
            "GetPileupSummaries",
            "CalculateContamination",
            "LearnReadOrientationModel",
            "PathSeqPipelineSpark",
        }
        for token in tokens:
            if token in gatk_tools:
                return f"GATK {token}"
        return "java"
    if first == "fastp":
        return "fastp QC"
    if first == "bowtie2":
        return "Bowtie2 alignment"
    if first == "bwa":
        return f"BWA {tokens[1]}" if len(tokens) > 1 else "BWA"
    if first == "samtools":
        return f"samtools {tokens[1]}" if len(tokens) > 1 else "samtools"
    if first == "bcftools":
        return f"bcftools {tokens[1]}" if len(tokens) > 1 else "bcftools"
    if first == "bamdst":
        return "bamdst QC"
    if first == "stringtie":
        return "StringTie transcript assembly"
    if first == "salmon":
        return "Salmon quantification"
    if first == "STAR":
        return "STAR alignment"
    if first == "rm":
        return "Clean intermediate files"
    if first == "mkdir":
        return "Prepare output directory"
    if first == "mv":
        return "Move output files"
    if first == "cp":
        return "Copy output files"
    return first


class tools:
    def __init__(self, sys_path: str, log_type: str, log_lock):
        """
        Initialize the tools class with shared resources.

        Args:
            sys_path: Base project/system path used for logs and artifacts.
            log_type: Logical pipeline/module name used to label logs.
            log_lock: A threading/multiprocessing lock used to serialize status updates.
        """
        # Execution status tracking
        self.failed_cmds_allSamples = {}
        self.done_cmds_allSamples = {}
        self.run_cmds_allSamples = {}
        self.samples: List[str] = []

        self.sys_path = sys_path.rstrip("/") + "/"
        self.log_type = log_type
        self.log_lock = log_lock

        # Logging paths (timestamped session)
        now = datetime.now()
        self.start_date = now.strftime("%Y-%m-%d_%H:%M:%S")
        logger_token = now.strftime("%Y%m%d%H%M%S%f")
        self.start_log = f"{self.sys_path}log/{self.start_date}/{self.log_type}_start_{self.start_date}.log"
        self.cmd_log_dir = f"{self.sys_path}log/{self.start_date}/detail/"

        # Ensure base log directories
        self.mkdir(f"{self.sys_path}log/{self.start_date}")

        # Configure logger
        self.logger = logging.getLogger(f"pipeline_logger::{self.start_date}::{logger_token}::{self.log_type}::{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Avoid duplicate handlers if multiple instances are created
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            file_handler = logging.FileHandler(self.start_log)
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

        # Default environment inherited from current process (can be extended/overridden)
        self.default_env: Dict[str, str] = os.environ.copy()

    # ---------------------------
    # Environment helpers
    # ---------------------------
    def build_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Return a writable environment dict inheriting current env, with optional overrides."""
        env: Dict[str, str] = self.default_env.copy()
        if extra:
            env.update(extra)
        return env

    def set_path_prefix(self, *dirs: str) -> None:
        """Prepend one or more directories to PATH in self.default_env."""
        existing = self.default_env.get("PATH", "")
        prefix = ":".join(d.rstrip("/") for d in dirs if d)
        self.default_env["PATH"] = (prefix + ":" + existing) if prefix else existing

    # ---------------------------
    # Command execution
    # ---------------------------
    def exec_cmd(
        self,
        cmd: str,
        sample: str,
        flag: Optional[Any] = None,
        env: Optional[Dict[str, str]] = None,
        pipline: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Execute a shell command and track its status.

        Args:
            cmd: Shell command string executed with `shell=True`.
            sample: Sample identifier used for per-sample logs.
            flag: Optional suffix to disambiguate the command name in logs.
            env: Environment variables to use for this command (inherits default if None).
            pipline: Optional pipeline tag affecting how the command name is parsed.
        """
        cmd_name = display_name or infer_command_display_name(cmd, pipline=pipline)
        file_label = sanitize_command_label(cmd_name)
        if flag is not None:
            flag_label = Path(str(flag)).name if "/" in str(flag) else str(flag)
            cmd_name = f"{cmd_name} [{flag_label}]"
            file_label = f"{file_label}_{sanitize_command_label(flag_label)}"

        # Per-sample log path
        current_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        cmd_log = f"{self.sys_path}log/{self.start_date}/detail/{sample}/{current_date}_{self.log_type}_{file_label}.log"
        self.mkdir(f"{self.sys_path}log/{self.start_date}/detail/{sample}/")

        # Mark as running
        with self.log_lock:
            self.status(self.run_cmds_allSamples, sample, f"{cmd_name}", "run", "", cmd, detail_log=cmd_log)

        start = datetime.now()
        try:
            with open(cmd_log, "a") as logfile:
                result = subprocess.call(
                    f"set -o pipefail\n{cmd}",
                    shell=True,
                    executable="/bin/bash",
                    stdout=logfile,
                    stderr=logfile,
                    env=(env if env is not None else self.default_env),
                )
            run_time = self.print_time(datetime.now() - start)
            if result:
                with self.log_lock:
                    self.status(
                        self.failed_cmds_allSamples,
                        sample,
                        f"{cmd_name}",
                        "failed",
                        run_time,
                        cmd,
                        detail_log=cmd_log,
                    )
                raise subprocess.CalledProcessError(result, cmd)

            with self.log_lock:
                self.status(
                    self.done_cmds_allSamples,
                    sample,
                    f"{cmd_name}",
                    "done",
                    run_time,
                    cmd,
                    detail_log=cmd_log,
                )
        finally:
            with self.log_lock:
                if sample in self.run_cmds_allSamples and cmd_name in self.run_cmds_allSamples[sample]:
                    self.run_cmds_allSamples[sample].remove(cmd_name)

    def judge_then_exec(
        self,
        sample: str,
        cmd: str,
        file: str,
        flag: Optional[Any] = None,
        env: Optional[Dict[str, str]] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Conditionally execute a command if the specified output file is missing or empty.
        """
        if (not os.path.exists(file)) or (os.path.isfile(file) and os.path.getsize(file) == 0):
            self.exec_cmd(
                cmd,
                sample,
                flag,
                env=(env if env is not None else self.default_env),
                display_name=display_name,
            )
        else:
            label = display_name or infer_command_display_name(cmd)
            self.logger.warning(f"{label} skipped because output already exists: {file}")

    def exec_cmd_with_time(
        self,
        cmd: str,
        sample: str,
        timeout: int = 3600,
        env: Optional[Dict[str, str]] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Execute a command with a timeout.

        Args:
            cmd: Shell command string executed with `shell=True`.
            sample: Sample identifier used for per-sample logs.
            timeout: Timeout in seconds (default: 3600).
            env: Optional environment for the command.
        """
        cmd_name = display_name or infer_command_display_name(cmd)
        file_label = sanitize_command_label(cmd_name)

        current_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        cmd_log = f"{self.sys_path}log/{self.start_date}/detail/{sample}/{current_date}_{self.log_type}_{file_label}.log"
        self.mkdir(f"{self.sys_path}log/{self.start_date}/detail/{sample}/")

        with self.log_lock:
            self.status(
                self.run_cmds_allSamples,
                sample,
                f"{cmd_name} timeout={timeout}",
                "run",
                "",
                cmd,
                detail_log=cmd_log,
            )
        start = datetime.now()

        try:
            with open(cmd_log, "a") as logfile:
                result = subprocess.run(
                    f"set -o pipefail\n{cmd}",
                    shell=True,
                    executable="/bin/bash",
                    stdout=logfile,
                    stderr=logfile,
                    timeout=timeout,
                    env=(env if env is not None else self.default_env),
                )
            end = datetime.now()
            run_time = self.print_time(end - start)

            if result.returncode != 0:
                with self.log_lock:
                    self.status(
                        self.failed_cmds_allSamples,
                        sample,
                        f"{cmd_name}",
                        "failed",
                        run_time,
                        cmd,
                        detail_log=cmd_log,
                    )
                raise subprocess.CalledProcessError(result.returncode, cmd)

            with self.log_lock:
                self.status(
                    self.done_cmds_allSamples,
                    sample,
                    f"{cmd_name}",
                    "done",
                    run_time,
                    cmd,
                    detail_log=cmd_log,
                )

        except subprocess.TimeoutExpired:
            end = datetime.now()
            run_time = self.print_time(end - start)
            with self.log_lock:
                self.status(
                    self.failed_cmds_allSamples,
                    sample,
                    f"{cmd_name}",
                    "timeout",
                    run_time,
                    cmd,
                    detail_log=cmd_log,
                )
            self.logger.error(f"Command for sample '{sample}' exceeded timeout of {timeout} seconds.")
            raise
        finally:
            running_label = f"{cmd_name} timeout={timeout}"
            with self.log_lock:
                if sample in self.run_cmds_allSamples and running_label in self.run_cmds_allSamples[sample]:
                    self.run_cmds_allSamples[sample].remove(running_label)

    def judge_then_exec_with_time(
        self,
        sample: str,
        cmd: str,
        file: str,
        timeout: int = 3600,
        env: Optional[Dict[str, str]] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Conditionally execute a command (with timeout) if the specified output file is missing or empty.
        """
        if (not os.path.exists(file)) or (os.path.isfile(file) and os.path.getsize(file) == 0):
            self.exec_cmd_with_time(
                cmd,
                sample,
                timeout=timeout,
                env=(env if env is not None else self.default_env),
                display_name=display_name,
            )
        else:
            label = display_name or infer_command_display_name(cmd)
            self.logger.warning(f"{label} skipped because output already exists: {file}")

    # ---------------------------
    # Shared state / reporting
    # ---------------------------
    def sharing_variable(self, manager, samples: List[str]) -> None:
        """
        Initialize shared structures (Manager dicts/lists) for multiprocessing.

        Args:
            manager: multiprocessing.Manager instance.
            samples: List of sample identifiers to track.
        """
        self.samples = samples + ["summary"]

        self.failed_cmds_allSamples = manager.dict()
        self.done_cmds_allSamples = manager.dict()
        self.run_cmds_allSamples = manager.dict()

        for sample in self.samples:
            self.failed_cmds_allSamples[sample] = manager.list()
            self.done_cmds_allSamples[sample] = manager.list()
            self.run_cmds_allSamples[sample] = manager.list()

    def status(
        self,
        cmds_dict,
        sample: str,
        info: str,
        info_type: str,
        run_time: str,
        cmd: str,
        detail_log: Optional[str] = None,
    ) -> None:
        """
        Update a status dictionary and log progress.

        Args:
            cmds_dict: One of {run/done/failed} manager dicts.
            sample: Sample identifier.
            info: Status descriptor for this command.
            info_type: One of {"run", "done", "failed", "timeout"}.
            run_time: Formatted runtime string.
            cmd: Original command string.
        """
        # Append info with or without runtime
        if run_time == "0h0m0s" or info_type == "run":
            cmds_dict[sample].append(info)
        else:
            cmds_dict[sample].append(f"{info} {run_time}")

        # Compose a compact progress snapshot
        message = ""
        for key in cmds_dict.keys():
            if len(cmds_dict[key]) != 0:
                message += f"{key}-{cmds_dict[key][:]}\n"
        message = message[:-1] if message else ""

        # Log by status type
        if info_type == "run":
            detail_log_message = f"Detail log:\n{detail_log}\n" if detail_log else ""
            self.logger.info(
                f"Run {sample} {info}! Progress: {self.samples.index(sample) + 1}/{len(self.samples)}.\n"
                f"{detail_log_message}"
                f"Detail cmd:\n{cmd}.\nRunning:\n{message}."
            )
        elif info_type == "failed":
            detail = f" Detail log: {detail_log}" if detail_log else ""
            self.logger.error(f"{sample} {info} failed! Time: {run_time}.{detail}")
        elif info_type == "timeout":
            detail = f" Detail log: {detail_log}" if detail_log else ""
            self.logger.error(f"{sample} {info} exceeded timeout! Time: {run_time}.{detail}")
        elif info_type == "done":
            self.logger.info(f"{sample} {info} succeeded! Time: {run_time}. ")
        else:
            self.logger.info("Status: other")

    def summary(self) -> None:
        """Emit a final summary covering all samples and command outcomes."""
        self.logger.info("end")

        done_message = "\n".join(
            [f"{key}-{self.done_cmds_allSamples[key][:]}"
             for key in self.done_cmds_allSamples.keys()
             if len(self.done_cmds_allSamples[key]) != 0]
        )
        failed_message = "\n".join(
            [f"{key}-{self.failed_cmds_allSamples[key][:]}"
             for key in self.failed_cmds_allSamples.keys()
             if len(self.failed_cmds_allSamples[key]) != 0]
        )

        if done_message:
            self.logger.info(f"all done cmds:\n{done_message}")
        if failed_message:
            self.logger.error(f"all failed cmds:\n{failed_message}")

    def has_failures(self) -> bool:
        """Return True when any sample has a recorded command failure."""

        return any(
            len(self.failed_cmds_allSamples[sample]) != 0
            for sample in self.failed_cmds_allSamples.keys()
        )

    # ---------------------------
    # Misc utilities
    # ---------------------------
    def write_log(self, text: str, type_log: str) -> None:
        """Write a message at the requested log level."""
        if type_log == "info":
            self.logger.info(text)
        elif type_log == "error":
            self.logger.error(text)
        elif type_log in ("warn", "warning"):
            self.logger.warning(text)
        else:
            self.logger.debug(text)

    def mkdir(self, path: str) -> None:
        """Create a directory if it does not exist (idempotent)."""
        os.makedirs(path, exist_ok=True)

    def print_pool_error(self, value: Any) -> None:
        """Log an error object originating from a multiprocessing pool."""
        self.logger.error(f"error: {value}")

    def cp_configure(self, file: Any) -> None:
        """
        Persist a copy of the (already loaded) configuration YAML for reproducibility.
        """
        out = f"{self.sys_path}log/{self.start_date}/{self.log_type}_configure_{self.start_date}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(file, f, sort_keys=False)

    def print_time(self, delta: timedelta) -> str:
        """
        Format a timedelta into 'XdXhXmXs' (days omitted if zero).

        Args:
            delta: Duration to format.
        """
        total_seconds = int(delta.total_seconds())
        days = total_seconds // 86400
        remaining = total_seconds % 86400
        hour = remaining // 3600
        minutes = (remaining % 3600) // 60
        seconds = remaining % 60
        if days > 0:
            return f"{days}d{hour}h{minutes}m{seconds}s"
        return f"{hour}h{minutes}m{seconds}s"

    def get_configure(self, args_configure: Optional[str]):
        """
        Load a configuration YAML.

        Behavior:
          - If args_configure is None/empty: load sys_path/configures/{log_type}_configure.yaml
          - Otherwise: load the YAML at args_configure
        A copy of the loaded YAML is saved to the run log directory for reproducibility.
        """
        if not args_configure:
            cfg_path = f"{self.sys_path}configures/{self.log_type}_configure.yaml"
        else:
            cfg_path = args_configure

        with open(cfg_path, "r") as file:
            configure = yaml.safe_load(file)

        self.cp_configure(configure)
        return configure

    def _abspath_like(self, base: Path, v: Union[str, Path]):
        """
        Convert relative path-like strings to absolute paths w.r.t. `base`,
        WITHOUT resolving symlinks (i.e., do NOT use Path.resolve() / realpath()).
        """
        if v is None:
            return v

        # keep non-string scalars as-is (e.g., int/bool)
        if not isinstance(v, (str, Path)):
            return v

        s = str(v).strip()
        if s == "":
            return v

        # keep the original quoting style in YAML for blast outfmt etc.
        # e.g. OUTFMT: "'6 ...'"  -> we should NOT treat it as a path
        if (s[0] in "\"'" and s[-1] == s[0]) and ("/" not in s and "\\" not in s):
            return v

        # If it's an absolute path already, just normalize (no symlink follow)
        p = Path(s)
        if p.is_absolute():
            return os.path.normpath(str(p))

        # Relative path: join with base, then abspath+normpath (no symlink follow)
        joined = os.path.join(str(base), s)
        return os.path.normpath(os.path.abspath(joined))

    def _map_paths_recursive(self, node: Any, conv) -> Any:
        """Recursively apply conv() to all leaf values in nested dict/list."""
        if isinstance(node, dict):
            return {k: self._map_paths_recursive(v, conv) for k, v in node.items()}
        if isinstance(node, list):
            return [self._map_paths_recursive(v, conv) for v in node]
        return conv(node)

    def get_paths(self, args_paths: Optional[str]) -> Dict[str, Any]:
        """
        Load a paths YAML and resolve relative paths robustly.

        Behavior:
          1) If args_paths is None/empty: try to load mimicneoai/configures/paths.yaml via importlib.resources;
             if that fails, fall back to <sys_path>/mimicneoai/configures/paths.yaml.
          2) If args_paths is provided: load that file and resolve relative paths against its directory.

        Returns:
          A dict with "../" or "./" strings resolved to absolute paths.
        """
        # Determine YAML location
        if not args_paths:
            try:
                from importlib.resources import files
                yaml_path = Path(files("mimicneoai") / "configures" / "paths.yaml")
            except Exception:
                yaml_path = Path(self.sys_path).resolve() / "mimicneoai" / "configures" / "paths.yaml"
        else:
            yaml_path = Path(args_paths).resolve()

        if not yaml_path.exists():
            raise FileNotFoundError(f"paths.yaml not found at: {yaml_path}")

        with open(yaml_path, "r") as f:
            paths = yaml.safe_load(f) or {}

        base = yaml_path.parent
        paths = self._map_paths_recursive(paths, lambda v: self._abspath_like(base, v))

        # Optionally persist the resolved paths for reproducibility
        try:
            out = f"{self.sys_path}log/{self.start_date}/{self.log_type}_paths_resolved_{self.start_date}.yaml"
            with open(out, "w") as f:
                yaml.safe_dump(paths, f, sort_keys=False)
        except Exception:
            pass

        return paths

    def write2shell(self, cmd: str, fileName: str) -> str:
        """
        Append a command to a shell script under the current run's log directory.

        Args:
            cmd: Command line to append.
            fileName: Base name of the shell script (without path).

        Returns:
            Absolute path to the created/updated shell script.
        """
        script_path = f"{self.sys_path}log/{self.start_date}/{fileName}.sh"
        self.mkdir(os.path.dirname(script_path))
        with open(script_path, "a") as f:
            f.write(f"{cmd}\n")
        return script_path
