"""Compatibility module; prefer :mod:`mimicneoai.immunogenicity_prediction.cli`."""

from mimicneoai.immunogenicity_prediction.cli import (
    _build_cfg_dict,
    _load_yaml,
    main,
    run_from_config,
)

__all__ = ["_build_cfg_dict", "_load_yaml", "main", "run_from_config"]


if __name__ == "__main__":
    raise SystemExit(main())
