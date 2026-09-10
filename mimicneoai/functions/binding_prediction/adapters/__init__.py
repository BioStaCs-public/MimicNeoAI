"""Predictor adapter registry."""

from __future__ import annotations

from .base import AdapterConfig, PredictorAdapter
from .iedb_mhci import IedbMhciAdapter
from .iedb_mhcii import IedbMhciiAdapter
from .mhcflurry import MhcflurryAdapter
from .mhcnuggets import MhcnuggetsAdapter
from .netmhciipan import NetMHCIIpanAdapter
from .netmhcpan import NetMHCpanAdapter


ADAPTER_CLASSES = (
    MhcflurryAdapter,
    MhcnuggetsAdapter,
    IedbMhciAdapter,
    IedbMhciiAdapter,
    NetMHCpanAdapter,
    NetMHCIIpanAdapter,
)


def adapter_for_algorithm(algorithm: str, config: AdapterConfig) -> PredictorAdapter:
    """Return an adapter instance for a canonical algorithm name."""

    for adapter_class in ADAPTER_CLASSES:
        if algorithm in adapter_class.supported_algorithms:
            return adapter_class(config)
    supported = sorted({item for cls in ADAPTER_CLASSES for item in cls.supported_algorithms})
    raise ValueError(f"Unsupported binding predictor algorithm: {algorithm}. Supported: {supported}")


SUPPORTED_ALGORITHMS = sorted({item for cls in ADAPTER_CLASSES for item in cls.supported_algorithms})
