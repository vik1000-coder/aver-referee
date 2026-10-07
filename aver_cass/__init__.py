"""Open-source CASS primitives for Aver assurance records.

This package is intentionally free of Aver product-service dependencies. The
hosted app may wrap these functions, but customer runners and reviewers should
be able to inspect the core math without trusting the service.
"""

from aver_cass.core import (
    aggregate_blocks,
    effective_block_quota,
    required_sample_size,
    sign_test_p_one_sided,
    wilson_interval,
    z_score,
)
from aver_cass.evidence import EvidenceVerdict, verdict_from_counts

__all__ = [
    "EvidenceVerdict",
    "aggregate_blocks",
    "effective_block_quota",
    "required_sample_size",
    "sign_test_p_one_sided",
    "verdict_from_counts",
    "wilson_interval",
    "z_score",
]

__version__ = "0.2.0"
