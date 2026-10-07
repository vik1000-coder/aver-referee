"""aver_cli — the free referee: read facts about repeated agent runs, rule on them.

This package is the OSS front door (docs/104). It reads evidence a team already
has, computes the repeat-churn floor of one side, and referees a candidate side
against an incumbent under acceptance rules a person wrote down. It calls no
model provider, replays no traffic, scores no output, and enforces nothing at
runtime.

Dependency boundary (pinned by ``tests/test_aver_policy_is_standalone.py``):
standard library plus ``aver_policy`` and ``aver_cass`` only. Importing this
package must never pull in the hosted service.
"""

from __future__ import annotations

__version__ = "0.1.0"

EVIDENCE_SCHEMA = "aver-evidence/v1"
POLICY_SCHEMA = "aver-policy/v3"

__all__ = ["EVIDENCE_SCHEMA", "POLICY_SCHEMA", "__version__"]
