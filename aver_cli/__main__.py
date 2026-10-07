"""``python -m aver_cli`` — the same entry point as the ``aver`` console script."""

from __future__ import annotations

import sys

from aver_cli.cli import main

if __name__ == "__main__":
    sys.exit(main())
