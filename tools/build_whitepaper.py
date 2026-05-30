"""
One-shot orchestrator: run the component test, then build the LaTeX whitepaper.

Equivalent to:

    python tools/component_test.py
    python tools/whitepaper_builder.py

Usage:

    python tools/build_whitepaper.py
    python tools/build_whitepaper.py --skip-tests       # reuse existing manifest
    python tools/build_whitepaper.py --no-compile       # emit .tex only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import component_test, whitepaper_builder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-tests", action="store_true",
                   help="Reuse the existing reports/component_test_manifest.json")
    p.add_argument("--no-compile", action="store_true",
                   help="Emit .tex only (skip pdflatex)")
    args = p.parse_args(argv)

    if not args.skip_tests:
        rc = component_test.main()
        if rc != 0:
            print(f"component_test exited with rc={rc}", file=sys.stderr)
            return rc

    builder_args = []
    if args.no_compile:
        builder_args.append("--no-compile")
    return whitepaper_builder.main(builder_args)


if __name__ == "__main__":
    raise SystemExit(main())
