#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_DIR = PROJECT_ROOT / "tests"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run project tests.")
    parser.add_argument(
        "--dependencies",
        action="store_true",
        help="also check that setup dependencies are available",
    )
    args = parser.parse_args()

    if args.dependencies:
        os.environ["RUN_DEPENDENCY_TESTS"] = "1"

    pattern = "test_dependencies.py" if args.dependencies else "test_*.py"
    suite = unittest.defaultTestLoader.discover(str(TESTS_DIR), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
