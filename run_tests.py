#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
import argparse
import os
import fnmatch
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
    if not args.dependencies:
        suite = unittest.TestSuite(
            test
            for test in iter_tests(suite)
            if not fnmatch.fnmatch(test.__class__.__module__, "test_dependencies")
        )
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def iter_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


if __name__ == "__main__":
    sys.exit(main())
