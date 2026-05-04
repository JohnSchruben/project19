#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
import argparse
import os
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_DIR = PROJECT_ROOT / "tests"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
ALPAMAYO_SRC_DIR = PROJECT_ROOT / "alpamayo" / "src"


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

    suite = build_dependency_suite() if args.dependencies else build_default_suite()
    runner = unittest.TextTestRunner(verbosity=1, buffer=True)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def build_default_suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()

    root_suite = unittest.defaultTestLoader.discover(str(TESTS_DIR), pattern="test_*.py")
    suite.addTests(
        test
        for test in iter_tests(root_suite)
        if test.__class__.__module__ != "test_dependencies"
    )

    pipeline_db_test = PIPELINE_DIR / "test_database.py"
    if pipeline_db_test.exists():
        add_to_syspath(PIPELINE_DIR)
        suite.addTests(load_unittest_module("pipeline_test_database", pipeline_db_test))

    alpamayo_nav_test = PROJECT_ROOT / "alpamayo" / "tests" / "test_navigation_command.py"
    if alpamayo_nav_test.exists():
        add_to_syspath(ALPAMAYO_SRC_DIR)
        suite.addTests(load_function_tests("alpamayo_test_navigation_command", alpamayo_nav_test))

    return suite


def build_dependency_suite() -> unittest.TestSuite:
    return unittest.defaultTestLoader.discover(str(TESTS_DIR), pattern="test_dependencies.py")


def add_to_syspath(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_unittest_module(module_name: str, path: Path) -> unittest.TestSuite:
    module = load_module(module_name, path)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def load_function_tests(module_name: str, path: Path) -> unittest.TestSuite:
    module = load_module(module_name, path)
    suite = unittest.TestSuite()
    for name in sorted(dir(module)):
        test_func = getattr(module, name)
        if name.startswith("test_") and callable(test_func):
            suite.addTest(unittest.FunctionTestCase(test_func, description=name))
    return suite


def iter_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


if __name__ == "__main__":
    sys.exit(main())
