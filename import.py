#!/usr/bin/env python3
"""
Import route annotations and Alpamayo prediction JSON files into pipeline/annotations.db.

Usage:
  python3 import.py datasets/route_1
  python3 import.py datasets/route_1/segment_00
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both pipeline DB import scripts for a route or segment."
    )
    parser.add_argument("target", help="Route folder or segment folder")
    parser.add_argument("--db", default="pipeline/annotations.db", help="SQLite DB path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Alpamayo predictions for matching frames.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_segments(target: Path) -> tuple[Path, list[Path]]:
    target = target.expanduser().resolve()
    if not target.is_dir():
        raise SystemExit(f"[ERROR] Folder does not exist: {target}")

    if target.name.startswith("segment_"):
        return target.parent, [target]

    segments = sorted(path for path in target.iterdir() if path.is_dir() and path.name.startswith("segment_"))
    if not segments:
        raise SystemExit(f"[ERROR] No segment_* folders found under {target}")
    return target, segments


def run_command(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    route_dir, segments = discover_segments(Path(args.target))

    print(f"[INFO] Route folder: {route_dir}")
    print("[INFO] Segments:")
    for segment in segments:
        print(f"  - {segment}")

    for segment in segments:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "pipeline" / "import_route_annotations.py"),
                str(segment),
                "--db",
                args.db,
                *(["--dry-run"] if args.dry_run else []),
            ]
        )

    for segment in segments:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "pipeline" / "import_alpamayo_prediction_json.py"),
                str(segment),
                "--db",
                args.db,
                *(["--overwrite"] if args.overwrite else []),
                *(["--dry-run"] if args.dry_run else []),
            ]
        )

    print(f"[SUCCESS] Imported into {args.db}")


if __name__ == "__main__":
    main()
