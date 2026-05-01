#!/usr/bin/env python3
"""
View/export Alpamayo predictions stored by run_alpamayo_db_predictions.py.

Examples:
  python3 llm-model-tests/view_alpamayo_predictions.py --db llm-model-tests/annotations.db --limit 10
  python3 llm-model-tests/view_alpamayo_predictions.py --db llm-model-tests/annotations.db --export-json alpamayo_predictions.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="View Alpamayo DB predictions.")
    parser.add_argument("--db", default="llm-model-tests/annotations.db")
    parser.add_argument("--source", default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--export-json", default=None)
    return parser.parse_args()


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_predictions(conn, args):
    where = []
    params = []

    if args.source:
        where.append("f.source = ?")
        params.append(args.source)
    if args.frame_id is not None:
        where.append("f.id = ?")
        params.append(args.frame_id)

    where_sql = "WHERE " + " AND ".join(where) if where else ""
    limit_sql = ""
    if args.limit is not None and args.limit > 0:
        limit_sql = "LIMIT ?"
        params.append(args.limit)

    return conn.execute(
        f"""
        SELECT
            ap.id AS prediction_id,
            ap.frame_id,
            f.filename,
            f.source,
            f.frame_number,
            ap.nav_command,
            ap.nav_command_source,
            ap.selection_mode,
            ap.selected_sample_index,
            ap.frames_stored,
            ap.cot,
            ap.selected_path_json,
            ap.gt_path_json,
            ap.created_at
        FROM alpamayo_predictions ap
        JOIN frames f ON f.id = ap.frame_id
        {where_sql}
        ORDER BY ap.id DESC
        {limit_sql}
        """,
        params,
    ).fetchall()


def fetch_points(conn, prediction_id):
    rows = conn.execute(
        """
        SELECT step_index, x_m, y_m, z_m
        FROM alpamayo_prediction_points
        WHERE prediction_id = ?
        ORDER BY step_index
        """,
        (prediction_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def main():
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = connect(str(db_path))
    rows = fetch_predictions(conn, args)

    if args.export_json:
        output = []
        for row in rows:
            record = dict(row)
            record["points"] = fetch_points(conn, row["prediction_id"])
            record["selected_path"] = json.loads(record.pop("selected_path_json"))
            if record.get("gt_path_json"):
                record["gt_path"] = json.loads(record.pop("gt_path_json"))
            output.append(record)

        Path(args.export_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Exported {len(output)} prediction(s) to {args.export_json}")
        return

    print(f"Found {len(rows)} prediction(s)")
    for row in rows:
        points = fetch_points(conn, row["prediction_id"])
        final = points[-1] if points else None
        final_text = ""
        if final:
            final_text = f" final=({final['x_m']:.2f}, {final['y_m']:.2f}, {final['z_m']:.2f})"

        print(
            f"[prediction={row['prediction_id']}] frame_id={row['frame_id']} "
            f"{row['source']} {row['filename']} nav='{row['nav_command']}' "
            f"points={row['frames_stored']}{final_text}"
        )
        if row["cot"]:
            print(f"  reasoning: {row['cot'][:220]}")


if __name__ == "__main__":
    main()
