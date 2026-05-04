from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import dataset_manager
import import_alpamayo_prediction_json as prediction_importer
import route_caputure


def workspace_tempdir():
    base_dir = Path(os.environ.get("PROJECT19_TEST_TMP", tempfile.gettempdir()))
    test_temp_dir = base_dir / "project19_tests"
    test_temp_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=test_temp_dir, ignore_cleanup_errors=True)


class PredictionImportTests(unittest.TestCase):
    def make_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                source TEXT NOT NULL,
                frame_number INTEGER NOT NULL
            );

            CREATE TABLE annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE
            );
            """
        )
        prediction_importer.ensure_tables(conn)
        return conn

    def test_insert_prediction_keeps_command_reasoning_and_points(self):
        conn = self.make_connection()
        conn.execute(
            """
            INSERT INTO frames (filename, relative_path, source, frame_number)
            VALUES (?, ?, ?, ?)
            """,
            ("000005.png", "../datasets/route_1/segment_00/raw/000005.png", "route_1_segment_00", 5),
        )
        frame_id = conn.execute("SELECT id FROM frames").fetchone()["id"]
        conn.execute("INSERT INTO annotations (frame_id) VALUES (?)", (frame_id,))
        frame_row = prediction_importer.find_frame(conn, "route_1_segment_00", 5)

        payload = {
            "model_name": "nvidia/Alpamayo-1.5-10B",
            "nav_command": "Turn left in 20 meters",
            "command_text": "Turn left in 20 meters",
            "command_source": "ground_truth_heuristic",
            "selection_mode": "heuristic",
            "selected_sample_index": 0,
            "num_traj_samples": 1,
            "guidance_weight": 1.5,
            "max_generation_length": 256,
            "frames_requested": 2,
            "frames_stored": 2,
            "reasoning_text": "The lane curves left and the route continues left.",
            "selected_path": [
                {"step_index": 0, "x_m": 1.0, "y_m": 0.1, "z_m": 0.0},
                {"step_index": 1, "x_m": 2.0, "y_m": 0.4, "z_m": 0.0},
            ],
            "ground_truth_path": [
                {"step_index": 0, "x_m": 1.1, "y_m": 0.0, "z_m": 0.0},
                {"step_index": 1, "x_m": 2.1, "y_m": 0.3, "z_m": 0.0},
            ],
        }

        prediction_id = prediction_importer.insert_prediction(conn, frame_row, payload)
        row = conn.execute(
            """
            SELECT nav_command, command_text, reasoning_text, cot, frames_stored
            FROM alpamayo_predictions
            WHERE id = ?
            """,
            (prediction_id,),
        ).fetchone()
        point_count = conn.execute(
            "SELECT COUNT(*) FROM alpamayo_prediction_points WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()[0]

        self.assertEqual(row["command_text"], "Turn left in 20 meters")
        self.assertEqual(row["reasoning_text"], "The lane curves left and the route continues left.")
        self.assertEqual(row["cot"], row["reasoning_text"])
        self.assertEqual(row["frames_stored"], 2)
        self.assertEqual(point_count, 2)
        conn.close()

    def test_source_helpers_match_route_segment_layout(self):
        with workspace_tempdir() as tmp:
            segment = Path(tmp) / "route_7" / "segment_03"
            segment.mkdir(parents=True)

            self.assertEqual(
                prediction_importer.infer_source({"route": "route_7", "segment": "segment_03"}),
                "route_7_segment_03",
            )
            self.assertEqual(
                prediction_importer.infer_source_from_segment(segment),
                "route_7_segment_03",
            )
            self.assertEqual(
                prediction_importer.infer_predictions_dir(segment),
                segment.resolve() / "predictions",
            )


class RouteCaptureTests(unittest.TestCase):
    def test_count_raw_frames_counts_segment_pngs_only(self):
        with workspace_tempdir() as tmp:
            route_dir = Path(tmp)
            raw_0 = route_dir / "segment_00" / "raw"
            raw_1 = route_dir / "segment_01" / "raw"
            raw_0.mkdir(parents=True)
            raw_1.mkdir(parents=True)

            for path in [
                raw_0 / "000000.png",
                raw_0 / "000001.png",
                raw_1 / "000000.png",
            ]:
                path.write_bytes(b"fake image bytes")
            (raw_1 / "notes.txt").write_text("not a frame", encoding="utf-8")

            frame_count, latest_mtime = route_caputure.count_raw_frames(route_dir)

            self.assertEqual(frame_count, 3)
            self.assertIsNotNone(latest_mtime)


class ValidationHelperTests(unittest.TestCase):
    def test_turn_throttle_and_brake_validation_boundaries(self):
        self.assertTrue(dataset_manager.validate_turn_angle(-180))
        self.assertTrue(dataset_manager.validate_turn_angle(180))
        self.assertFalse(dataset_manager.validate_turn_angle(181))
        self.assertFalse(dataset_manager.validate_turn_angle("0"))

        self.assertTrue(dataset_manager.validate_throttle(0.0))
        self.assertTrue(dataset_manager.validate_throttle(1.0))
        self.assertFalse(dataset_manager.validate_throttle(1.1))

        self.assertTrue(dataset_manager.validate_brake(0.0))
        self.assertTrue(dataset_manager.validate_brake(1.0))
        self.assertFalse(dataset_manager.validate_brake(-0.1))


if __name__ == "__main__":
    unittest.main()
