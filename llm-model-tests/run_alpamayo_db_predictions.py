#!/usr/bin/env python3
"""
Run Alpamayo on frames selected from annotations.db and store predicted paths.

This is a separate DB-driven workflow. It does not replace the original
raw-image Alpamayo scripts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPAMAYO_SRC = PROJECT_ROOT / "alpamayo" / "src"
if str(ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO_SRC))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo on DB-selected frames and store predicted paths."
    )
    parser.add_argument("--db", default="llm-model-tests/annotations.db")
    parser.add_argument("--source", default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--start-frame-number", type=int, default=None)
    parser.add_argument("--end-frame-number", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--command", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--num-traj-samples", type=int, default=4)
    parser.add_argument("--selection-mode", choices=["heuristic", "mean", "median"], default="heuristic")
    parser.add_argument("--guidance-weight", type=float, default=1.5)
    parser.add_argument("--max-gen-length", type=int, default=256)
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=["wide", "left", "right", "front"],
        default=["wide", "left", "right", "front"],
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_tables(conn):
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS alpamayo_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
            annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL,
            model_name TEXT NOT NULL,
            nav_command TEXT NOT NULL,
            nav_command_source TEXT NOT NULL,
            selection_mode TEXT NOT NULL,
            selected_sample_index INTEGER NOT NULL,
            num_traj_samples INTEGER NOT NULL,
            guidance_weight REAL NOT NULL,
            max_generation_length INTEGER NOT NULL,
            frames_requested INTEGER NOT NULL,
            frames_stored INTEGER NOT NULL,
            cot TEXT,
            selected_path_json TEXT NOT NULL,
            all_samples_json TEXT,
            gt_path_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alpamayo_prediction_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL REFERENCES alpamayo_predictions(id) ON DELETE CASCADE,
            step_index INTEGER NOT NULL,
            x_m REAL NOT NULL,
            y_m REAL NOT NULL,
            z_m REAL NOT NULL,
            UNIQUE(prediction_id, step_index)
        );

        CREATE INDEX IF NOT EXISTS idx_alpamayo_predictions_frame_id
            ON alpamayo_predictions(frame_id);
        CREATE INDEX IF NOT EXISTS idx_alpamayo_prediction_points_prediction_id
            ON alpamayo_prediction_points(prediction_id);
        """
    )
    conn.commit()


def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_tables(conn)
    return conn


def selected_rows(conn, args):
    where = []
    params = []
    if args.source:
        where.append("f.source = ?")
        params.append(args.source)
    if args.frame_id is not None:
        where.append("f.id = ?")
        params.append(args.frame_id)
    if args.start_frame_number is not None:
        where.append("f.frame_number >= ?")
        params.append(args.start_frame_number)
    if args.end_frame_number is not None:
        where.append("f.frame_number <= ?")
        params.append(args.end_frame_number)
    if not args.overwrite:
        where.append("NOT EXISTS (SELECT 1 FROM alpamayo_predictions ap WHERE ap.frame_id = f.id)")

    where_sql = "WHERE " + " AND ".join(where) if where else ""
    limit_sql = ""
    if args.limit is not None:
        limit_sql = "LIMIT ? OFFSET ?"
        params.extend([args.limit, args.offset])
    elif args.offset:
        limit_sql = "LIMIT -1 OFFSET ?"
        params.append(args.offset)

    return conn.execute(
        f"""
        SELECT
            f.id AS frame_id,
            f.filename,
            f.relative_path,
            f.frame_number,
            f.source,
            MAX(a.id) AS annotation_id
        FROM frames f
        LEFT JOIN annotations a ON a.frame_id = f.id
        {where_sql}
        GROUP BY f.id
        ORDER BY f.source, f.frame_number, f.filename
        {limit_sql}
        """,
        params,
    ).fetchall()


def resolve_image_path(db_path, relative_path):
    return (Path(db_path).resolve().parent / relative_path).resolve()


def segment_and_frame_from_image(image_path):
    if image_path.parent.name not in {"raw", "raw_front", "raw_left", "raw_right"}:
        raise ValueError(f"Expected image under raw/raw_front/raw_left/raw_right: {image_path}")
    try:
        frame_idx = int(image_path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected numeric frame filename: {image_path.name}") from exc
    return image_path.parent.parent, frame_idx


def extract_cot(extra, sample_idx):
    try:
        cot_data = extra.get("cot", [])
        while hasattr(cot_data, "__len__") and not isinstance(cot_data, str) and len(cot_data) == 1:
            cot_data = cot_data[0]
        if hasattr(cot_data, "__len__") and not isinstance(cot_data, str):
            if len(cot_data) > sample_idx:
                return str(cot_data[sample_idx]).strip()
            if len(cot_data) > 0:
                return str(cot_data[0]).strip()
        return str(cot_data).strip()
    except Exception:
        return ""


def select_prediction(pred_tensor, nav_cmd, num_frames, selection_mode):
    import numpy as np

    pred_np = pred_tensor.detach().cpu().numpy()[0, 0]
    if pred_np.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32), 0, pred_np

    if selection_mode == "mean":
        selected = pred_np.mean(axis=0)
        sample_idx = int(
            np.argmin(
                np.linalg.norm(pred_np[:, :, :2] - selected[None, :, :2], axis=-1).mean(axis=1)
            )
        )
    elif selection_mode == "median":
        selected = np.median(pred_np, axis=0)
        sample_idx = int(
            np.argmin(
                np.linalg.norm(pred_np[:, :, :2] - selected[None, :, :2], axis=-1).mean(axis=1)
            )
        )
    else:
        nav_lower = nav_cmd.lower()
        final_lateral = pred_np[:, -1, 1]
        final_forward = pred_np[:, -1, 0]
        if "left" in nav_lower:
            sample_idx = int(np.argmax(final_lateral))
        elif "right" in nav_lower:
            sample_idx = int(np.argmin(final_lateral))
        elif "straight" in nav_lower:
            sample_idx = int(np.argmin(np.abs(final_lateral)))
        else:
            sample_idx = int(np.argmax(final_forward - np.abs(final_lateral)))
        selected = pred_np[sample_idx]

    return selected[:num_frames], sample_idx, pred_np


def run_inference(model, processor, data, device, args, nav_cmd):
    import torch
    from alpamayo1_5 import helper

    messages = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data.get("camera_indices"),
        nav_text=nav_cmd,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        device,
    )

    autocast_device = device.split(":", 1)[0]
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        return model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_gen_length,
            return_extra=True,
            diffusion_kwargs={
                "use_classifier_free_guidance": True,
                "inference_guidance_weight": args.guidance_weight,
                "temperature": 0.6,
            },
        )


def path_json(path_array):
    return [
        {
            "step_index": idx,
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "z_m": float(point[2]) if len(point) > 2 else 0.0,
        }
        for idx, point in enumerate(path_array)
    ]


def samples_json(samples_array, num_frames):
    return [
        {"sample_index": sample_idx, "path": path_json(sample[:num_frames])}
        for sample_idx, sample in enumerate(samples_array)
    ]


def insert_prediction(conn, row, args, nav_cmd, nav_source, selected_path, all_samples, gt_path, sample_idx, cot):
    selected_path_data = path_json(selected_path)
    cur = conn.execute(
        """
        INSERT INTO alpamayo_predictions (
            frame_id, annotation_id, model_name, nav_command, nav_command_source,
            selection_mode, selected_sample_index, num_traj_samples, guidance_weight,
            max_generation_length, frames_requested, frames_stored, cot,
            selected_path_json, all_samples_json, gt_path_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["frame_id"],
            row["annotation_id"],
            args.model_name,
            nav_cmd,
            nav_source,
            args.selection_mode,
            sample_idx,
            args.num_traj_samples,
            args.guidance_weight,
            args.max_gen_length,
            args.frames,
            len(selected_path_data),
            cot,
            json.dumps(selected_path_data),
            json.dumps(samples_json(all_samples, args.frames)),
            json.dumps(path_json(gt_path[: args.frames])) if gt_path is not None else None,
            datetime.utcnow().isoformat(),
        ),
    )
    prediction_id = cur.lastrowid
    conn.executemany(
        """
        INSERT INTO alpamayo_prediction_points (prediction_id, step_index, x_m, y_m, z_m)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (prediction_id, p["step_index"], p["x_m"], p["y_m"], p["z_m"])
            for p in selected_path_data
        ],
    )
    conn.commit()
    return prediction_id


def load_model(args):
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[WARN] Alpamayo is very large; CPU inference will be extremely slow.")

    print(f"[*] Loading Alpamayo model on {device}: {args.model_name}")
    model = Alpamayo1_5.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    if args.compile_model:
        model = torch.compile(model)
    processor = helper.get_processor(model.tokenizer)
    return model, processor, device


def main():
    args = parse_args()
    db_path = str(Path(args.db).resolve())
    conn = connect_db(db_path)
    rows = selected_rows(conn, args)

    print(f"[*] Selected {len(rows)} frame(s) from {db_path}")
    for row in rows[:10]:
        print(
            f"  frame_id={row['frame_id']} source={row['source']} "
            f"frame={row['frame_number']} image={resolve_image_path(db_path, row['relative_path'])}"
        )
    if len(rows) > 10:
        print(f"  ... {len(rows) - 10} more")
    if args.dry_run or not rows:
        return

    import torch
    from alpamayo1_5.load_custom_dataset import load_custom_dataset
    from alpamayo1_5.navigation_command import infer_navigation_command

    cam_mapping = {"left": 0, "wide": 1, "right": 2, "front": 6}
    excluded_cameras = [idx for name, idx in cam_mapping.items() if name not in args.cameras]
    model, processor, device = load_model(args)

    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(db_path, row["relative_path"])
        try:
            segment_dir, frame_idx = segment_and_frame_from_image(image_path)
            data = load_custom_dataset(str(segment_dir), frame_idx, exclude_cameras=excluded_cameras)
        except Exception as exc:
            print(f"[SKIP] frame_id={row['frame_id']}: {exc}")
            continue

        gt_xyz = data["ego_future_xyz"][0, 0].detach().cpu().numpy()
        if args.command:
            nav_cmd = args.command
            nav_source = "fixed"
        else:
            nav_cmd = infer_navigation_command(gt_xyz)
            nav_source = "ground_truth_heuristic"

        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(42)

        print(f"\n[{index}/{len(rows)}] frame_id={row['frame_id']} nav='{nav_cmd}'")
        try:
            pred_xyz, _, extra = run_inference(model, processor, data, device, args, nav_cmd)
            selected_path, sample_idx, all_samples = select_prediction(
                pred_xyz, nav_cmd, args.frames, args.selection_mode
            )
            cot = extract_cot(extra, sample_idx)
            prediction_id = insert_prediction(
                conn, row, args, nav_cmd, nav_source, selected_path, all_samples, gt_xyz, sample_idx, cot
            )
        except Exception as exc:
            print(f"[ERROR] Alpamayo failed for frame_id={row['frame_id']}: {exc}")
            continue

        print(f"[OK] prediction_id={prediction_id} points={len(selected_path)}")
        if cot:
            print(f"     reasoning: {cot[:240]}")

    conn.close()


if __name__ == "__main__":
    main()
