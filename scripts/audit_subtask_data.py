"""Inventory source metadata; inspect train/val labels, never decode held-out video."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SPLITS = {"train", "val", "historical_test", "blind_test"}
CAMERAS = ("cam0", "cam1", "cam2")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def under(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes root")
    return path


def audit(root, lock_path, fallback_root=None):
    root = root.resolve()
    lock = json.loads(lock_path.read_text())
    records = lock["recordings"]
    if not isinstance(records, list) or not records:
        raise ValueError("Empty recording lock")
    ids = [r["source_recording_id"] for r in records]
    if any(not isinstance(s, str) or not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate source recording ID across splits")
    counts = Counter(r["split"] for r in records)
    if not set(counts) <= SPLITS or dict(counts) != lock["counts"]:
        raise ValueError("Invalid or mismatching locked split counts")
    packages, rows = {}, []
    for record in records:
        pack = under(root, record["package"])
        if pack not in packages:
            info_path, episodes_path = pack / "meta/info.json", pack / "meta/episodes.jsonl"
            info = json.loads(info_path.read_text())
            episodes = [json.loads(s) for s in episodes_path.read_text().splitlines() if s.strip()]
            by_index = {e["episode_index"]: e for e in episodes}
            if len(by_index) != len(episodes):
                raise ValueError("Duplicate episode index")
            for camera in CAMERAS:
                if "observation.images." + camera not in info["features"]:
                    raise ValueError("Missing original camera metadata")
            packages[pack] = (info, by_index)
        info, episodes = packages[pack]
        index = record["source_episode_index"]
        if type(index) is not int or index < 0:
            raise ValueError("Invalid episode index")
        episode = episodes[index]
        if episode["source_recording_id"] != record["source_recording_id"] or episode["length"] != record["length"]:
            raise ValueError("Recording identity/length differs from frozen lock")
        formatting = {"episode_chunk": index // info.get("chunks_size", 1000), "episode_index": index}
        parquet = under(pack, info["data_path"].format(**formatting))
        videos = {c: under(pack, info["video_path"].format(**formatting, video_key="observation.images." + c))
                  for c in CAMERAS}
        annotation = pack / "annotations/subtasks/coarse" / f"episode_{index:06d}.json"
        available = {"parquet": parquet.is_file(), "coarse_annotation": annotation.is_file(),
                     **{c: p.is_file() for c, p in videos.items()}}
        row = {"source_recording_id": record["source_recording_id"], "split": record["split"],
               "package": record["package"], "episode_index": index, "frames": episode["length"],
               "fps": info["fps"], "available": available,
               "videos": {c: str(p) for c, p in videos.items()}, "parquet": str(parquet),
               "coarse_annotation": str(annotation), "label_inspection": "sealed_not_read"}
        if record["split"] in {"train", "val"}:
            row["label_inspection"] = "missing_annotation"
            if not available["coarse_annotation"]:
                rows.append(row)
                continue
            annotation_sha = digest(annotation)
            if annotation_sha != record["annotation_sha256"]:
                raise ValueError("Annotation differs from frozen lock")
            ann = json.loads(annotation.read_text())
            source = ann["source"]
            if (source["source_recording_id"] != record["source_recording_id"]
                    or source["episode_index"] != index or source["length"] != episode["length"]
                    or source["fps"] != info["fps"]):
                raise ValueError("Annotation source identity/time metadata mismatch")
            resolved = dict(videos)
            fallback_verified = {}
            # Explicitly allowed root, train/val only, exact annotation-bound video bytes.
            if fallback_root is not None:
                donor = Path(source.get("dataset", "")).resolve()
                if donor.is_relative_to(fallback_root.resolve()):
                    for camera, original in videos.items():
                        if original.is_file():
                            continue
                        relative = info["video_path"].format(**formatting, video_key="observation.images." + camera)
                        candidate = under(donor, relative)
                        expected = source.get("sha256", {}).get(relative)
                        if candidate.is_file() and expected is not None:
                            actual = digest(candidate)
                            if actual != expected:
                                raise ValueError("Fallback video bytes do not match frozen annotation")
                            resolved[camera] = candidate
                            fallback_verified[camera] = actual
            previous_end = 0
            segments = []
            for segment in ann["segments"]:
                start, end = segment["start_frame"], segment["end_frame"]
                if (type(start) is not int or type(end) is not int or start != previous_end
                        or not start < end <= episode["length"]):
                    raise ValueError("Coarse segments must be contiguous half-open frame intervals")
                segments.append({"label": segment["label"], "start_frame": start,
                                 "end_frame": end, "reviewed": segment.get("reviewed") is True})
                previous_end = end
            if previous_end != episode["length"]:
                raise ValueError("Coarse annotation does not cover recording")
            row.update(annotation_sha256=annotation_sha, label_inspection="coarse_only",
                       coarse_segments=segments, prompt_profile=ann.get("prompt_profile"),
                       inherited=ann.get("inherited"),
                       videos={c: str(p) for c, p in resolved.items()},
                       resolved_video_available={c: p.is_file() for c, p in resolved.items()},
                       fallback_video_sha256_verified=fallback_verified,
                       expected_video_sha256={c: source.get("sha256", {}).get(str(p.relative_to(pack)))
                                              for c, p in videos.items()},
                       per_fruit_events_verified=False,
                       available_annotation_types=sorted(p.name for p in annotation.parent.parent.iterdir() if p.is_dir()))
        rows.append(row)
    root_manifest = root / "manifest.json"
    declared = json.loads(root_manifest.read_text()) if root_manifest.is_file() else {}
    warnings = []
    if declared.get("training_ready") is False:
        warnings.append("Root manifest declares training_ready=false; later coarse sidecars are checked separately, not auto-approved for Qwen SFT.")
    return {"schema_version": 1, "root": str(root), "source_lock": str(lock_path.resolve()),
            "source_lock_sha256": digest(lock_path), "recordings": rows, "counts": dict(counts),
            "summary": {"packages": len(packages), "recordings": len(rows),
                        "frames": sum(r["frames"] for r in rows),
                        "train_val_coarse_inspected": sum(r["label_inspection"] == "coarse_only" for r in rows),
                        "heldout_recordings_sealed": sum(r["label_inspection"] == "sealed_not_read" for r in rows),
                        "available_original_video_files": sum(r["available"][c] for r in rows for c in CAMERAS),
                        "resolved_train_val_video_files": sum(
                            sum(r.get("resolved_video_available", {}).values()) for r in rows),
                        "fallback_video_hashes_verified": sum(
                            len(r.get("fallback_video_sha256_verified", {})) for r in rows),
                        "train_val_recordings_with_all_three_videos": sum(
                            r.get("label_inspection") == "coarse_only"
                            and all(r.get("resolved_video_available", {}).get(c, False) for c in CAMERAS)
                            for r in rows),
                        "availability_checked_by_file_presence_only": True,
                        "video_pixels_decoded": False, "all_video_hashes_verified": False,
                        "parquet_rows_verified": False, "per_fruit_events_verified": False,
                        "ready_for_qwen_training": False}, "warnings": warnings}


def review_queue(inventory, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError("Review limit must be positive")
    # Choose only train recordings, round-robin across packages, without inspecting heldout labels/pixels.
    groups = {}
    for row in inventory["recordings"]:
        if (row["split"] == "train" and row["label_inspection"] == "coarse_only"
                and row["available"]["parquet"]
                and all(row["resolved_video_available"].values())):
            groups.setdefault(row["package"], []).append(row)
    by_side = {}
    for package in groups:
        by_side.setdefault(package.split("/")[0], []).append(package)
    ordered_packages = []
    while any(by_side.values()):
        for side in sorted(by_side):
            if by_side[side]:
                ordered_packages.append(by_side[side].pop(0))
    selected = []
    while len(selected) < limit and any(groups.values()):
        for package in ordered_packages:
            group = groups[package]
            if group and len(selected) < limit:
                row = group.pop(0)
                selected.append({"schema_version": 1, "review_status": "unreviewed",
                                 "source_recording_id": row["source_recording_id"],
                                 "split": "train", "package": row["package"],
                                 "episode_index": row["episode_index"], "frames": row["frames"],
                                 "fps": row["fps"], "videos": row["videos"],
                                 "expected_video_sha256": row["expected_video_sha256"],
                                 "verified_fallback_video_sha256": row["fallback_video_sha256_verified"],
                                 "all_video_bytes_verified": False,
                                 "coarse_annotation_sha256": row["annotation_sha256"],
                                 "task_goal_count": None, "explicit_user_constraints": None,
                                 "attempts": [], "events": [], "decisions": [],
                                 "notes": "Do not infer requested count or adjustment necessity from a full-truss demonstration."})
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--source-lock", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fallback-root", type=Path)
    p.add_argument("--review-limit", type=int, default=5)
    args = p.parse_args()
    inventory = audit(args.root.expanduser(), args.source_lock.expanduser(),
                      args.fallback_root.expanduser() if args.fallback_root else None)
    queue = review_queue(inventory, args.review_limit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    for row in queue:
        (args.output / f"review-{row['source_recording_id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": inventory["summary"], "counts": inventory["counts"],
                      "review_queue_size": len(queue), "warnings": inventory["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
