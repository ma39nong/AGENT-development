"""Index unreviewed teacher candidates and warnings; do not infer true counts."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(inventory, directories):
    sources = {row["source_recording_id"]: row for row in inventory["recordings"]}
    recordings, candidates, timings, prompt_versions = {}, [], [], set()
    for directory in directories:
        config = json.loads((directory / "run.json").read_text())
        sid = config["source_recording_id"]
        row = sources[sid]
        if config["split"] != "train" or row["split"] != "train":
            raise ValueError("only train candidates allowed")
        prompt_versions.add(config["prompt_sha256"])
        group = recordings.setdefault(sid, {"split": "train", "chunks": 0, "quarantined_chunks": [],
                                            "ranges": [], "unique_success_count": None,
                                            "review_status": "machine_unreviewed"})
        for path in sorted(directory.glob("chunk-*.json")):
            chunk = json.loads(path.read_text())
            if "raw" not in chunk:
                raise ValueError("missing raw API response")
            group["chunks"] += 1
            group["ranges"].append([chunk["start"], chunk["end"]])
            usable = chunk["status"] == "complete_structural_only"
            if not usable:
                group["quarantined_chunks"].append(str(path.resolve()))
            timings.append(chunk["api_s"])
            for index, event in enumerate(chunk.get("answer", {}).get("events", [])):
                candidate = {"source_recording_id": sid, "chunk": str(path.resolve()),
                             "event_index": index, "result": event.get("result"),
                             "schema_usable": usable, "review_status": "machine_unreviewed",
                             "training_ready": False, "warnings": []}
                for key, target in (("detached_i", "detached_s"), ("delivered_i", "delivered_s")):
                    i = event.get(key)
                    frame = chunk["frames"][i] if type(i) is int and 0 <= i < len(chunk["frames"]) else None
                    candidate[target] = frame["t"] if frame else None
                    if frame:
                        source_index = frame["sources"]["cam0"]["frame"]
                        phase = next((s["label"] for s in row["coarse_segments"]
                                      if s["start_frame"] <= source_index < s["end_frame"]), None)
                        candidate[target + "_coarse_phase"] = phase
                        if phase != "harvest":
                            candidate["warnings"].append(target + "_outside_demonstrated_harvest")
                candidates.append(candidate)
    for sid, group in recordings.items():
        endpoint = 0.
        gap = False
        for start, end in sorted(group["ranges"]):
            if start > endpoint + 1e-6:
                gap = True
            endpoint = max(endpoint, end)
        duration = (sources[sid]["frames"] - 1) / sources[sid]["fps"]
        group["requested_ranges_cover_recording"] = not gap and endpoint >= duration - 1e-6
        group["duration_s"] = duration
    for n, first in enumerate(candidates):
        if first["delivered_s"] is None:
            continue
        for second in candidates[n + 1:]:
            if (first["source_recording_id"] == second["source_recording_id"]
                    and first["chunk"] != second["chunk"] and second["delivered_s"] is not None
                    and abs(first["delivered_s"] - second["delivered_s"]) <= 3):
                for item in (first, second):
                    if "possible_cross_chunk_duplicate" not in item["warnings"]:
                        item["warnings"].append("possible_cross_chunk_duplicate")
    return {"recordings": recordings, "candidates": candidates, "prompt_versions": sorted(prompt_versions),
            "recording_count": len(recordings), "request_count": len(timings),
            "api_s": {"median": statistics.median(timings), "min": min(timings), "max": max(timings)} if timings else None,
            "training_ready": False, "control_eligible": False,
            "limits": "Candidate counts are not true fruit counts. Phase mismatches and proximity only flag review; no silent repair or deduplication. Offline API timing is not online response latency."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.inventory.read_text()), args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("recording_count", "request_count", "api_s", "training_ready")}))
