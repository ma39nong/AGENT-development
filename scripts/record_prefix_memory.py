"""Import replay observations into persistent memory, without confirming events."""
import argparse
import json
from pathlib import Path

if __package__:
    from .observation_memory import ObservationMemory
else:
    from observation_memory import ObservationMemory


def record(run, database):
    protocol = json.loads((run/"protocol.json").read_text())
    source = protocol["source"]
    task = run.name+":"+source["source_recording_id"]
    memory = ObservationMemory(database)
    added = 0
    try:
        for r in json.loads((run/"results.json").read_text()):
            added += memory.append(task, f'prefix-{r["source_last_frame"]}', r["cutoff_seconds"],
                r["simulated_result_available_seconds"],
                [{"video": source["video"], "video_sha256": source["video_sha256"],
                  "source_last_frame": r["source_last_frame"], "source_last_seconds": r["source_last_frame_seconds"]}],
                {"raw_output": r["raw_output"], "current_phase": r["current_phase"], "schema_check": r["schema_check"]},
                protocol["adapter"])
        return {"task": task, "added": added, "database": str(database.resolve()),
                "time_basis": "simulated source replay seconds, not wall-clock timestamps",
                "query_at_20s": memory.query(task, cutoff=20, available_at=20, limit=5),
                "memory_fed_to_model": False, "control_eligible": False}
    finally:
        memory.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(record(args.run, args.database), ensure_ascii=False, indent=2))
