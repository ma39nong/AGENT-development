"""Make stage-balanced causal prefixes from the frozen train/val manifests only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import subprocess

if __package__:
    from .audit_subtask_data import digest
    from .build_coarse_sft import target_json
    from .train_coarse_lora import write_json
else:
    from audit_subtask_data import digest
    from build_coarse_sft import target_json
    from train_coarse_lora import write_json


def clipped_segments(segments, cutoff):
    result = []
    for segment in segments:
        if segment["start_frame"] >= cutoff:
            break
        item = copy.deepcopy(segment)
        item["end_frame"] = min(item["end_frame"], cutoff)
        result.append(item)
    return result


def build(source, output, prompt_path, workers=4):
    if output.exists():
        raise ValueError("output exists")
    summary = json.loads((source/"summary.json").read_text())
    original = {}
    for split in ("train", "val"):
        file = source/(split+".jsonl")
        if digest(file) != summary["manifests_sha256"][split]:
            raise ValueError("frozen source manifest changed")
        original[split] = [json.loads(s) for s in file.read_text().splitlines()]
        if any(r["split"] != split for r in original[split]):
            raise ValueError("source split mismatch")
    if {r["source_recording_id"] for r in original["train"]} & {r["source_recording_id"] for r in original["val"]}:
        raise ValueError("source recording leakage")
    output.mkdir(parents=True)
    (output/"videos").mkdir()
    template = prompt_path.read_text()
    jobs = []
    for split, rows in original.items():
        for row in rows:
            if digest(Path(row["annotation"])) != row["annotation_sha256"] or digest(Path(row["video"])) != row["video_sha256"]:
                raise ValueError("annotation or source video changed")
            segments = json.loads(Path(row["annotation"]).read_text())["segments"]
            for phase, segment in enumerate(segments, 1):
                # One midpoint per coarse phase. No full-duration or future stage in input.
                cutoff = (segment["start_frame"]+segment["end_frame"])//2
                if cutoff <= segment["start_frame"]:
                    raise ValueError("phase too short")
                target = target_json(clipped_segments(segments, cutoff), cutoff, row["fps"])
                jobs.append((row, split, phase, cutoff, target))

    def encode(job):
        row, split, phase, cutoff, target = job
        identity = row["source_recording_id"]+f'-prefix-{cutoff:06d}'
        video = output/"videos"/(identity+".mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", row["video"],
                        "-frames:v", str(cutoff), "-an", "-c:v", "libx264", "-threads", "1",
                        "-preset", "ultrafast", "-crf", "18", str(video)], check=True)
        import cv2
        cap = cv2.VideoCapture(str(video))
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if frames != cutoff or abs(fps-row["fps"]) > .01:
            raise ValueError("prefix encoding changed frame clock")
        return {"sample_id": identity, "source_recording_id": row["source_recording_id"], "split": split,
                "video": str(video.resolve()), "video_sha256": digest(video), "frames": cutoff, "fps": fps,
                "source_video": row["video"], "source_video_sha256": row["video_sha256"],
                "annotation_sha256": row["annotation_sha256"], "camera": row["camera"],
                "prompt": template.replace("{duration_seconds}", f'{cutoff/fps:.2f}'), "answer": target,
                "current_phase_index": phase, "current_phase": json.loads(target)["segments"][-1]["label"],
                "source_last_frame": cutoff-1, "source_last_seconds": (cutoff-1)/fps,
                "scope": "observed_coarse_prefix_only_no_count_or_control"}

    result = {"train": [], "val": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(encode, jobs), 1):
            result[row["split"]].append(row)
            if index % 50 == 0:
                print(json.dumps({"prefixes_encoded": index, "total": len(jobs)}), flush=True)
    for split, rows in result.items():
        (output/(split+".jsonl")).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    report = {"counts": {s: len(v) for s, v in result.items()},
              "source_recordings": {s: len(v) for s, v in original.items()},
              "source_lock_sha256": summary["source_lock_sha256"], "camera": "cam0",
              "prompt_sha256": digest(prompt_path),
              "manifests_sha256": {s: digest(output/(s+".jsonl")) for s in result},
              "physical_prefix_video": True, "source_last_frame_strictly_before_cutoff": True,
              "sampling": "one midpoint per annotated phase; pilot only, boundary density not covered",
              "test_or_new_recordings_used": False, "label_scope": "last stage right censored at cutoff; no future phases",
              "encoding": "h264 crf18 ultrafast; same frame clock, not byte-identical pixels"}
    write_json(output/"summary.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", type=Path, default=Path(__file__).resolve().parents[1]/"prompts/coarse_prefix_sft_zh.txt")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    print(json.dumps(build(args.source, args.output, args.prompt, args.workers), ensure_ascii=False))
