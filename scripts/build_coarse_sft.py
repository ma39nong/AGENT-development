"""Create full-video coarse-only SFT manifests from frozen train/val sidecars."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

if __package__:
    from .audit_subtask_data import digest
else:
    from audit_subtask_data import digest

LABELS = {"pickup", "harvest", "adjust", "return"}


def target_json(segments, frames, fps):
    if frames <= 0 or fps <= 0:
        raise ValueError("invalid time base")
    end = 0
    entries = []
    for i, segment in enumerate(segments, 1):
        a, b = segment["start_frame"], segment["end_frame"]
        if type(a) is not int or type(b) is not int or a != end or not a < b <= frames:
            raise ValueError("segments must cover contiguous half-open frame intervals")
        if segment["label"] not in LABELS or segment.get("reviewed") is not True:
            raise ValueError("unsupported or unreviewed coarse annotation")
        start_s, end_s = round(a/fps, 2), round(b/fps, 2)
        if start_s >= end_s:
            raise ValueError("rounding collapses segment")
        entries.append('{"phase_index":'+str(i)+',"label":'+json.dumps(segment["label"])+
                       ',"start_seconds":'+f'{start_s:.2f}'+',"end_seconds":'+f'{end_s:.2f}'+'}')
        end = b
    if end != frames:
        raise ValueError("annotation does not cover full recording")
    return '{"segments":['+','.join(entries)+']}'


def build(inventory, output, prompt_path, camera="cam0"):
    if output.exists():
        raise ValueError("output exists")
    ids = [r["source_recording_id"] for r in inventory["recordings"]]
    if len(ids) != len(set(ids)):
        raise ValueError("source recording crosses split or duplicates")
    template = prompt_path.read_text()
    rows = {"train": [], "val": []}
    sequences = Counter()
    for record in inventory["recordings"]:
        if record["split"] not in rows:
            continue  # Do not even open sealed annotations or videos.
        annotation = Path(record["coarse_annotation"])
        if digest(annotation) != record["annotation_sha256"]:
            raise ValueError("coarse sidecar hash mismatch")
        ann = json.loads(annotation.read_text())
        source = ann["source"]
        if (source["source_recording_id"] != record["source_recording_id"]
                or source["fps"] != record["fps"] or source["length"] != record["frames"]):
            raise ValueError("annotation identity/time mismatch")
        video = Path(record["videos"][camera])
        checksum = digest(video)
        if checksum != record["expected_video_sha256"][camera]:
            raise ValueError("video differs from annotation-bound bytes")
        import cv2
        cap = cv2.VideoCapture(str(video))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if count != record["frames"] or abs(fps-record["fps"]) > .01:
            raise ValueError("video frame clock differs from annotations")
        answer = target_json(ann["segments"], count, fps)
        duration = f'{count/fps:.2f}'
        prompt = template.replace("{duration_seconds}", duration)
        row = {"source_recording_id": record["source_recording_id"], "split": record["split"],
               "video": str(video.resolve()), "video_sha256": checksum, "camera": camera,
               "frames": count, "fps": fps, "width": width, "height": height,
               "annotation": str(annotation.resolve()), "annotation_sha256": record["annotation_sha256"],
               "prompt": prompt, "answer": answer,
               "conversations": [{"from": "human", "value": "<video>\n"+prompt}, {"from": "gpt", "value": answer}],
               "scope": "coarse_only_no_count_or_control"}
        rows[record["split"]].append(row)
        sequences[tuple(s["label"] for s in ann["segments"])] += 1
    if not rows["train"] or not rows["val"]:
        raise ValueError("train and validation must both be present")
    output.mkdir(parents=True)
    for split, records in rows.items():
        (output/(split+".jsonl")).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    summary = {"counts": {k: len(v) for k, v in rows.items()}, "camera": camera,
               "sealed_records_not_opened": sum(r["split"] not in rows for r in inventory["recordings"]),
               "all_selected_video_sha256_checked": True, "all_selected_frame_clocks_checked": True,
               "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
               "source_lock_sha256": inventory["source_lock_sha256"],
               "manifests_sha256": {s: digest(output/(s+".jsonl")) for s in rows},
               "sequence_counts": {','.join(k): v for k, v in sequences.items()},
               "limitations": "All current examples have same five-stage sequence; variation in stage number/order is untested. Two decimals represent rounded source-frame annotations, not centisecond truth."}
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt", type=Path, default=Path(__file__).resolve().parents[1]/"prompts/coarse_sft_zh.txt")
    args = p.parse_args()
    print(json.dumps(build(json.loads(args.inventory.read_text()), args.output, args.prompt), ensure_ascii=False))
