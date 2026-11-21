"""Offline teacher labels from complete three-camera recordings; never robot control.

Only explicit train review manifests are accepted. Chunk labels remain unreviewed;
no automatic cumulative count, cross-chunk deduplication, or SFT export.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1/chat/completions"
KEY_PATH = Path.home() / ".config/qwen-recognition-test/token-plan.key"
MODELS = ("qwen3.8-flash", "qwen3.8-max", "qwen3.6-flash")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for blob in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(blob)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def validate(answer, frames):
    """Check references only: passing this is not a visual correctness check."""
    if not isinstance(answer, dict):
        raise ValueError("answer must be object")
    def ref(value):
        if type(value) is not int or not 0 <= value < len(frames):
            raise ValueError("invalid image reference")
        return frames[value]["t"]
    events = answer.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be list")
    for event in events:
        if event["result"] not in {"delivered", "held", "failed_grasp", "dropped_outside", "unknown"}:
            raise ValueError("invalid result")
        for key in ("detached_i", "delivered_i"):
            if event[key] is not None:
                ref(event[key])
        if event["result"] == "delivered" and event["delivered_i"] is None:
            raise ValueError("delivery without evidence")
        if event["result"] != "delivered" and event["delivered_i"] is not None:
            raise ValueError("delivery conflicts with outcome")
        if event["detached_i"] is not None and event["delivered_i"] is not None:
            if ref(event["detached_i"]) > ref(event["delivered_i"]):
                raise ValueError("delivery precedes detachment")
        if not isinstance(event["evidence_i"], list) or not 1 <= len(event["evidence_i"]) <= 4:
            raise ValueError("missing bounded evidence")
        for i in event["evidence_i"]:
            ref(i)
        if not isinstance(event["reason"], str) or not event["reason"].strip():
            raise ValueError("missing reason")
    for key in ("observed_adjustments", "uncertain_ranges"):
        if not isinstance(answer.get(key), list):
            raise ValueError("missing interval list")
        for interval in answer[key]:
            if ref(interval["start_i"]) > ref(interval["end_i"]):
                raise ValueError("reversed interval")
    return answer


def sample(caps, start, end, fps, width):
    frames, images = [], []
    for k in range(math.floor((end - start) * fps) + 1):
        t = round(start + k / fps, 6)
        tiles, sources = [], {}
        for cam, cap in caps.items():
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            index = int(math.floor(t * native_fps + 1e-6))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, pixels = cap.read()
            if not ok:
                raise ValueError(f"failed decode {cam} frame {index}")
            h, w = pixels.shape[:2]
            resized = cv2.resize(pixels, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            tiles.append(resized)
            sources[cam] = {"frame": index, "t": index / native_fps, "original_size": [w, h]}
        height = max(tile.shape[0] for tile in tiles)
        tiles = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)) for tile in tiles]
        mosaic = cv2.hconcat(tiles)
        ok, encoded = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("JPEG encode failed")
        blob = encoded.tobytes()
        frames.append({"i": len(frames), "t": t, "sources": sources,
                       "jpeg_sha256": hashlib.sha256(blob).hexdigest()})
        images.append(blob)
    if sum(map(len, images)) > 12 * 1024 * 1024 or len(images) > 64:
        raise ValueError("input exceeds pilot request budget")
    return frames, images


def call(payload):
    key = KEY_PATH.read_text().strip()
    if not key:
        raise ValueError("missing API key")
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
                                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=90) as response:
            blob = response.read(2 * 1024 * 1024 + 1)
            if len(blob) > 2 * 1024 * 1024:
                raise ValueError("response too large")
            raw = json.loads(blob)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}; no automatic retry") from None
    choice = raw["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete response")
    if choice["message"].get("reasoning_content") or raw.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0):
        raise ValueError("unexpected thinking")
    return {"raw": raw, "api_s": time.perf_counter() - start,
            "answer": json.loads(choice["message"]["content"])}


def run(args):
    review = json.loads(args.review.read_text())
    if review["split"] != "train":
        raise ValueError("pilot accepts train only; do not label sealed tests")
    if not .5 <= args.fps <= 2 or not 5 <= args.chunk_seconds <= 30 or not 0 <= args.overlap < args.chunk_seconds:
        raise ValueError("invalid sampling budget")
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        raise ValueError("invalid start time")
    if args.chunk_seconds * args.fps > 63 or not 256 <= args.width <= 640:
        raise ValueError("request too large")
    prompt = (ROOT / "prompts/video_teacher_zh.txt").read_text()
    config = {"source_recording_id": review["source_recording_id"], "split": "train",
              "review_sha256": digest(args.review), "model": args.model, "fps": args.fps,
              "chunk_seconds": args.chunk_seconds, "overlap": args.overlap, "width_per_camera": args.width,
              "start_seconds": args.start_seconds,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "script_sha256": digest(__file__), "enable_thinking": False, "control_eligible": False,
              "review_status": "machine_unreviewed", "timeline": "dataset_frame_over_fps_not_verified_ros_sync"}
    run_path = args.output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != config:
        raise ValueError("resume configuration mismatch; use a new output directory")
    save(run_path, config)
    caps = {}
    try:
        for cam in ("cam0", "cam1", "cam2"):
            path = review["videos"][cam]
            if digest(path) != review["expected_video_sha256"][cam]:
                raise ValueError("source video hash mismatch")
            cap = cv2.VideoCapture(path)
            caps[cam] = cap
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - review["fps"]) > .01:
                raise ValueError("invalid video or FPS")
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != review["frames"]:
                raise ValueError("video length differs from manifest")
        duration = (review["frames"] - 1) / review["fps"]
        if args.start_seconds > duration:
            raise ValueError("start exceeds video duration")
        start, chunk_index = args.start_seconds, 0
        while start <= duration:
            end = min(duration, start + args.chunk_seconds)
            dest = args.output / f"chunk-{chunk_index:03d}.json"
            if dest.exists():
                prior = json.loads(dest.read_text())
                if prior.get("status") not in {"complete_structural_only", "quarantined_schema"}:
                    raise ValueError("previous error: inspect before retrying in new output directory")
            else:
                frames, images = sample(caps, start, end, args.fps, args.width)
                content = [{"type": "text", "text": f"录像分块 [{start:.3f}, {end:.3f}] 秒。无原用户目标数量。"}]
                for frame, blob in zip(frames, images):
                    content.extend([{"type": "text", "text": json.dumps({"i": frame["i"], "t": frame["t"]})},
                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(blob).decode()}}])
                payload = {"model": args.model, "enable_thinking": False, "stream": False,
                           "max_tokens": 2400, "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}]}
                record = {"start": start, "end": end, "frames": frames, "review_status": "machine_unreviewed"}
                try:
                    record.update(call(payload))
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    save(dest, record)
                    raise
                try:
                    validate(record["answer"], frames)
                    record["status"] = "complete_structural_only"
                except (ValueError, KeyError, TypeError) as exc:
                    # Preserve the teacher answer verbatim and quarantine; never repair a label silently.
                    record.update(status="quarantined_schema", validation_error=f"{type(exc).__name__}: {exc}")
                save(dest, record)
                print(json.dumps({"source": review["source_recording_id"], "chunk": chunk_index,
                                  "range": [start, end], "api_s": record["api_s"], "status": record["status"],
                                  "answer": record["answer"]}, ensure_ascii=False), flush=True)
            if end == duration:
                break
            start += args.chunk_seconds - args.overlap
            chunk_index += 1
        quarantined = [p.name for p in sorted(args.output.glob("chunk-*.json"))
                       if json.loads(p.read_text()).get("status") == "quarantined_schema"]
        save(args.output / "complete.json", {"chunks": chunk_index + 1, "start_seconds": args.start_seconds,
                                            "end_seconds": duration, "review_status": "machine_unreviewed",
                                            "quarantined_chunks": quarantined,
                                            "unique_count": None, "training_ready": False,
                                            "note": "Needs visual review and cross-chunk deduplication; not online latency evidence."})
    finally:
        for cap in caps.values():
            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="qwen3.8-max")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=30)
    parser.add_argument("--overlap", type=float, default=3)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--start-seconds", type=float, default=0,
                        help="Explicit partial-video diagnostic; not a complete-recording label when >0")
    run(parser.parse_args())
