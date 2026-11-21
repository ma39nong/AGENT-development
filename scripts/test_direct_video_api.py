"""Send original MP4 bytes as video_url; no client extraction, resize or chunks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from annotate_video_api import ENDPOINT, KEY_PATH, MODELS, NoRedirect, digest, save

PROMPT = "请观看这段番茄采摘视频，按时间顺序列出每次成功采摘：第几颗、大约何时摘下、何时放入收集盘。最后说明一共成功摘下几颗、成功放入盘中几颗。看不清的地方明确说明，不要猜测。时间使用相对视频开头的分:秒。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--camera", choices=("cam0", "cam1", "cam2"), required=True)
    parser.add_argument("--additional-camera", choices=("cam0", "cam1", "cam2"), action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--model", choices=MODELS, default="qwen3.8-max")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; do not silently repeat paid requests")
    review = json.loads(args.review.read_text())
    if review["split"] != "train":
        raise ValueError("direct-video development test accepts train only")
    cameras = [args.camera] + args.additional_camera
    if len(cameras) != len(set(cameras)):
        raise ValueError("duplicate camera")
    prompt = args.prompt_file.read_text() if args.prompt_file else PROMPT
    content, sources = [], []
    for camera in cameras:
        path = Path(review["videos"][camera])
        checksum = digest(path)
        if checksum != review["expected_video_sha256"][camera]:
            raise ValueError("video hash mismatch")
        blob = path.read_bytes()
        sources.append({"camera": camera, "source": str(path), "video_sha256": checksum, "video_bytes": len(blob)})
        if len(cameras) > 1:
            content.append({"type": "text", "text": f"同步视角{camera}；cam0头部，cam1右腕，cam2左腕。各视频从同一时刻开始，重复视角不重复计数。"})
        content.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(blob).decode()}})
    content.append({"type": "text", "text": prompt})
    payload = {"model": args.model, "enable_thinking": False, "stream": False, "max_tokens": 2400,
               "messages": [{"role": "user", "content": content}]}
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    meta = {"source_recording_id": review["source_recording_id"], "split": "train", "sources": sources,
            "camera": cameras, "payload_bytes": len(encoded),
            "frames": review["frames"], "fps": review["fps"], "duration_s": review["frames"] / review["fps"],
            "model": args.model, "enable_thinking": False, "prompt": prompt,
            "client_frame_extraction": False, "client_resize": False, "client_chunking": False,
            "server_sampling": "unspecified; provider default", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "control_eligible": False, "training_ready": False}
    save(args.output / "metadata.json", meta)
    key = KEY_PATH.read_text().strip()
    if not key:
        raise ValueError("missing API key")
    request = urllib.request.Request(ENDPOINT, data=encoded,
                                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    start = time.perf_counter()
    result = {}
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=180) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("response too large")
            result = {"http_status": response.status, "raw": json.loads(raw)}
    except urllib.error.HTTPError as exc:
        result = {"http_status": exc.code, "error_body": exc.read(20000).decode(errors="replace")}
    except Exception as exc:
        result = {"error": type(exc).__name__ + ": " + str(exc)}
    result["total_s"] = time.perf_counter() - start
    save(args.output / "result.json", result)
    print(json.dumps({"camera": args.camera, **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
