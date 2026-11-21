"""Observation-history ablation; prerecorded open-loop replay, no robot commands."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

if __package__:
    from .annotate_video_api import ENDPOINT, KEY_PATH, NoRedirect, digest, save
else:
    from annotate_video_api import ENDPOINT, KEY_PATH, NoRedirect, digest, save

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (27, 37, 47, 62, 77, 86)


def cloud_call(payload):
    key = KEY_PATH.read_text().strip()
    if not key:
        raise ValueError("missing API key")
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
                                    headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=90) as response:
            blob = response.read(2*1024*1024+1)
            if len(blob) > 2*1024*1024:
                raise ValueError("response too large")
            return json.loads(blob), time.perf_counter()-start
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}") from None


def validate(answer, start, cutoff, mode, history):
    if not isinstance(answer, dict) or answer.get("phase") not in {"pickup", "harvest", "adjust", "return", "unknown"}:
        raise ValueError("invalid phase")
    stamps = answer["delivery_seconds"]
    if not isinstance(stamps, list) or any(type(t) not in (int, float) or not math.isfinite(t) or not start <= t <= cutoff for t in stamps):
        raise ValueError("event outside observed source-time range")
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("duplicate or reversed events")
    if mode == "recent_memory" and history and any(t <= start for t in stamps):
        raise ValueError("boundary event already belongs to previous window")
    count = answer["confirmed_total"]
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("invalid count")
    if count is not None and mode in {"prefix", "recent_memory"}:
        expected = len(stamps) + (len(history) if mode == "recent_memory" else 0)
        if count != expected:
            raise ValueError("count conflicts with event ledger")
    if type(answer["uncertain"]) is not bool:
        raise ValueError("uncertainty must be boolean")
    for task in ("target4", "two_then_two"):
        if answer[task]["request"] not in {"pickup", "harvest", "adjust", "return", "wait"}:
            raise ValueError("unsupported Prompt")
        if not isinstance(answer[task]["reason"], str):
            raise ValueError("missing reason")
    return answer


def prepare(review_path, output):
    import cv2
    review = json.loads(review_path.read_text())
    if review["split"] != "train":
        raise ValueError("train only")
    source = Path(review["videos"]["cam2"])
    if digest(source) != review["expected_video_sha256"]["cam2"]:
        raise ValueError("source hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    clips = []
    previous = 0
    for cutoff in CUTOFFS:
        for mode, start in (("prefix", 0), ("recent", previous)):
            target = output / f"{mode}-{cutoff:03d}.mp4"
            first = round(start * review["fps"])
            last = round(cutoff * review["fps"])
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vf",
                            f"trim=start_frame={first}:end_frame={last+1},setpts=N/({review['fps']}*TB)",
                            "-an", "-r", str(review["fps"]), "-fps_mode", "cfr",
                            "-c:v", "libx264", "-crf", "18", "-preset", "fast", str(target)], check=True)
            cap = cv2.VideoCapture(str(target))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if count != last-first+1 or abs(fps-review["fps"]) > .01:
                raise ValueError("prefix encoding changed frame clock")
            clips.append({"mode": mode, "start": start, "cutoff": cutoff, "path": str(target.resolve()),
                          "sha256": digest(target), "source_first_frame": first, "source_last_frame": last,
                          "frames": count, "fps": fps})
        previous = cutoff
    save(output / "manifest.json", {"source_recording_id": review["source_recording_id"], "split": "train",
                                     "source_sha256": digest(source), "clips": clips,
                                     "no_future_frames": True, "native_fps_preserved": True,
                                     "spatial_resize": False, "encoding": "h264_crf18_not_byte_identical",
                                     "limitations": "Open-loop recorded video will not follow proposed skill changes."})


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["split"] != "train" or not manifest["no_future_frames"]:
        raise ValueError("invalid replay manifest")
    if args.output.exists():
        raise ValueError("output exists")
    args.output.mkdir(parents=True)
    prompt = (ROOT / "prompts/streaming_baseline_zh.txt").read_text()
    metadata = {"backend": args.backend, "cloud_model": "qwen3.8-max" if args.backend == "api" else None,
                "local_model": "Qwen3-VL-4B-Instruct" if args.backend == "local" else None,
                "manifest_sha256": digest(args.manifest), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "script_sha256": digest(__file__), "robot_control": False, "training": False,
                "memory": "prior_model_candidates_never_reference_labels", "reference_counts_supplied": False,
                "modes": ["recent_only", "recent_memory", "prefix"],
                "max_new_tokens": 1024 if args.backend == "api" else 512,
                "cloud_response_format": "json_object" if args.backend == "api" else None,
                "execution": "open_loop_prerecorded_counterfactual_decisions_only"}
    if args.backend == "local":
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from transformers.video_utils import load_video
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, dtype=torch.bfloat16,
                    device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        metadata["load_s"] = time.perf_counter()-start
    save(args.output / "run.json", metadata)
    for mode in metadata["modes"]:
        history = []
        memory_complete = True
        for cutoff in CUTOFFS:
            clip = next(c for c in manifest["clips"] if c["cutoff"] == cutoff and c["mode"] == ("prefix" if mode == "prefix" else "recent"))
            context = {"mode": mode, "source_start_s": clip["start"], "observation_cutoff_s": cutoff,
                       "timestamp_format": "absolute original video seconds, display 0.00s; actual precision not guaranteed",
                       "executor_actual_prompt": "adjust" if cutoff == 47 else "harvest",
                       "executor_adjustment_completed": cutoff >= 62,
                       "replay_does_not_follow_your_proposals": True,
                       "historical_candidate_delivery_seconds": history if mode == "recent_memory" else None,
                       "history_is_complete_since_task_start": memory_complete if mode == "recent_memory" else mode == "prefix"}
            question = prompt + "\n本次输入说明：" + json.dumps(context, ensure_ascii=False)
            record = {"mode": mode, "cutoff": cutoff, "clip": clip, "context": context,
                      "control_eligible": False, "training_ready": False}
            begin = time.perf_counter()
            try:
                if digest(clip["path"]) != clip["sha256"]:
                    raise ValueError("clip changed")
                if args.backend == "api":
                    blob = Path(clip["path"]).read_bytes()
                    payload = {"model": "qwen3.8-max", "enable_thinking": False, "stream": False, "max_tokens": 1024,
                               "response_format": {"type": "json_object"},
                               "messages": [{"role": "user", "content": [
                                   {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,"+base64.b64encode(blob).decode()}},
                                   {"type": "text", "text": question}]}]}
                    raw_response, api_s = cloud_call(payload)
                    record["raw_response"] = raw_response
                    record["raw_output"] = raw_response["choices"][0]["message"]["content"]
                    record["api_s"] = api_s
                    choice = raw_response["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("incomplete response, raw retained")
                    if choice["message"].get("reasoning_content") or raw_response.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0):
                        raise ValueError("unexpected thinking, raw retained")
                    answer = json.loads(record["raw_output"])
                else:
                    video, video_meta = load_video(clip["path"], backend="opencv", fps=2)
                    text = processor.apply_chat_template([{"role": "user", "content": [
                        {"type": "video"}, {"type": "text", "text": question}]}], tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], videos=[video], video_metadata=[video_meta], do_sample_frames=False,
                                       return_tensors="pt").to("cuda:0")
                    torch.cuda.synchronize()
                    record["preparation_s"] = time.perf_counter()-begin
                    generation_start = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
                    torch.cuda.synchronize()
                    ids = output[0, inputs["input_ids"].shape[-1]:]
                    raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
                    record.update(raw_output=raw, input_tokens=inputs["input_ids"].shape[-1],
                                  generation_s=time.perf_counter()-generation_start,
                                  sampled_last_source_frame=clip["source_first_frame"]+int(video_meta.frames_indices[-1]))
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos
                    if not ids.numel() or int(ids[-1]) not in eos:
                        raise ValueError("incomplete generation")
                    del inputs, output, ids, video
                    answer = json.loads(raw)
                record["answer"] = answer
                validate(answer, clip["start"], cutoff, mode, history)
                record["structural_valid"] = True
                if mode == "recent_memory":
                    history = history + answer["delivery_seconds"]
                    memory_complete = memory_complete and not answer["uncertain"] and answer["confirmed_total"] is not None
            except Exception as exc:
                record.update(structural_valid=False, error=type(exc).__name__+": "+str(exc))
                if mode == "recent_memory":
                    memory_complete = False
            record["total_s"] = time.perf_counter()-begin
            save(args.output / f"{mode}-{cutoff:03d}.json", record)
            print(json.dumps({k: record.get(k) for k in ("mode", "cutoff", "total_s", "answer", "error")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--review", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--backend", choices=("local", "api"), required=True)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--model-dir", type=Path, default=Path("/home/user/models/Qwen3-VL-4B-Instruct"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.review, args.output)
    else:
        run(args)
