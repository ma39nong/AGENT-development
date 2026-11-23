"""Frozen-model causal probes: full prefix vs bounded windows with/without memory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import time

if __package__:
    from .audit_subtask_data import digest
    from .observation_memory import ObservationMemory
    from .train_coarse_lora import write_json
else:
    from audit_subtask_data import digest
    from observation_memory import ObservationMemory
    from train_coarse_lora import write_json


def parse_observation(raw):
    answer = json.loads(raw)
    if answer.get("phase") not in {"pickup", "harvest", "adjust", "return", "unknown"}:
        raise ValueError("unsupported current phase")
    index = answer.get("phase_index")
    if index is not None and (type(index) is not int or not 1 <= index <= 5):
        raise ValueError("invalid phase index")
    if type(answer.get("uncertain")) is not bool:
        raise ValueError("missing uncertainty")
    return answer


def select_history(records, start, token_count, budget=384, maximum=2):
    """Only pre-window candidates; never select reference labels or unavailable results."""
    selected, seen = [], set()
    for r in records:
        p = r["prediction"]
        if r["observation_cutoff_seconds"] > start or not p.get("valid") or p["phase"] == "unknown":
            continue
        key = (p["phase"], p["phase_index"])
        if key in seen:
            continue
        item = {"observed_until": r["observation_cutoff_seconds"], "candidate_phase": p["phase"],
                "candidate_phase_index": p["phase_index"], "uncertain": p["uncertain"],
                "status": "unverified", "evidence_seconds": r["evidence"][0]["source_last_seconds"]}
        proposed = selected+[(item, r["evidence"][0])]
        if token_count(json.dumps([x[0] for x in proposed], ensure_ascii=False)) > budget:
            continue
        selected = proposed
        seen.add(key)
        if len(selected) == maximum:
            break
    return selected


def summarize(rows):
    return {"samples": len(rows), "valid_outputs": sum(r["valid"] for r in rows),
            "current_label_accuracy_all": sum(r["label_correct"] for r in rows)/len(rows),
            "current_phase_index_accuracy_all": sum(r["index_correct"] for r in rows)/len(rows),
            "median_prepare_generate_s": statistics.median(r["prepare_generate_s"] for r in rows),
            "max_prepare_generate_s": max(r["prepare_generate_s"] for r in rows),
            "median_input_tokens": statistics.median(r["input_tokens"] for r in rows),
            "max_input_tokens": max(r["input_tokens"] for r in rows),
            "abstentions": sum(r.get("answer", {}).get("phase") == "unknown" for r in rows),
            "history_items_total": sum(len(r["history"]) for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path, help="Adapter override for post-training causal replay")
    p.add_argument("--recordings", type=int, default=3)
    p.add_argument("--only-mode", choices=["prefix", "window8", "window16", "window32", "memory8", "memory16", "memory32"])
    p.add_argument("--current-frame", action="store_true", help="Exploratory explicit newest-frame input after history")
    p.add_argument("--prepare-training", action="store_true", help="Frozen candidate rollout on train/val midpoints; export bounded SFT inputs")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    data = root/"data/derived/coarse_sft_20261123_v1"
    rows = sorted([json.loads(s) for s in (data/"val.jsonl").read_text().splitlines()], key=lambda r: r["source_recording_id"])[:args.recordings]
    if args.prepare_training:
        rows = [json.loads(s) for split in ("train", "val") for s in (data/(split+".jsonl")).read_text().splitlines()]
    modes = [("prefix", None, False)]+[(f"window{w}", w, False) for w in (8, 16, 32)]+[(f"memory{w}", w, True) for w in (8, 16, 32)]
    if args.only_mode:
        modes = [m for m in modes if m[0] == args.only_mode]
    if args.current_frame:
        modes = [(m+"_current_frame", w, memory) for m, w, memory in modes]
    if args.prepare_training:
        modes = [("memory8", 8, True)]
    adapter = root/"runs/coarse_lora_20261123_v1/best"
    if args.adapter:
        adapter = args.adapter
    jobs = []
    for row in rows:
        if row["split"] not in ({"train", "val"} if args.prepare_training else {"val"}) or digest(Path(row["video"])) != row["video_sha256"] or digest(Path(row["annotation"])) != row["annotation_sha256"]:
            raise ValueError("validation identity changed")
        segments = json.loads(Path(row["annotation"]).read_text())["segments"]
        cutoffs = {(s["start_frame"]+s["end_frame"])//2 for s in segments}
        for s in ([] if args.prepare_training else segments[:-1]):
            cutoffs.update([s["end_frame"]-round(row["fps"]), s["end_frame"]+round(row["fps"])])
        for cutoff in sorted(c for c in cutoffs if 0 < c < row["frames"]):
            # Last supplied frame is cutoff-1; scoring refers to this exact observed frame.
            phase = next(i for i, s in enumerate(segments, 1) if s["start_frame"] <= cutoff-1 < s["end_frame"])
            jobs.append({"row": row, "cutoff_frame": cutoff, "target_phase": segments[phase-1]["label"], "target_index": phase})
    write_json(args.output/"protocol.json", {"adapter": str(adapter), "adapter_sha256": digest(adapter/"adapter_model.safetensors"),
        "recording_ids": [r["source_recording_id"] for r in rows], "split": "train_and_val" if args.prepare_training else "validation",
        "modes": modes, "probes_per_mode": len(jobs), "training": False, "robot_control": False,
        "explicit_newest_frame": args.current_frame,
        "history": "previous actual model candidates only, at most 2 distinct label/index candidates and 2 original sampled frames; 384 text-token budget",
        "memory_availability": "observation cutoff and estimated result receipt must be <= current cutoff",
        "sampling": "annotation-stratified phase midpoints and +/-1s boundary probes; labels used only for coverage/scoring",
        "limitations": "Frozen whole-video adapter, new short-output prompt. Offline causal probes, not continuous live execution; receipt estimate excludes queue/transport; no persistent KV reuse."})
    import cv2
    from PIL import Image
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import load_video
    torch.set_num_threads(8)
    torch.manual_seed(20261123)
    processor = AutoProcessor.from_pretrained("/home/user/models/Qwen3-VL-4B-Instruct", local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained("/home/user/models/Qwen3-VL-4B-Instruct", dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    memory = ObservationMemory(args.output/"memory.sqlite")
    predictions = []
    prompt = (root/"prompts/bounded_history_probe_zh.txt").read_text()
    write_json(args.output/"prompt.json", {"template": prompt, "sha256": digest(root/"prompts/bounded_history_probe_zh.txt")})
    try:
        for mode, width, use_memory in modes:
            for job in jobs:
                row, cutoff_frame = job["row"], job["cutoff_frame"]
                cutoff = cutoff_frame/row["fps"]
                first = 0 if width is None else max(0, cutoff_frame-round(width*row["fps"]))
                start = first/row["fps"]
                task = mode+":"+row["source_recording_id"]
                clip = args.output/(f"{row['source_recording_id']}-{cutoff_frame}.mp4" if args.prepare_training else "current.mp4")
                # Physical trim means even the video loader cannot access future frames.
                subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-threads", "1", "-i", row["video"],
                    "-vf", f"trim=start_frame={first}:end_frame={cutoff_frame},setpts=N/({row['fps']}*TB)", "-an",
                    "-r", str(row["fps"]), "-fps_mode", "cfr",
                    "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-crf", "18", str(clip)], check=True)
                cap = cv2.VideoCapture(str(clip))
                count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if count != cutoff_frame-first or abs(fps-row["fps"]) > .01:
                    raise ValueError("physical window frame clock mismatch")
                begin = time.perf_counter()
                video, metadata = load_video(str(clip), backend="opencv", fps=2)
                history = select_history(memory.query(task, cutoff=start, available_at=cutoff, limit=100), start,
                    lambda text: len(processor.tokenizer.encode(text))) if use_memory else []
                context = {"window_start_seconds": round(start, 3), "observation_cutoff_seconds": round(cutoff, 3),
                           "history_candidates": [x[0] for x in history]}
                question = prompt+"\n本次观测："+json.dumps(context, ensure_ascii=False)
                content = [{"type": "text", "text": question}, {"type": "video"}]
                images = []
                for item, evidence in history:
                    content += [{"type": "text", "text": f'历史原始帧，源时间{item["evidence_seconds"]:.3f}s；不是当前画面。'}, {"type": "image"}]
                    im = Image.open(evidence["image_path"]).convert("RGB")
                    im.thumbnail((448, 280))
                    images.append(im)
                newest = None
                if args.current_frame:
                    newest_path = args.output/f'{row["source_recording_id"]}-current-{cutoff_frame-1:06d}.jpg'
                    cap = cv2.VideoCapture(row["video"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cutoff_frame-1)
                    ok, frame = cap.read()
                    cap.release()
                    if not ok or not cv2.imwrite(str(newest_path), frame):
                        raise ValueError("cannot read current frame")
                    newest = {"source_frame": cutoff_frame-1, "source_seconds": (cutoff_frame-1)/row["fps"],
                              "image_path": str(newest_path.resolve()), "image_sha256": digest(newest_path)}
                    im = Image.open(newest_path).convert("RGB")
                    im.thumbnail((640, 400))
                    images.append(im)
                    content += [{"type": "text", "text": f'下面是当前最新头部原始帧，源时间{newest["source_seconds"]:.3f}秒。请结合视频末尾最后一秒判断当前正在进行的动作，不要复述较早画面的动作。'}, {"type": "image"}]
                text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                kwargs = {"images": images} if images else {}
                inputs = processor(text=[text], videos=[video], video_metadata=[metadata], do_sample_frames=False,
                                   return_tensors="pt", **kwargs).to("cuda:0")
                tokens = inputs["input_ids"].shape[-1]
                if tokens > 32768:
                    raise ValueError("input token cap exceeded; no silent truncation")
                torch.cuda.synchronize()
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=160, do_sample=False, use_cache=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-begin
                raw = processor.tokenizer.decode(output[0, tokens:], skip_special_tokens=True)
                answer, error = {}, None
                try:
                    answer = parse_observation(raw)
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    error = str(exc)
                sampled = first+int(max(metadata.frames_indices))
                if sampled >= cutoff_frame:
                    raise ValueError("future frame sampled")
                image_path = args.output/f'{row["source_recording_id"]}-{sampled:06d}.jpg'
                if not image_path.exists():
                    cap = cv2.VideoCapture(row["video"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sampled)
                    ok, frame = cap.read()
                    cap.release()
                    if not ok or not cv2.imwrite(str(image_path), frame):
                        raise ValueError("cannot retain sampled evidence")
                memory.append(task, str(cutoff_frame), cutoff, cutoff+elapsed,
                    [{"source_last_seconds": sampled/row["fps"], "source_last_frame": sampled,
                      "image_path": str(image_path.resolve()), "image_sha256": digest(image_path), "source_video_sha256": row["video_sha256"]}],
                    {"valid": error is None, **answer}, str(adapter.resolve()))
                result = {"mode": mode, "source_recording_id": row["source_recording_id"], "window_start_seconds": start,
                          "cutoff_seconds": cutoff, "source_first_frame": first, "source_exclusive_end_frame": cutoff_frame,
                          "last_sampled_source_frame": sampled, "raw_output": raw, "answer": answer, "valid": error is None,
                          "error": error, "target_phase": job["target_phase"], "target_index": job["target_index"],
                          "label_correct": error is None and answer["phase"] == job["target_phase"],
                          "index_correct": error is None and answer["phase_index"] == job["target_index"],
                          "history": [x[0] for x in history], "history_visual_evidence": [x[1] for x in history],
                          "explicit_current_frame": newest,
                          "prepare_generate_s": elapsed, "input_tokens": tokens}
                if args.prepare_training:
                    # No reference label is inserted into candidate history. Ordinal harvest identity
                    # cannot reliably be inferred from an isolated window: explicitly leave it null.
                    target = {"phase": job["target_phase"], "phase_index": None if job["target_phase"] == "harvest" else job["target_index"],
                              "uncertain": False, "reason": ""}
                    sample = {"sample_id": row["source_recording_id"]+":"+str(cutoff_frame),
                              "source_recording_id": row["source_recording_id"], "split": row["split"],
                              "video": str(clip.resolve()), "video_sha256": digest(clip), "frames": count, "fps": fps,
                              "prompt": question, "history_items": [x[0] for x in history],
                              "history_images": [x[1] for x in history], "bounded_observation": True,
                              "source_first_frame": first, "source_exclusive_end_frame": cutoff_frame,
                              "source_video_sha256": row["video_sha256"], "reference_phase_index": job["target_index"],
                              "answer": json.dumps(target, ensure_ascii=False)}
                    with (args.output/(row["split"]+".jsonl")).open("a") as stream:
                        stream.write(json.dumps(sample, ensure_ascii=False)+"\n")
                predictions.append(result)
                with (args.output/"predictions.jsonl").open("a") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                status = {"status": "running", "mode": mode, "completed": len(predictions), "total": len(jobs)*len(modes)}
                write_json(args.output/"status.json", status)
                print(json.dumps({**status, "phase": answer.get("phase"), "valid": error is None}), flush=True)
                del inputs, output, video
            write_json(args.output/"summary.json", {m: summarize([r for r in predictions if r["mode"] == m])
                                                    for m in Counter(r["mode"] for r in predictions)})
        if args.prepare_training:
            write_json(args.output/"summary.json", {"counts": {split: sum(r["split"] == split for r in rows)*5 for split in ("train", "val")},
                "manifests_sha256": {split: digest(args.output/(split+".jsonl")) for split in ("train", "val")},
                "history": "own frozen-model candidates, never ground truth; <=2 images and 384 text tokens",
                "harvest_ordinal_supervision": "null; unsupported first/second distinction is not fabricated",
                "reason_supervision": "empty: coarse labels do not supply visual evidence descriptions",
                "source_manifest_sha256": {split: digest(data/(split+".jsonl")) for split in ("train", "val")},
                "test_used": False})
        write_json(args.output/"status.json", {"status": "complete", "completed": len(predictions)})
    except Exception as exc:
        write_json(args.output/"status.json", {"status": "failed", "completed": len(predictions), "error": str(exc)})
        raise
    finally:
        memory.close()


if __name__ == "__main__":
    main()
