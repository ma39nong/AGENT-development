"""Local-only debug replay. Produces reports/advice, never robot commands."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time


SKILLS = {"continue_harvest", "finish_delivery", "adjust", "return_truss", "observe"}
VIEWS = {"head", "left_wrist", "right_wrist", "head_crop"}


def load_frames(manifest_path, include_crop):
    from PIL import Image

    manifest = json.loads(manifest_path.read_text())
    cutoff = manifest["cutoff_s"]
    if type(cutoff) not in (int, float) or not math.isfinite(cutoff):
        raise ValueError("Invalid observation cutoff")
    frames, images, seen = [], [], set()
    all_rows = manifest["frames"]
    parents = {r["path"]: r for r in all_rows if r["view"] == "head"}
    for row in all_rows:
        if row["view"] not in VIEWS:
            raise ValueError("Unknown view")
        if row["view"] == "head_crop" and not include_crop:
            continue
        stamp = row["source_timestamp_s"]
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= stamp <= cutoff:
            raise ValueError("Invalid/future source timestamp")
        if type(row["i"]) is not int or row["i"] in seen:
            raise ValueError("Invalid or duplicate image ID")
        seen.add(row["i"])
        if row["view"] == "head_crop":
            crop = row["crop"]
            parent = parents[crop["parent_path"]]
            if (crop["uses_future_coordinates"] is not False
                    or parent["source_timestamp_s"] != stamp
                    or parent["source_frame"] != row["source_frame"]
                    or parent["source_camera"] != row["source_camera"]):
                raise ValueError("Noncausal/unrelated crop provenance")
            box = crop["xyxy_pixels"]
            w, h = crop["source_size"]
            if len(box) != 4 or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h:
                raise ValueError("Invalid crop geometry")
        path = (manifest_path.parent / row["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Image escapes replay directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image digest mismatch")
        with Image.open(path) as source:
            if source.size != (row["width"], row["height"]):
                raise ValueError("Image size mismatch")
            images.append(source.convert("RGB"))
        frames.append(row)
    if not frames or {r["view"] for r in frames} != (VIEWS if include_crop else VIEWS - {"head_crop"}):
        raise ValueError("Missing required views")
    return manifest, frames, images


def validate_answer(text, frames):
    try:
        answer = json.loads(text)
        if not isinstance(answer, dict) or set(answer) != {"d", "h", "r", "e", "next_skill"}:
            raise ValueError("Unexpected output fields")
        d = answer["d"]
        if type(d) is not int or d not in (-1, 0, 1) or answer["next_skill"] not in SKILLS:
            raise ValueError("Invalid decision or skill")
        if d == 1:
            by_id = {r["i"]: r["source_timestamp_s"] for r in frames}
            ids = [answer[k] for k in ("h", "r", "e")]
            if any(type(i) is not int or i not in by_id for i in ids):
                raise ValueError("Evidence not in supplied frames")
            h, r, e = [by_id[i] for i in ids]
            if not h < r < e:
                raise ValueError("Evidence timestamps not strictly increasing")
        elif any(answer[k] is not None for k in ("h", "r", "e")):
            raise ValueError("Evidence must be null for absent/unknown event")
        return {"valid": True, "answer": answer,
                "visual_evidence_audited": False, "executable": False}
    except (ValueError, KeyError, TypeError) as error:
        return {"valid": False, "error": str(error), "executable": False}


def run(args):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.target <= 0 or not 0 <= args.confirmed <= args.target:
        raise ValueError("Invalid task scenario counts")
    if args.repeat < 1 or args.max_new_tokens < 1 or not 4096 <= args.max_pixels <= 262144:
        raise ValueError("Invalid generation bounds")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this GPU baseline")
    # No remote model resolution or remote code execution; Instruct has no thinking mode.
    model_dir = args.model_dir.expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl" or "Instruct" not in model_dir.name:
        raise ValueError("Expected local Qwen3-VL Instruct weights")
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = (Path(__file__).resolve().parents[1] / "prompts/event_baseline_zh.txt").read_text()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
    ).eval()
    # Deterministic baseline; remove sampling-only defaults shipped with the model.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    metadata = {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                "model_dir": str(model_dir), "load_s": load_s, "dtype": "bfloat16",
                "attention": "sdpa", "thinking": False, "do_sample": False,
                "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                "include_crop": args.include_crop, "repeat": args.repeat,
                "task_scenario": {"target": args.target, "confirmed_before_window": args.confirmed,
                                  "synthetic_context_not_actual_execution": True},
                "limits": "Debug clips, no independent accuracy estimate; repeated greedy outputs are not independent trials.",
                "robot_control": False, "training": False}
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    reports = []
    # First pass includes cold generation; mark it rather than hiding its cost.
    for repeat in range(args.repeat):
        for manifest_path in args.manifest:
            begin = time.perf_counter()
            manifest_path = manifest_path.expanduser().resolve()
            manifest, frames, images = load_frames(manifest_path, args.include_crop)
            content = [{"type": "text", "text": json.dumps({
                "target": args.target, "confirmed_before_window": args.confirmed,
                "active_skill": "harvest", "observation_cutoff_s": manifest["cutoff_s"],
            }, ensure_ascii=False)}]
            for row in frames:
                content += [{"type": "text", "text": json.dumps({
                    "i": row["i"], "view": row["view"], "t": row["source_timestamp_s"],
                    "same_source": [row["source_camera"], row["source_frame"]]}, ensure_ascii=False)},
                            {"type": "image"}]
            text = processor.apply_chat_template(
                [{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=[text], images=images, return_tensors="pt",
                               images_kwargs={"min_pixels": 4096, "max_pixels": args.max_pixels})
            inputs = inputs.to("cuda:0")
            torch.cuda.synchronize()
            preparation_s = time.perf_counter() - begin
            torch.cuda.reset_peak_memory_stats()
            generate_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            generate_s = time.perf_counter() - generate_start
            ids = output[0, inputs["input_ids"].shape[-1]:]
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            complete = bool(ids.numel() and int(ids[-1]) in (eos or []))
            validation = validate_answer(raw, frames)
            report = {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "repeat": repeat, "first_generation": not reports, "images": len(frames),
                      "roi_status": manifest["roi_status"], "raw_output": raw,
                      "validation": validation, "complete_eos": complete,
                      "input_tokens": inputs["input_ids"].shape[-1], "output_tokens": ids.numel(),
                      "image_grid_thw": inputs["image_grid_thw"].tolist(),
                      "timing": {"preparation_s": preparation_s, "generation_s": generate_s,
                                 "request_to_decoded_s": time.perf_counter() - begin},
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                      "output_complete_and_structural": complete and validation["valid"],
                      "control_eligible": False,
                      "proposed_skill_correctness_assessed": False}
            reports.append(report)
            (args.output / "results.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k: report[k] for k in ("manifest", "repeat", "images", "raw_output", "complete_eos", "timing", "output_complete_and_structural")}, ensure_ascii=False), flush=True)
            del inputs, output, ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path, nargs="+")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--include-crop", action="store_true")
    p.add_argument("--target", type=int, default=5)
    p.add_argument("--confirmed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--max-pixels", type=int, default=65536)
    p.add_argument("--max-new-tokens", type=int, default=160)
    run(p.parse_args())


if __name__ == "__main__":
    main()
