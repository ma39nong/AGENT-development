"""Run the frozen coarse LoRA on new complete videos without reference labels."""
import argparse
import json
from pathlib import Path
import time

if __package__:
    from .audit_subtask_data import digest
    from .train_coarse_lora import evaluate_answer, write_json
else:
    from audit_subtask_data import digest
    from train_coarse_lora import evaluate_answer, write_json


def schema_check(raw, duration):
    # Check format/coverage only. New videos may have a different phase count or order.
    try:
        reference = json.loads(raw)
        segments = reference["segments"]
        if not segments or any(s["label"] not in {"pickup", "harvest", "adjust", "return"} for s in segments):
            raise ValueError("empty stages or unsupported label")
        reference["segments"][-1]["end_seconds"] = duration
        checked = evaluate_answer(raw, json.dumps(reference))
        return {k: v for k, v in checked.items() if k in ("valid", "error", "predicted_stage_count")}
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"valid": False, "error": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    root = Path(__file__).resolve().parents[1]
    run = json.loads((args.run/"run.json").read_text())
    config = run["config"]
    prompt_path = root/"prompts/coarse_sft_zh.txt"
    if digest(prompt_path) != run["manifest_summary"]["prompt_sha256"]:
        raise ValueError("prompt differs from training")
    rows = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True)
    write_json(args.output/"protocol.json", {"adapter": str((args.run/"best").resolve()),
        "adapter_sha256": digest(args.run/"best/adapter_model.safetensors"),
        "prompt_sha256": digest(prompt_path), "manifest_sha256": digest(args.manifest),
        "config": config, "reference_annotations": False, "training": False,
        "note": "Schema checks only; no accuracy metric without independent reference annotations."})
    import cv2
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import load_video
    torch.set_num_threads(8)
    torch.manual_seed(config["seed"])
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(config["model_dir"], dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation=config["attention"], local_files_only=True)
    model = PeftModel.from_pretrained(base, args.run/"best", is_trainable=False).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    results = []
    try:
        for row in rows:
            if digest(Path(row["video"])) != row["video_sha256"]:
                raise ValueError("video differs from manifest")
            cap = cv2.VideoCapture(row["video"])
            frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if frames != row["frames"] or abs(fps-row["fps"]) > .01:
                raise ValueError("video frame clock differs from metadata")
            duration = round(frames/fps, 2)
            prompt = prompt_path.read_text().replace("{duration_seconds}", f'{duration:.2f}')
            start = time.perf_counter()
            video, metadata = load_video(row["video"], backend="opencv", fps=config["video_sampling_fps"])
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=[video], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt").to("cuda:0")
            tokens = inputs["input_ids"].shape[-1]
            if tokens > config["max_sequence_tokens"]:
                raise ValueError("full video exceeds token budget; no truncation")
            torch.cuda.synchronize()
            gen_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=config["generation_max_new_tokens"], do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            gen_seconds = time.perf_counter()-gen_start
            raw = processor.tokenizer.decode(output[0, tokens:], skip_special_tokens=True)
            check = schema_check(raw, duration)
            result = {**row, "duration_seconds": duration, "raw_output": raw, "schema_check": check,
                      "input_tokens": tokens, "generation_seconds": gen_seconds,
                      "preparation_and_generation_seconds": time.perf_counter()-start,
                      "reference_annotation_available": False, "accuracy": None}
            if check["valid"]:
                result["segments"] = json.loads(raw)["segments"]
            results.append(result)
            write_json(args.output/(row["source_recording_id"]+".json"), result)
            write_json(args.output/"status.json", {"status": "running", "completed": len(results), "total": len(rows)})
            print(json.dumps(result, ensure_ascii=False), flush=True)
            del video, inputs, output
        write_json(args.output/"results.json", results)
        write_json(args.output/"status.json", {"status": "complete", "completed": len(results), "total": len(rows)})
    except Exception as exc:
        write_json(args.output/"status.json", {"status": "failed", "completed": len(results), "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
