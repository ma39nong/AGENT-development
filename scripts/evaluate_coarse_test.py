"""Evaluate the frozen coarse LoRA and base model on both held-out splits."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

if __package__:
    from .audit_subtask_data import digest
    from .build_coarse_sft import target_json
    from .train_coarse_lora import evaluate_answer, write_json
else:
    from audit_subtask_data import digest
    from build_coarse_sft import target_json
    from train_coarse_lora import evaluate_answer, write_json


def summarize(rows):
    valid = [r for r in rows if r["metrics"]["valid"]]
    errors = [e for r in valid for e in r["metrics"]["boundary_errors_s"]]
    return {
        "samples": len(rows), "valid_outputs": len(valid),
        "boundary_mae_s_valid_only": statistics.mean(errors) if errors else None,
        "max_boundary_error_s_valid_only": max(errors) if errors else None,
        "boundary_within_0_5s_valid_only": sum(e <= .500001 for e in errors)/len(errors) if errors else None,
        "boundary_within_1s_valid_only": sum(e <= 1.000001 for e in errors)/len(errors) if errors else None,
        "mean_segment_iou_valid_only": statistics.mean(r["metrics"]["mean_segment_iou"] for r in valid) if valid else None,
        "generation_seconds_median": statistics.median(r["generation_seconds"] for r in rows) if rows else None,
    }


def prepare(inventory, training_data, template):
    lock = json.loads(Path(inventory["source_lock"]).read_text())
    if digest(Path(inventory["source_lock"])) != inventory["source_lock_sha256"]:
        raise ValueError("source lock changed")
    locked = {r["source_recording_id"]: r for r in lock["recordings"]}
    training_rows = [json.loads(line) for split in ("train", "val")
                     for line in (training_data/(split+".jsonl")).read_text().splitlines()]
    used = {r["source_recording_id"] for r in training_rows}
    used_hashes = {r["video_sha256"] for r in training_rows}
    rows = []
    import cv2
    for record in inventory["recordings"]:
        if record["split"] not in ("historical_test", "blind_test"):
            continue
        identity = record["source_recording_id"]
        if identity in used or locked[identity]["split"] != record["split"]:
            raise ValueError("held-out identity overlaps training/validation or split changed")
        annotation = Path(record["coarse_annotation"])
        if digest(annotation) != locked[identity]["annotation_sha256"]:
            raise ValueError("test annotation differs from frozen lock")
        ann = json.loads(annotation.read_text())
        source = ann["source"]
        if source["source_recording_id"] != identity or source["length"] != record["frames"] or source["fps"] != record["fps"]:
            raise ValueError("source identity or clock mismatch")
        relative = f'videos/chunk-000/observation.images.cam0/episode_{record["episode_index"]:06d}.mp4'
        video = Path(record["videos"]["cam0"])
        if not video.is_file():
            video = Path(source["dataset"])/relative
        checksum = digest(video)
        if checksum != source["sha256"][relative]:
            raise ValueError("test video differs from annotation-bound bytes")
        if checksum in used_hashes:
            raise ValueError("test video bytes overlap training/validation")
        cap = cv2.VideoCapture(str(video))
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if frames != record["frames"] or abs(fps-record["fps"]) > .01:
            raise ValueError("test video frame clock mismatch")
        rows.append({"source_recording_id": identity, "split": record["split"],
                     "video": str(video), "video_sha256": checksum,
                     "annotation": str(annotation), "annotation_sha256": digest(annotation),
                     "prompt": template.replace("{duration_seconds}", f'{frames/fps:.2f}'),
                     "answer": target_json(ann["segments"], frames, fps)})
    if len(rows) != 32 or len({r["source_recording_id"] for r in rows}) != 32:
        raise ValueError("expected exactly 32 unique held-out recordings")
    if sum(r["split"] == "blind_test" for r in rows) != 20:
        raise ValueError("expected frozen split counts: blind 20, historical 12")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evaluation output already exists")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((args.run/"run.json").read_text())["config"]
    data = root/"data/derived/coarse_sft_20261123_v1"
    template = (root/"prompts/coarse_sft_zh.txt").read_text()
    manifest_summary = json.loads((args.run/"run.json").read_text())["manifest_summary"]
    for split in ("train", "val"):
        if digest(data/(split+".jsonl")) != manifest_summary["manifests_sha256"][split]:
            raise ValueError("training manifests changed after training")
    if digest(root/"prompts/coarse_sft_zh.txt") != manifest_summary["prompt_sha256"]:
        raise ValueError("evaluation prompt changed after training")
    inventory = json.loads((root/"runs/data-intake-204-v2/inventory.json").read_text())
    rows = prepare(inventory, data, template)
    write_json(args.output/"test-manifest.json", rows)
    write_json(args.output/"protocol.json", {
        "adapter": str((args.run/"best").resolve()),
        "adapter_sha256": digest(args.run/"best/adapter_model.safetensors"),
        "source_lock_sha256": inventory["source_lock_sha256"],
        "prompt_sha256": digest(root/"prompts/coarse_sft_zh.txt"),
        "config": config, "checkpoint_selection": "best validation loss, fixed before test unsealing",
        "test_samples": len(rows), "no_training_or_checkpoint_selection_on_test": True,
        "input": "full original head video; prompt with duration only; no target answers in model input",
        "comparisons": ["best_lora", "base_model", "train_only_duration_prior"],
    })
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
    predictions = []
    try:
        for mode in ("best_lora", "base_model"):
            with model.disable_adapter() if mode == "base_model" else nullcontext():
                for row in rows:
                    video, metadata = load_video(row["video"], backend="opencv", fps=config["video_sampling_fps"])
                    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": row["prompt"]}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], videos=[video], video_metadata=[metadata],
                                       do_sample_frames=False, return_tensors="pt").to("cuda:0")
                    tokens = inputs["input_ids"].shape[-1]
                    if tokens > config["max_sequence_tokens"]:
                        raise ValueError("full video exceeds token budget; no truncation permitted")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=config["generation_max_new_tokens"],
                                                do_sample=False, use_cache=True)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    raw = processor.tokenizer.decode(output[0, tokens:], skip_special_tokens=True)
                    result = {"mode": mode, "source_recording_id": row["source_recording_id"], "split": row["split"],
                              "raw_output": raw, "target": row["answer"], "metrics": evaluate_answer(raw, row["answer"]),
                              "input_tokens": tokens, "generation_seconds": elapsed}
                    predictions.append(result)
                    with (args.output/"predictions.jsonl").open("a") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                    status = {"status": "running", "mode": mode, "completed": len(predictions), "total": len(rows)*2,
                              "last_id": row["source_recording_id"], "last_metrics": result["metrics"]}
                    write_json(args.output/"status.json", status)
                    print(json.dumps(status), flush=True)
                    del inputs, output, video
        fractions = json.loads((args.run/"duration_only_prior.json").read_text())["normalized_training_median_boundaries"]
        prior = []
        for row in rows:
            target = json.loads(row["answer"])
            duration = target["segments"][-1]["end_seconds"]
            bounds = [0]+[round(duration*f, 2) for f in fractions]+[duration]
            answer = {"segments": [{**s, "start_seconds": bounds[i], "end_seconds": bounds[i+1]}
                                   for i, s in enumerate(target["segments"])]}
            prior.append({"mode": "train_only_duration_prior", "split": row["split"],
                          "source_recording_id": row["source_recording_id"], "raw_output": json.dumps(answer),
                          "metrics": evaluate_answer(json.dumps(answer), row["answer"]), "generation_seconds": 0})
        write_json(args.output/"duration-prior-predictions.json", prior)
        summaries = {}
        for mode in ("best_lora", "base_model", "train_only_duration_prior"):
            selected = [p for p in predictions+prior if p["mode"] == mode]
            summaries[mode] = {split: summarize([p for p in selected if split == "all" or p["split"] == split])
                               for split in ("all", "historical_test", "blind_test")}
        write_json(args.output/"summary.json", summaries)
        write_json(args.output/"status.json", {"status": "complete", "completed": len(predictions), "total": len(rows)*2})
        print(json.dumps(summaries), flush=True)
    except Exception as exc:
        write_json(args.output/"status.json", {"status": "failed", "completed": len(predictions), "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
