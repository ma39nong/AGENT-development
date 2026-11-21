"""Native local Qwen3-VL video input, deterministic offline-only jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/user/models/Qwen3-VL-4B-Instruct"))
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import load_video

    if args.output.exists():
        raise ValueError("output already exists")
    jobs = json.loads(args.jobs.read_text())
    if not jobs or any(j.get("split") != "train" for j in jobs):
        raise ValueError("diagnostic jobs must use train only")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.synchronize()
    save(args.output / "run.json", {"model_dir": str(args.model_dir), "torch": torch.__version__,
                                    "transformers": transformers.__version__, "load_s": time.perf_counter()-start,
                                    "gpu": torch.cuda.get_device_name(0), "do_sample": False, "thinking": False,
                                    "input_type": "native_video", "decoder": "transformers.load_video/opencv",
                                    "internal_sampling_fps": 2, "processor_size": dict(processor.video_processor.size),
                                    "jobs_sha256": hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
                                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    "robot_control": False, "training": False})
    for job in jobs:
        begin = time.perf_counter()
        path = Path(job["video"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != job["sha256"]:
            raise ValueError("video hash mismatch")
        record = {"job": job, "control_eligible": False, "training_ready": False}
        try:
            video, metadata = load_video(str(path), backend="opencv", fps=2)
            prompt = Path(job["prompt_file"]).read_text()
            text = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}]}],
                tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=[video], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt").to("cuda:0")
            torch.cuda.synchronize()
            record.update(preparation_s=time.perf_counter()-begin, input_tokens=inputs["input_ids"].shape[-1],
                          video_grid_thw=inputs["video_grid_thw"].tolist(),
                          source_frame_indices=[int(x) for x in metadata.frames_indices],
                          source_fps=metadata.fps)
            print(json.dumps({"job": job["id"], "input_tokens": record["input_tokens"],
                              "video_grid_thw": record["video_grid_thw"]}), flush=True)
            torch.cuda.reset_peak_memory_stats()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=job.get("max_new_tokens", 1800),
                                        do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            ids = output[0, inputs["input_ids"].shape[-1]:]
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            record.update(raw_output=raw, output_tokens=ids.numel(),
                          generation_s=time.perf_counter()-generation_start,
                          complete_eos=bool(ids.numel() and int(ids[-1]) in (eos or [])),
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            del inputs, output, ids, video
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
        record["total_s"] = time.perf_counter()-begin
        save(args.output / (job["id"] + ".json"), record)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
