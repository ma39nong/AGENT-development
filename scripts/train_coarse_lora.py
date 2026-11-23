"""Single-GPU full-video coarse-segmentation LoRA, assistant-only loss."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def assistant_start(ids, header):
    matches = [i for i in range(len(ids)-len(header)+1) if ids[i:i+len(header)] == header]
    if len(matches) != 1:
        raise ValueError("expected exactly one assistant header")
    start = matches[0]+len(header)
    if start <= 0 or start >= len(ids):
        raise ValueError("no supervised assistant answer")
    return start


def suffix_loss(logits, target):
    import torch.nn.functional as F
    if logits.shape[1] != target.shape[1]+1:
        raise ValueError("causal suffix alignment mismatch")
    return F.cross_entropy(logits[:, :-1].contiguous().float().view(-1, logits.shape[-1]), target.reshape(-1))


def evaluate_answer(raw, truth):
    try:
        answer = json.loads(raw)
        if not isinstance(answer, dict):
            raise ValueError("output must be an object")
        if "phase" in json.loads(truth):
            if answer.get("phase") not in {"pickup", "harvest", "adjust", "return", "unknown"} or type(answer.get("uncertain")) is not bool:
                raise ValueError("invalid observation output")
            idx = answer.get("phase_index")
            if idx is not None and (type(idx) is not int or not 1 <= idx <= 5):
                raise ValueError("invalid phase index")
            return {"valid": True, "boundary_mae_s": 0., "classification_only": True}
        if set(answer) != {"segments"} or not isinstance(answer["segments"], list):
            raise ValueError("unexpected output schema")
        expected = json.loads(truth)["segments"]
        predicted = answer["segments"]
        if len(predicted) != len(expected):
            return {"valid": False, "error": "wrong stage count", "predicted_stage_count": len(predicted)}
        previous = 0.
        for index, (p, t) in enumerate(zip(predicted, expected), 1):
            if type(p["phase_index"]) is not int or p["phase_index"] != index or p["label"] != t["label"]:
                raise ValueError("wrong phase order/label")
            a, b = p["start_seconds"], p["end_seconds"]
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b)) or abs(a-previous) > .011 or a >= b:
                raise ValueError("noncontiguous or invalid times")
            previous = b
        if abs(previous-expected[-1]["end_seconds"]) > .011:
            raise ValueError("does not cover supplied duration")
        errors = [abs(p["end_seconds"]-t["end_seconds"]) for p, t in zip(predicted[:-1], expected[:-1])]
        ious = []
        for p, t in zip(predicted, expected):
            intersection = max(0., min(p["end_seconds"], t["end_seconds"])-max(p["start_seconds"], t["start_seconds"]))
            union = max(p["end_seconds"], t["end_seconds"])-min(p["start_seconds"], t["start_seconds"])
            ious.append(intersection/union)
        return {"valid": True, "boundary_errors_s": errors,
                "boundary_mae_s": sum(errors)/len(errors) if errors else 0., "mean_segment_iou": sum(ious)/len(ious)}
    except (ValueError, KeyError, TypeError) as exc:
        return {"valid": False, "error": str(exc)}


def current_phase_metrics(raw, truth):
    """Current label/index accuracy is separate from complete-history correctness."""
    try:
        if "phase" in json.loads(truth):
            actual, expected = json.loads(raw), json.loads(truth)
            return {"current_label_correct": actual.get("phase") == expected["phase"],
                    "current_phase_index_correct": actual.get("phase_index") == expected["phase_index"]}
        actual = json.loads(raw)["segments"][-1]
        expected = json.loads(truth)["segments"][-1]
        return {"current_label_correct": actual["label"] == expected["label"],
                "current_phase_index_correct": actual["phase_index"] == expected["phase_index"]}
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return {"current_label_correct": False, "current_phase_index_correct": False}


def observation_report(predictions):
    """Errors remain in recall denominators; null index agreement is not ordinal accuracy."""
    labels = ("pickup", "harvest", "adjust", "return")
    confusion = {label: {x: 0 for x in (*labels, "unknown", "invalid")} for label in labels}
    known_total = known_correct = reference_total = reference_correct = 0
    for prediction in predictions:
        truth = json.loads(prediction["target"])
        try:
            actual = json.loads(prediction["raw_output"])
            phase = actual["phase"] if prediction["metrics"]["valid"] else "invalid"
        except (ValueError, TypeError, KeyError):
            actual, phase = {}, "invalid"
        confusion[truth["phase"]][phase] += 1
        if truth["phase_index"] is not None:
            known_total += 1
            known_correct += prediction["metrics"]["valid"] and actual.get("phase_index") == truth["phase_index"]
        if prediction.get("reference_phase_index") is not None:
            reference_total += 1
            reference_correct += prediction["metrics"]["valid"] and actual.get("phase_index") == prediction["reference_phase_index"]
    recalls = {k: row[k]/sum(row.values()) if sum(row.values()) else None for k, row in confusion.items()}
    present = [x for x in recalls.values() if x is not None]
    return {"confusion": confusion, "per_class_recall": recalls,
            "macro_recall_present_classes": sum(present)/len(present) if present else None,
            "false_return_count": sum(row["return"] for label, row in confusion.items() if label != "return"),
            "non_null_target_index_samples": known_total,
            "non_null_target_index_accuracy": known_correct/known_total if known_total else None,
            "reference_full_five_stage_index_accuracy": reference_correct/reference_total if reference_total else None,
            "note": "Nullable target agreement is not full five-stage index accuracy; static frozen-candidate validation is not online closed-loop evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; do not overwrite training/checkpoints")
    args.output.mkdir(parents=True)
    import torch
    import peft
    import transformers
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup
    from transformers.video_utils import load_video

    config = json.loads(args.config.read_text())
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    train = [json.loads(s) for s in (args.data/"train.jsonl").read_text().splitlines()]
    val = [json.loads(s) for s in (args.data/"val.jsonl").read_text().splitlines()]
    if any(r["split"] != "train" for r in train) or any(r["split"] != "val" for r in val):
        raise ValueError("manifest split mismatch")
    if {r["source_recording_id"] for r in train} & {r["source_recording_id"] for r in val}:
        raise ValueError("source recording leakage")
    if config["micro_batch_size"] != 1:
        raise ValueError("runner currently supports one complete video per microbatch")
    manifest_summary = json.loads((args.data/"summary.json").read_text())
    for name in ("train", "val"):
        if hashlib.sha256((args.data/(name+".jsonl")).read_bytes()).hexdigest() != manifest_summary["manifests_sha256"][name]:
            raise ValueError("manifest changed since preparation")
    metrics_path = args.output/"metrics.jsonl"
    start_wall = time.time()

    def log(event, **fields):
        record = {"event": event, "unix_time": time.time(), "elapsed_s": time.time()-start_wall, **fields}
        with metrics_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    write_json(args.output/"status.json", {"status": "loading", "global_step": 0})
    log("loading", config=config, smoke=args.smoke, train_count=len(train), val_count=len(val))
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(config["model_dir"], dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation=config["attention"], local_files_only=True, trust_remote_code=False)
    targets = [name for name, module in model.named_modules() if ".language_model.layers." in name
               and name.rsplit(".", 1)[-1] in config["target_language_linear_modules"] and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("no language-side LoRA targets found")
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                             lora_dropout=config["lora_dropout"], target_modules=targets, bias="none")
    if config.get("initial_adapter"):
        model = PeftModel.from_pretrained(model, config["initial_adapter"], is_trainable=True)
        loaded = model.peft_config["default"]
        if loaded.r != config["lora_rank"] or loaded.lora_alpha != config["lora_alpha"]:
            raise ValueError("initial adapter configuration differs")
    else:
        model = get_peft_model(model, lora_config)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any("lora_" not in name or ".visual." in name for name, _ in parameters):
        raise ValueError("unexpected trainable base or visual weights")
    smoke_parameter = next(p for name, p in parameters if "lora_B" in name) if args.smoke else None
    smoke_before = smoke_parameter.detach().cpu().clone() if args.smoke else None
    run = {"config": config, "smoke": args.smoke, "train_count": len(train), "val_count": len(val),
           "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
           "gpu": torch.cuda.get_device_name(0), "trainable_parameters": sum(p.numel() for _, p in parameters),
           "total_parameters": sum(p.numel() for p in model.parameters()), "lora_target_count": len(targets),
           "target_modules": targets, "manifest_summary": manifest_summary,
           "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "video_processor_size": dict(processor.video_processor.size), "sealed_test_used": False,
           "robot_control": False, "counting_supervision": False,
           "input_type": "trailing_8s_head_video_and_bounded_candidate_memory" if config.get("bounded_observation_task") else ("physical_causal_head_video_prefix" if config.get("prefix_task") else "full_original_head_video_native_video_tokens")}
    if config.get("initial_adapter"):
        run["initial_adapter_sha256"] = hashlib.sha256((Path(config["initial_adapter"])/"adapter_model.safetensors").read_bytes()).hexdigest()
    write_json(args.output/"run.json", run)
    log("model_ready", trainable_parameters=run["trainable_parameters"], lora_target_count=len(targets))
    header = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def inputs_for(row, teacher=True):
        # Preserve native video processing; bounded manifests add separately timestamped history images.
        video, metadata = load_video(row["video"], backend="opencv", fps=config["video_sampling_fps"])
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": row["prompt"]}]}]
        images = []
        if row.get("bounded_observation"):
            from PIL import Image
            content = [{"type": "text", "text": row["prompt"]}, {"type": "video"}]
            if len(row["history_items"]) != len(row["history_images"]) or len(row["history_images"]) > 2:
                raise ValueError("invalid bounded history count")
            for item, evidence in zip(row["history_items"], row["history_images"]):
                if item["observed_until"] > row["source_first_frame"]/row["fps"] or evidence["source_last_seconds"] >= item["observed_until"]:
                    raise ValueError("future history evidence")
                path = Path(evidence["image_path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["image_sha256"]:
                    raise ValueError("history evidence changed")
                content += [{"type": "text", "text": f'历史原始帧，源时间{item["evidence_seconds"]:.3f}s；不是当前画面。'}, {"type": "image"}]
                im = Image.open(path).convert("RGB")
                im.thumbnail((448, 280))
                images.append(im)
            messages = [{"role": "user", "content": content}]
        if teacher:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=not teacher)
        inputs = processor(text=[text], videos=[video], video_metadata=[metadata], do_sample_frames=False, return_tensors="pt", **({"images": images} if images else {}))
        if inputs["input_ids"].shape[-1] > config["max_sequence_tokens"]:
            raise ValueError("complete video exceeds token budget; no silent truncation")
        k = assistant_start(inputs["input_ids"][0].tolist(), header) if teacher else None
        return inputs.to("cuda:0"), k

    def loss_for(row):
        inputs, k = inputs_for(row)
        target = inputs["input_ids"][:, k:]
        # Retain exactly the preceding predictive position and all supervised assistant positions.
        output = model(**inputs, logits_to_keep=target.shape[1]+1, use_cache=False)
        loss = suffix_loss(output.logits, target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        return loss, inputs["input_ids"].shape[-1], target.shape[1]

    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    accumulation = 1 if args.smoke else config["gradient_accumulation_steps"]
    epochs = 1 if args.smoke else config["epochs"]
    total_steps = 1 if args.smoke else math.ceil(len(train)/accumulation)*epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0 if args.smoke else math.ceil(total_steps*config["warmup_ratio"]), total_steps)
    step = 0

    def checkpoint(name, epoch, offset):
        directory = args.output/name
        model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
        processor.save_pretrained(directory)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": step,
                    "epoch": epoch, "next_sample_offset": offset, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}, directory/"trainer_state.pt")
        write_json(directory/"training_metadata.json", {"global_step": step, "epoch": epoch, "next_sample_offset": offset,
                   "config": config, "manifest_sha256": manifest_summary["manifests_sha256"], "base_model": config["model_dir"]})
        log("checkpoint_saved", path=str(directory.resolve()), global_step=step)

    def evaluate(tag, generation_rows):
        write_json(args.output/"status.json", {"status": "evaluating", "tag": tag, "global_step": step})
        model.eval()
        losses = []
        with torch.inference_mode():
            for row in val:
                loss, _, _ = loss_for(row)
                losses.append(float(loss))
            predictions = []
            for row in generation_rows:
                inputs, _ = inputs_for(row, teacher=False)
                output = model.generate(**inputs, max_new_tokens=config["generation_max_new_tokens"], do_sample=False, use_cache=True)
                raw = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                predictions.append({"source_recording_id": row["source_recording_id"], "raw_output": raw,
                                    "sample_id": row.get("sample_id", row["source_recording_id"]),
                                    "reference_phase_index": row.get("reference_phase_index"),
                                    "target": row["answer"], "metrics": evaluate_answer(raw, row["answer"]),
                                    "online_metrics": current_phase_metrics(raw, row["answer"])})
        valid = [p["metrics"] for p in predictions if p["metrics"]["valid"]]
        result = {"tag": tag, "global_step": step, "validation_loss": sum(losses)/len(losses),
                  "loss_samples": len(losses), "generation_samples": len(predictions), "valid_outputs": len(valid),
                  "boundary_mae_s_valid_only": sum(v["boundary_mae_s"] for v in valid)/len(valid) if valid and not config.get("bounded_observation_task") else None,
                  "current_label_accuracy": sum(p["online_metrics"]["current_label_correct"] for p in predictions)/len(predictions) if predictions else None,
                  "current_phase_index_accuracy": sum(p["online_metrics"]["current_phase_index_correct"] for p in predictions)/len(predictions) if predictions else None,
                  "predictions": predictions}
        if config.get("bounded_observation_task"):
            result["observation_report"] = observation_report(predictions)
            result["nullable_target_index_agreement"] = result.pop("current_phase_index_accuracy")
        write_json(args.output/("eval-"+tag+".json"), result)
        log("evaluation", **{k: v for k, v in result.items() if k != "predictions"})
        return result

    # Fixed validation subset across baseline and checkpoints; never inspect sealed tests.
    generation_rows = sorted(val, key=lambda r: r["source_recording_id"])[:config["generation_eval_samples"]]
    if config.get("prefix_task") or config.get("bounded_observation_task"):
        # Select complete prefix sets from fixed validation recordings, rather than only early phases.
        identities = sorted({r["source_recording_id"] for r in val})[:config.get("generation_eval_recordings", 3)]
        generation_rows = [r for r in val if r["source_recording_id"] in identities]
    sequence_lengths = set() if config.get("bounded_observation_task") else {len(json.loads(row["answer"])["segments"]) for row in train}
    if len(sequence_lengths) == 1:
        count = next(iter(sequence_lengths))
        fractions = []
        for index in range(count-1):
            fractions.append(statistics.median(json.loads(row["answer"])["segments"][index]["end_seconds"]/
                                               json.loads(row["answer"])["segments"][-1]["end_seconds"] for row in train))
        write_json(args.output/"duration_only_prior.json", {"normalized_training_median_boundaries": fractions,
                   "note": "Nonvisual baseline, train-only statistics; compare boundary errors rather than fixed five-label accuracy."})
    best_loss = float("inf")
    if not args.smoke and config.get("evaluate_base_model", True):
        with model.disable_adapter():
            evaluate("baseline", generation_rows)
    if not args.smoke and config.get("initial_adapter"):
        evaluate("initial-adapter", generation_rows)
    try:
        for epoch in range(epochs):
            order = list(train)
            random.Random(config["seed"]+epoch).shuffle(order)
            if args.smoke:
                order = [max(train, key=lambda r: len(r.get("history_images", [])))] if config.get("bounded_observation_task") else [min(train, key=lambda r: r["frames"])]
            model.train()
            model.get_base_model().model.visual.eval()
            for offset in range(0, len(order), accumulation):
                begin = time.perf_counter()
                group = order[offset:offset+accumulation]
                optimizer.zero_grad(set_to_none=True)
                losses, sample_ids, lengths = [], [], []
                torch.cuda.reset_peak_memory_stats()
                for row in group:
                    loss, tokens, answer_tokens = loss_for(row)
                    (loss/len(group)).backward()
                    losses.append(float(loss.detach()))
                    sample_ids.append(row["source_recording_id"])
                    lengths.append({"tokens": tokens, "assistant_tokens": answer_tokens})
                    del loss
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], config["max_grad_norm"])
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite gradient norm")
                optimizer.step()
                scheduler.step()
                step += 1
                torch.cuda.synchronize()
                log("train_step", global_step=step, total_optimizer_steps=total_steps, epoch=epoch+1,
                    loss=sum(losses)/len(losses), grad_norm=float(norm), lr=scheduler.get_last_lr()[0],
                    sample_ids=sample_ids, lengths=lengths, step_s=time.perf_counter()-begin,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                write_json(args.output/"status.json", {"status": "training", "global_step": step,
                           "total_optimizer_steps": total_steps, "epoch": epoch+1, "updated_unix": time.time()})
                if args.smoke or step % config["save_every_optimizer_steps"] == 0:
                    checkpoint(f"checkpoint-{step:04d}", epoch, offset+len(group))
            if args.smoke:
                nonzero = any(bool(torch.count_nonzero(p)) for name, p in parameters if "lora_B" in name)
                changed = not torch.equal(smoke_before, smoke_parameter.detach().cpu())
                if not nonzero or not changed:
                    raise ValueError("LoRA B weights did not update")
                write_json(args.output/"smoke_result.json", {"passed": True, "loss_finite": True,
                           "gradients_finite": True, "lora_weights_updated": True,
                           "checkpoint": str((args.output/"checkpoint-0001").resolve())})
                log("smoke_passed")
            else:
                checkpoint(f"epoch-{epoch+1}", epoch+1, 0)
                result = evaluate(f"epoch-{epoch+1}", generation_rows)
                if result["validation_loss"] < best_loss:
                    best_loss = result["validation_loss"]
                    checkpoint("best", epoch+1, 0)
        if not args.smoke:
            model.load_adapter(args.output/"best", adapter_name="best_eval")
            model.set_adapter("best_eval")
            evaluate("best-full-validation", val)
        write_json(args.output/"status.json", {"status": "complete", "global_step": step,
                   "total_optimizer_steps": total_steps, "updated_unix": time.time()})
        log("complete", global_step=step)
    except Exception as exc:
        write_json(args.output/"status.json", {"status": "failed", "global_step": step,
                   "error": type(exc).__name__+": "+str(exc), "traceback": traceback.format_exc(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
