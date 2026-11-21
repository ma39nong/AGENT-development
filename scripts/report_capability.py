"""Summarize offline screening automatically; reference never enters model inputs."""
import argparse
import json
from pathlib import Path
import statistics

if __package__:
    from .prompt_plan_guard import check_plan
else:
    from prompt_plan_guard import check_plan


def summarize(root, reference):
    report = {"reference": reference, "full_local": [], "streaming": {},
              "robot_control_enabled": False, "training_started": False,
              "limits": "One development recording, not independent accuracy or real closed-loop execution. All event truths and transition times remain provisional developer references."}
    for path in sorted((root / "local-full").glob("full-*.json")):
        raw = json.loads(path.read_text())
        row = {"job": path.stem, "total_s": raw["total_s"], "error": raw.get("error"),
               "raw_result": str(path.resolve()), "raw_output": raw.get("raw_output")}
        try:
            answer = json.loads(raw["raw_output"])
            boundaries = [h["seconds"] for h in answer["handoffs"]]
            row.update(sequence_supported=answer["sequence_supported"], boundaries_s=boundaries,
                       successful_counts=[answer["harvest_summary"][x]["successful_count"] for x in ("first_harvest", "second_harvest")],
                       boundary_delta_from_existing_annotation_s=[a-b for a, b in zip(boundaries, reference["coarse_boundaries_s"])])
        except (ValueError, KeyError, TypeError) as exc:
            row["parse_error"] = str(exc)
        report["full_local"].append(row)
    for backend, folder in (("local", "local-stream"), ("max_api", "max-stream-v2")):
        groups = {}
        for mode in ("recent_only", "recent_memory", "prefix"):
            rows = []
            for path in sorted((root / folder).glob(mode+"-*.json")):
                result = json.loads(path.read_text())
                answer = result.get("answer") or {}
                cutoff = result["cutoff"]
                row = {"cutoff_s": cutoff, "expected_count_at_cutoff": reference["counts_at_cutoffs"][str(cutoff)],
                       "reported_total": answer.get("confirmed_total"), "events": answer.get("delivery_seconds"),
                       "uncertain": answer.get("uncertain"), "structural_valid": result["structural_valid"],
                       "total_s": result["total_s"], "error": result.get("error"),
                       "tasks": {k: answer.get(k) for k in ("target4", "two_then_two")},
                       "raw_result": str(path.resolve())}
                row["count_matches_reference"] = row["reported_total"] == row["expected_count_at_cutoff"]
                events = (result["context"].get("historical_candidate_delivery_seconds") or []) + (answer.get("delivery_seconds") or [])
                unknown = not result["structural_valid"] or answer.get("uncertain") is not False or answer.get("confirmed_total") is None
                if mode in {"prefix", "recent_memory"}:
                    for task in ("target4", "two_then_two"):
                        if task in answer:
                            row[task+"_logical_guard"] = check_plan(events, 4, answer[task]["request"], uncertain=unknown,
                                required_first_batch=2 if task == "two_then_two" else None,
                                adjustment_started_s=reference["coarse_boundaries_s"][1] if cutoff >= 47 else None,
                                adjustment_finished_s=reference["coarse_boundaries_s"][2] if cutoff >= 62 else None)
                if str(cutoff) in reference["next_observed_detached_s"]:
                    next_detachment = reference["next_observed_detached_s"][str(cutoff)]
                    row["replay_earliest_response_source_time_s"] = cutoff + result["total_s"]
                    row["next_observed_detached_s"] = next_detachment
                    row["response_after_next_observed_detachment"] = cutoff+result["total_s"] > next_detachment
                rows.append(row)
            times = [r["total_s"] for r in rows]
            groups[mode] = {"queries": len(rows), "known_correct_totals": sum(r["count_matches_reference"] for r in rows),
                            "unknown_totals": sum(r["reported_total"] is None for r in rows),
                            "structural_rejections": sum(not r["structural_valid"] for r in rows),
                            "total_s_median": statistics.median(times) if times else None,
                            "rows": rows}
        report["streaming"][backend] = groups
    timeline = root / "timeline-context-check.json"
    if timeline.exists():
        result = json.loads(timeline.read_text())
        answer = result.get("answer") or {}
        report["timeline_context_check"] = {"answer": answer, "total_s": result["total_s"], "raw_result": str(timeline.resolve()),
            "two_then_two_logical_guard": check_plan(answer.get("delivery_seconds", []), 4,
                answer.get("two_then_two", {}).get("request"), uncertain=answer.get("uncertain") is not False,
                required_first_batch=2, adjustment_started_s=reference["coarse_boundaries_s"][1],
                adjustment_finished_s=reference["coarse_boundaries_s"][2])}
    def count_policy_checks(backend):
        rows = report["streaming"][backend]["prefix"]["rows"]
        checks = {"all_prefix_counts_match_reference": bool(rows) and all(r["count_matches_reference"] for r in rows),
                  "required_requests_present": True, "required_requests_before_next_observed_detachment": True,
                  "strict_order_not_falsely_declared_complete": True}
        for cutoff, task, request in ((37, "two_then_two", "adjust"), (77, "target4", "return")):
            row = next((r for r in rows if r["cutoff_s"] == cutoff), None)
            present = bool(row and row["structural_valid"] and row["tasks"][task]
                           and row["tasks"][task]["request"] == request and row["count_matches_reference"])
            checks["required_requests_present"] &= present
            checks["required_requests_before_next_observed_detachment"] &= bool(present and not row.get("response_after_next_observed_detachment", True))
        for row in rows:
            task = row["tasks"]["two_then_two"]
            if task and task["request"] == "return" and not row.get("two_then_two_logical_guard", {}).get("logical_conditions_satisfied", False):
                checks["strict_order_not_falsely_declared_complete"] = False
        return checks
    local_full_pass = any(r.get("successful_counts") == reference["full_successful_counts"]
                          and len(r.get("boundaries_s", [])) == 4
                          and all(abs(x) <= reference["boundary_screen_tolerance_s"] for x in r.get("boundary_delta_from_existing_annotation_s", []))
                          for r in report["full_local"])
    report["screening_checks"] = {b: count_policy_checks(b) for b in report["streaming"]}
    report["screening_checks"]["local"]["full_video_coarse_stage_and_count"] = local_full_pass
    report["screening_passed_by_backend"] = {b: all(checks.values()) for b, checks in report["screening_checks"].items()}
    report["overall_screening_passed"] = any(report["screening_passed_by_backend"].values())
    report["diagnostic_incomplete_run_records"] = len(list((root / "max-stream").glob("*-0*.json")))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, json.loads(args.reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"overall_screening_passed": report["overall_screening_passed"],
                      "model_backends": list(report["streaming"]), "full_local_queries": len(report["full_local"])}))
