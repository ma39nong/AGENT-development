"""Check count/order constraints against candidate events and execution acknowledgements.

Passing is only logical consistency, never visual truth or robot authorization.
"""


def check_plan(events, target, request, *, uncertain, required_first_batch=None,
               adjustment_started_s=None, adjustment_finished_s=None):
    if target <= 0 or (required_first_batch is not None and not 0 < required_first_batch < target):
        raise ValueError("invalid task quantities")
    issues = []
    if uncertain:
        issues.append("perception_or_history_uncertain")
    if len(events) > target:
        issues.append("target_already_exceeded")
    if required_first_batch is not None and adjustment_started_s is None:
        if len(events) > required_first_batch:
            issues.append("first_batch_already_exceeded_before_adjustment")
        if request == "adjust" and len(events) < required_first_batch:
            issues.append("required_first_batch_not_reached")
        if request == "harvest" and len(events) >= required_first_batch:
            issues.append("required_adjustment_before_next_harvest")
    if required_first_batch is not None and adjustment_started_s is not None:
        before = sum(t < adjustment_started_s for t in events)
        if before != required_first_batch:
            issues.append("first_batch_count_violates_instruction")
        if adjustment_finished_s is None:
            issues.append("adjustment_not_acknowledged_complete")
        elif any(adjustment_started_s <= t < adjustment_finished_s for t in events):
            issues.append("delivery_overlaps_adjustment_requires_review")
    if request == "return":
        if len(events) < target:
            issues.append("target_not_reached")
        if required_first_batch is not None and adjustment_started_s is None:
            issues.append("required_adjustment_missing")
    return {"logical_conditions_satisfied": not issues, "issues": issues,
            "visual_truth_verified": False, "control_eligible": False}
