"""Compute 12k mixed-VLM minus Base success-rate sensitivity to benchmark composition.

Run beside libero-tasks.csv with 'python leave_one_out.py'. Outputs two scopes times
40 individual task exclusions and four suite exclusions. Each condition has 50
episodes per task. The unit of exclusion is a task or suite, never a training seed.
"""
import argparse
import csv
from pathlib import Path


def select_conditions(path, scope):
    reference = {}
    treatment = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = row["suite"], int(row["task_id"]), row["task_name"]
            if row["evidence_id"] == f"L-{scope}-base-12000":
                selected = reference
            elif row["evidence_id"] == f"L-{scope}-finetuned-12000":
                selected = treatment
            else:
                continue
            if key in selected:
                raise ValueError(f"Duplicate task: {key}")
            successes, episodes = int(row["successes"]), int(row["episodes"])
            if episodes != 50 or not 0 <= successes <= episodes:
                raise ValueError(f"Invalid rollout counts in {row}")
            selected[key] = successes
    if reference.keys() != treatment.keys() or len(reference) != 40:
        raise ValueError(f"Unmatched or incomplete condition: {scope}")
    return reference, treatment


def exclusions(tasks):
    for key in tasks:
        yield "task", key[0], key[1], key[2], {key}
    for suite in sorted({key[0] for key in tasks}):
        omitted = {key for key in tasks if key[0] == suite}
        if len(omitted) != 10:
            raise ValueError(f"Expected ten tasks in {suite}")
        yield "suite", suite, "", "", omitted


def measure(path):
    for scope in ("frozen", "full"):
        base, mixed = select_conditions(path, scope)
        for unit, suite, task_id, name, omitted in exclusions(base):
            kept = set(base) - omitted
            total = 50 * len(kept)
            base_success = sum(base[key] for key in kept)
            mixed_success = sum(mixed[key] for key in kept)
            yield {
                "scope": scope,
                "excluded_unit": unit,
                "excluded_suite": suite,
                "excluded_task_id": task_id,
                "excluded_task_name": name,
                "retained_tasks": len(kept),
                "episodes_per_arm": total,
                "base_successes": base_success,
                "mixed_successes": mixed_success,
                "mixed_minus_base_pp": f"{100 * (mixed_success - base_success) / total:.8f}",
            }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=here / "libero-tasks.csv")
    parser.add_argument("--out", type=Path, default=here / "sensitivity.csv")
    args = parser.parse_args()
    rows = list(measure(args.tasks))
    with args.out.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
