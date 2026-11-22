"""Aggregate one frozen v1 generation run into an EncoderTrial Gate measurement."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1_freeze
import v1_score

ARMS = ("ref", "en", "ko")
STUDENTS = ("en", "ko")
SEEDS = (17, 43, 97)
STEPS = 100
BUNDLE_NAMES = ("language_gap", "intent_en", "quality_en", "cost_en", "intent_ko", "quality_ko", "cost_ko")


def rows_of(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append(row)
    return rows


def sha256(path):
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def clip_score(arm_dir, row, seed, yaw_sign):
    directory = arm_dir / f"{row['id']}-{seed}"
    meta_path = directory / "motion.json"
    if not meta_path.is_file():
        return dict(hit=False, usable=False, reason="missing_artifacts", path=None)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = tuple(joint["name"] for joint in meta["joints"])
    fps = float(meta["fps"])
    expected = int(round(float(row["seconds"]) * fps))
    if int(meta.get("frameCount", -1)) != expected:
        return dict(hit=False, usable=False, reason="frame_count_mismatch", path=str(directory))
    with np.load(directory / "motion.npz", allow_pickle=False) as motion:
        result = v1_score.score_clip(motion, names, fps, expected, row["intent"], yaw_sign,
                                     footing=bool(row["footing"]))
    result["path"] = str(directory)
    result["reason"] = result["intent"]["reason"]
    if result["quality"]["reasons"]:
        result["quality_reasons"] = result["quality"]["reasons"]
    return result


def read_summary(arm_dir):
    summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("pilot"):
        raise ValueError(f"{arm_dir.name} is marked as a pilot; not v1 evidence")
    return summary


def candidate_of(run_root, arm):
    path = run_root / "student.json"
    if not path.is_file():
        return None
    student = json.loads(path.read_text(encoding="utf-8"))
    return {"checkpoint_sha256": student["checkpoint_sha256"],
            "checkpoint_run_json_sha256": student.get("checkpoint_run_json_sha256"),
            "entry_sha256": student["entry_sha256"],
            "backbone": student["backbone"], "environment": student["environment"]}


def measure(lock_path, run_root, stage, yaw_sign, ref_root=None, export_evidence=None, videos=None, out=None):
    freeze = v1_freeze.verify(lock_path, stage)
    split = "dev" if stage == "Dev" else "test"
    run_root = Path(run_root)
    rows = rows_of(freeze["manifests"][split]["path"])
    problems = []

    def check(label, condition):
        if not condition:
            problems.append(label)

    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    check("run manifest differs from frozen manifest",
          config["manifest_sha256"] == freeze["manifests"][split]["sha256"])
    check("run protocol differs from frozen protocol",
          config["steps"] == STEPS and config["seeds"] == list(SEEDS) and config["pilot"] is False
          and config["exports"] is True and config["concurrency"] == 1 and not config["constraints"])
    plan = [(row["id"], seed) for row in rows for seed in SEEDS]

    roots = {arm: run_root for arm in ARMS}
    if ref_root:
        ref_root = Path(ref_root)
        ref_config = json.loads((ref_root / "config.json").read_text(encoding="utf-8"))
        check("reused ref manifest differs", ref_config["manifest_sha256"] == freeze["manifests"][split]["sha256"])
        check("reused ref protocol differs",
              ref_config["steps"] == STEPS and ref_config["seeds"] == list(SEEDS) and ref_config["pilot"] is False)
        check("reused ref motion model differs", ref_config["motion_model"] == config["motion_model"])
        roots["ref"] = ref_root

    hits = {arm: [0] * 4 for arm in ARMS}
    usable = {arm: 0 for arm in ARMS}
    peak, p95 = {arm: 0 for arm in ARMS}, {arm: 0 for arm in ARMS}
    clips = {}
    expected_kinds = Counter(row["kind"] for row in rows)
    check("manifest is not the frozen balanced v1 split", expected_kinds == Counter({k: 16 for k in range(1, 5)}))
    for arm in ARMS:
        arm_dir = roots[arm] / arm
        try:
            summary = read_summary(arm_dir)
        except (OSError, ValueError) as exc:
            problems.append(f"{arm} summary unreadable: {exc}")
            continue
        check(f"{arm}: incomplete measurement", bool(summary.get("single_session_measurement_complete")))
        check(f"{arm}: requests {summary.get('completed')}/{len(plan)}",
              summary.get("completed") == len(plan))
        peak[arm] = int(summary.get("peak_used_mib") or 0)
        p95[arm] = int(summary.get("p95_ms") or 0)
        keys = set()
        for line in (arm_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            keys.add((record["id"], record["seed"]))
        check(f"{arm}: journal does not cover the plan", keys == set(plan))
        for row in rows:
            for seed in SEEDS:
                result = clip_score(arm_dir, row, seed, yaw_sign)
                clips[(arm, row["id"], seed)] = result
                if result["hit"]:
                    hits[arm][row["kind"] - 1] += 1
                if result["usable"]:
                    usable[arm] += 1
        check(f"{arm}: scoring missed clips", len([k for k in clips if k[0] == arm]) == len(plan))

    candidates = {arm: candidate_of(roots[arm], arm) for arm in STUDENTS}
    check("en/ko checkpoints differ", candidates["en"] and candidates["en"] == candidates["ko"])
    for arm in STUDENTS:
        env = (candidates[arm] or {}).get("environment", {})
        check(f"{arm}: student environment is not Python 3.13", env.get("python", [0, 0])[:2] == [3, 13])
    hub = Path(roots["en"] / "student.json")
    student = json.loads(hub.read_text(encoding="utf-8")) if hub.is_file() else {}
    cache_dir = student.get("cache_env", {}).get("HF_HUB_CACHE")
    forbidden = ("models--meta-llama", "models--mcgill-nlp")
    names = {entry.name for entry in Path(cache_dir).iterdir()} if cache_dir and Path(cache_dir).is_dir() else set()
    check("student cache holds Llama or LLM2Vec teacher weights",
          not any(name.lower().startswith(forbidden) for name in names))
    standalone = (not problems and all(
        clips.get((arm, row["id"], seed), {}).get("hit") is not None and
        (roots[arm] / "student.json").is_file() for arm in STUDENTS for row in rows for seed in SEEDS)
        and all(clips[(arm, row["id"], seed)]["reason"] not in ("missing_artifacts",) for arm in STUDENTS
                for row in rows for seed in SEEDS))

    if export_evidence:
        evidence = json.loads(Path(export_evidence).read_text(encoding="utf-8"))
        exports = bool(evidence.get("verified")) and set(evidence.get("arms", [])) == set(STUDENTS)
    else:
        exports = False
        problems.append("no export evidence supplied")
    if videos:
        directory = Path(videos)
        check("comparison videos missing", directory.is_dir() and any(directory.iterdir()))
    else:
        problems.append("no comparison video directory supplied")

    measurement = {"hits": {arm: hits[arm] for arm in ARMS}, "usable": {arm: usable[arm] for arm in ARMS},
                   "peak": {arm: peak[arm] for arm in ARMS}, "p95": {arm: p95[arm] for arm in ARMS},
                   "evidence": False, "standalone": bool(standalone), "exports": bool(exports)}
    ranking = v1_score.score(measurement) if v1_score.valid(measurement) else None
    passed = v1_score._checks(measurement)[0] if v1_score.valid(measurement) else [False] * len(BUNDLE_NAMES)
    report = {
        "schema": "kimodo.v1-measurement.v1", "stage": stage, "split": split, "run_root": str(run_root),
        "ref_root": str(roots["ref"]), "candidate": candidates["en"], "measurement": measurement,
        "bundle_passes": dict(zip(BUNDLE_NAMES, map(bool, passed))), "dev_ranking": ranking,
        "clips": {"-".join((arm, row_id, str(seed))): result for (arm, row_id, seed), result in clips.items()},
        "evidence_problems": problems,
        "reproduction": " ".join(
            [f"python docs/experiment/v1_evidence.py measure --lock {lock_path} --run-root {run_root}",
             f"--stage {stage} --yaw-sign {yaw_sign}"] + ([f"--ref-root {roots['ref']}"] if ref_root else [])
             + ([f"--export-evidence {export_evidence}"] if export_evidence else [])
             + ([f"--videos {videos}"] if videos else [])),
        "evidence": not problems,
    }
    measurement["evidence"] = bool(report["evidence"])
    if out:
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    return report


def decide(report_path, state_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    statePath = Path(state_path)
    state = json.loads(statePath.read_text(encoding="utf-8")) if statePath.is_file() else v1_score.init_state()
    if state["phase"] == "Dev" and state.get("candidate_id") and report["candidate"]:
        same = (report["candidate"]["checkpoint_sha256"] == state["candidate_id"]
                and report["candidate"]["entry_sha256"] == state.get("candidate_entry"))
        if report["stage"] == "Dev" and not same and state["trial"] > 0:
            problems = report.get("evidence_problems") or []
            problems.append("dev round changed both candidate and inputs together")
            report["evidence_problems"] = problems
    measurement = report["measurement"]
    passed, student_hits = v1_score._checks(measurement) if v1_score.valid(measurement) else ([False] * 7, 0)
    entry = {"trial": state["trial"] + 1 if state["phase"] == "Dev" else state["trial"],
             "stage": report["stage"], "candidate_id": (report["candidate"] or {}).get("checkpoint_sha256"),
             "gate": bool(v1_score.gate(measurement)), "bundle_passes": sum(map(bool, passed)),
             "ranking": v1_score.score(measurement) if v1_score.valid(measurement) else None,
             "student_intent_hits": student_hits,
             "totals": {arm: sum(measurement["hits"][arm]) for arm in ARMS},
             "usable": measurement["usable"], "evidence": measurement["evidence"],
             "standalone": measurement["standalone"], "exports": measurement["exports"],
             "peak_mib": measurement["peak"], "p95_ms": measurement["p95"],
             "evidence_problems": report.get("evidence_problems", [])}
    state = v1_score.advance(state, measurement)
    state.setdefault("history", []).append(entry)
    if state["phase"] == "Frozen" and entry["candidate_id"]:
        state["candidate_id"] = entry["candidate_id"]
        state["candidate_entry"] = (report["candidate"] or {}).get("entry_sha256")
    elif state["phase"] == "Dev" and entry["candidate_id"] and not state.get("candidate_id"):
        state["candidate_id"] = entry["candidate_id"]
        state["candidate_entry"] = (report["candidate"] or {}).get("entry_sha256")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(statePath.parent),
                                     prefix=".v1-state-", delete=False) as tmp:
        json.dump(state, tmp, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(statePath)
    return state, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="Score one run and emit a measurement report")
    measuring.add_argument("--lock", required=True, type=Path)
    measuring.add_argument("--run-root", required=True, type=Path)
    measuring.add_argument("--stage", choices=("Dev", "Frozen"), default="Dev")
    measuring.add_argument("--yaw-sign", type=int, required=True, choices=(-1, 1))
    measuring.add_argument("--ref-root", type=Path, help="Reuse a prior ref arm from another run root")
    measuring.add_argument("--export-evidence", type=Path)
    measuring.add_argument("--videos", type=Path)
    measuring.add_argument("--out", required=True, type=Path)
    deciding = commands.add_parser("decide", help="Consume a report and persist the TLA+ state")
    deciding.add_argument("--report", required=True, type=Path)
    deciding.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        report = measure(args.lock, args.run_root, args.stage, args.yaw_sign, args.ref_root,
                         args.export_evidence, args.videos, args.out)
        print(json.dumps({"stage": args.stage, "gate": v1_score.gate(report["measurement"]) if
                          v1_score.valid(report["measurement"]) else False,
                          "evidence": report["evidence"],
                          "totals": {arm: sum(report["measurement"]["hits"][arm]) for arm in ARMS},
                          "usable": report["measurement"]["usable"],
                          "peak": report["measurement"]["peak"], "p95": report["measurement"]["p95"]},
                         ensure_ascii=False))
    else:
        state, entry = decide(args.report, args.state)
        print(json.dumps({"state": state, "entry": entry}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
