#!/usr/bin/env python3
"""Tool bridge from Gemma function calls to the Pi vision overlay state."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("GEMMA4_ROBOT_ROOT", Path.home() / "gemma4-robot"))
STATE_PATH = Path(os.environ.get("GEMMA_VISION_STATE", ROOT / "kiosk" / "vision_state.json"))
COMMAND_PATH = Path(os.environ.get("GEMMA_VISION_COMMAND", ROOT / "kiosk" / "vision_command.json"))


def read_tool_call() -> dict[str, Any]:
    try:
        raw = os.read(0, 1_000_000)
    except OSError:
        raw = b""
    if not raw.strip():
        return {"args": {}}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {"args": {}, "input_error": str(exc)}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def now() -> float:
    return time.time()


def state_age(state: dict[str, Any]) -> float | None:
    updated = state.get("updated_at_unix")
    if not isinstance(updated, (int, float)):
        return None
    return max(0.0, now() - float(updated))


def wait_for_human(args: dict[str, Any]) -> dict[str, Any]:
    stable_seconds = float(args.get("stable_seconds") or 0.8)
    timeout_seconds = float(args.get("timeout_seconds") or 120.0)
    deadline = now() + timeout_seconds
    visible_since: float | None = None
    last_state: dict[str, Any] = {}

    while now() < deadline:
        state = read_json(STATE_PATH)
        last_state = state
        age = state_age(state)
        fresh = age is not None and age <= 2.0
        human_present = bool(state.get("human_present")) and fresh
        if human_present:
            if visible_since is None:
                visible_since = now()
            if now() - visible_since >= stable_seconds:
                return {
                    "human_detected": True,
                    "stable_seconds": round(now() - visible_since, 3),
                    "pose_presence": state.get("pose_presence"),
                    "state_age_seconds": round(age or 0.0, 3),
                }
        else:
            visible_since = None
        time.sleep(0.1)

    return {
        "human_detected": False,
        "timeout_seconds": timeout_seconds,
        "last_state_age_seconds": state_age(last_state),
        "last_pose_presence": last_state.get("pose_presence"),
    }


def normalized_milestones(value: Any, target: int, default: list[int] | None = None) -> list[int]:
    if isinstance(value, list):
        milestones = []
        for item in value:
            try:
                count = int(item)
            except (TypeError, ValueError):
                continue
            if count > 0:
                milestones.append(count)
    else:
        milestones = list(default or [3, 6, 9, target])
    if target not in milestones:
        milestones.append(target)
    return sorted(set(count for count in milestones if count <= target))


def coach_squats(args: dict[str, Any]) -> dict[str, Any]:
    target = int(args.get("target_reps") or 10)
    target = max(1, target)
    milestones = normalized_milestones(args.get("milestones"), target)
    reset = bool(args.get("reset", False))
    timeout_seconds = float(args.get("timeout_seconds") or 180.0)

    state = read_json(STATE_PATH)
    active = bool(state.get("coach_active"))
    done = bool(state.get("coach_done"))
    current_target = int(state.get("target_reps") or 0)

    if reset or not active or done or current_target != target:
        command = {
            "command": "coach_squats",
            "session_id": uuid.uuid4().hex,
            "target_reps": target,
            "milestones": milestones,
            "created_at_unix": now(),
        }
        write_json_atomic(COMMAND_PATH, command)
        start_count = 0
    else:
        start_count = int(state.get("squat_count") or 0)

    next_milestone = next((count for count in milestones if count > start_count), target)
    deadline = now() + timeout_seconds
    last_state = state

    while now() < deadline:
        state = read_json(STATE_PATH)
        last_state = state
        count = int(state.get("squat_count") or 0)
        reached = [count for count in milestones if count <= int(state.get("squat_count") or 0)]
        if count >= next_milestone or bool(state.get("coach_done")):
            return {
                "milestone_reached": min(count, target),
                "squat_count": count,
                "target_reps": target,
                "done": count >= target or bool(state.get("coach_done")),
                "next_milestone": next((m for m in milestones if m > count), None),
                "milestones_reached": reached,
                "pose_presence": state.get("pose_presence"),
            }
        time.sleep(0.1)

    return {
        "milestone_reached": False,
        "timeout_seconds": timeout_seconds,
        "squat_count": int(last_state.get("squat_count") or 0),
        "target_reps": target,
        "next_milestone": next_milestone,
        "pose_presence": last_state.get("pose_presence"),
        "human_present": bool(last_state.get("human_present")),
    }


def squat_counter(args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args)
    args["target_reps"] = int(args.get("target_reps") or 4)
    args["milestones"] = normalized_milestones(args.get("milestones"), int(args["target_reps"]), [2, 4])
    return coach_squats(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["wait_for_human", "coach_squats", "squat_counter"])
    ns = parser.parse_args()
    call = read_tool_call()
    args = call.get("args") if isinstance(call.get("args"), dict) else {}

    if ns.tool == "wait_for_human":
        result = wait_for_human(args)
    elif ns.tool == "squat_counter":
        result = squat_counter(args)
    else:
        result = coach_squats(args)

    print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
