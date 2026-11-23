"""Run offline tests, optionally check native seams or make ONE keyless-action API smoke call."""
import argparse
import ast
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_upstream():
    contracts = {
        "src/engine/AutoControl.py": {"run", "_check_current_platform", "_check_vertical_passage",
                                      "_find_nearest_verti_passage", "_move_to_verti_passage",
                                      "_verti_movement", "_enable_player_patrol", "_fk_that_mob",
                                      "_unstuck_player", "get_debug_geometry"},
        "src/engine/GameBot.py": {"run", "_load_game_resources", "_is_window_valid",
                                  "_is_game_window_foreground", "_toggle_bot"},
        "src/action/KeyBoardController.py": {"_key_down", "_key_up", "_press_direction_locked",
                                             "_release_direction_locked", "release_all"},
    }
    for filename, required in contracts.items():
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
        methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        if required - methods:
            raise RuntimeError(f"Upstream interface changed: {filename}: {sorted(required - methods)}")
        if filename.endswith("KeyBoardController.py"):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr == "Thread":
                    for keyword in node.keywords:
                        if keyword.arg == "target" and isinstance(keyword.value, ast.Attribute):
                            if not keyword.value.attr.endswith("_command"):
                                raise RuntimeError("Unwrapped upstream keyboard worker target")
    return {"status": "passed", "kind": "static_interface_check", "files": list(contracts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="store_true", help="Check native source interfaces in the checkout")
    parser.add_argument("--live", action="store_true", help="One billed Jev API call on synthetic data; no game or input")
    parser.add_argument("--output", default="artifacts/policy")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests/policy"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    print(text)
    (output / "tests.txt").write_text(text, encoding="utf-8")
    report = {"kind": "offline_tests", "tests": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "live_api": "not_requested", "live_gameplay": "not_tested"}
    success = result.wasSuccessful()
    if args.upstream:
        try:
            report["upstream"] = verify_upstream()
        except Exception as exc:
            success = False
            report["upstream"] = {"status": "failed", "error": str(exc)}
    if args.live and success:
        from src.policy.client import JevClient, Settings
        started = time.monotonic()
        try:
            answer = JevClient(Settings.from_env()).decide(
                {"experiment": "synthetic API contract smoke test; no live game or input backend",
                 "player": {"minimap_xy": [20, 5], "hp_percent": 75},
                 "platforms": [{"t_l": [0, 0], "b_r": [100, 10]}], "mob_count": 0},
                ("IDLE", "PATROL"),
            )
            report["live_api"] = {"status": "passed", "latency_ms": (time.monotonic() - started) * 1000,
                                  "answer": asdict(answer), "note": "Schema/connectivity only; not gameplay quality"}
        except Exception as exc:
            success = False
            report["live_api"] = {"status": "failed", "error_type": type(exc).__name__}
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
