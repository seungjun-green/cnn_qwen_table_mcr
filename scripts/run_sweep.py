from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an ordered, disconnect-safe configuration sweep"
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--mirror-root",
        required=True,
        help="Persistent directory containing one mirrored folder per config",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Persistent sweep state JSON (defaults to MIRROR_ROOT/sweep_state.json)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later configs if one training command fails",
    )
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "created_at_unix": time.time(), "runs": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("version") != 1 or not isinstance(state.get("runs"), dict):
        raise ValueError(f"Unsupported sweep state format: {path}")
    return state


def _config_fingerprint(config: Any) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    mirror_root = Path(args.mirror_root)
    mirror_root.mkdir(parents=True, exist_ok=True)
    state_path = (
        Path(args.state_file) if args.state_file else mirror_root / "sweep_state.json"
    )
    state = _load_state(state_path)

    resolved_configs: list[tuple[str, Path, Path, str]] = []
    run_names: set[str] = set()
    for config_argument in args.configs:
        config_path = Path(config_argument).resolve()
        config = load_config(config_path)
        run_name = Path(config.training.output_dir).name
        if run_name in run_names:
            raise ValueError(
                f"Sweep configs must have unique output directory names: {run_name}"
            )
        run_names.add(run_name)
        resolved_configs.append(
            (
                config_argument,
                config_path,
                mirror_root / run_name,
                _config_fingerprint(config),
            )
        )

    state["order"] = [item[0] for item in resolved_configs]
    state["updated_at_unix"] = time.time()
    write_json(state, state_path)

    for position, (config_key, config_path, mirror_dir, fingerprint) in enumerate(
        resolved_configs, start=1
    ):
        entry = state["runs"].setdefault(
            config_key,
            {
                "position": position,
                "status": "pending",
                "attempts": 0,
                "mirror_output_dir": str(mirror_dir),
                "config_fingerprint": fingerprint,
            },
        )
        if entry.get("config_fingerprint") != fingerprint:
            entry["status"] = "pending"
            entry["config_changed_at_unix"] = time.time()
        entry["position"] = position
        entry["mirror_output_dir"] = str(mirror_dir)
        entry["config_fingerprint"] = fingerprint
        if entry.get("status") == "completed":
            print(
                f"[{position}/{len(resolved_configs)}] Skipping completed {config_key}"
            )
            continue

        entry["status"] = "running"
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["started_at_unix"] = time.time()
        state["active_config"] = config_key
        state["updated_at_unix"] = time.time()
        write_json(state, state_path)
        separator = "#" * 88
        print(f"\n{separator}", flush=True)
        print(
            f"SWEEP CONFIG {position}/{len(resolved_configs)}: {mirror_dir.name}",
            flush=True,
        )
        print(f"CONFIG FILE:  {config_path}", flush=True)
        print(f"DRIVE OUTPUT: {mirror_dir}", flush=True)
        print(f"{separator}\n", flush=True)

        command = [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--config",
            str(config_path),
            "--mirror-output-dir",
            str(mirror_dir),
        ]
        try:
            result = subprocess.run(command, cwd=ROOT, check=False)
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            entry["interrupted_at_unix"] = time.time()
            state["updated_at_unix"] = time.time()
            write_json(state, state_path)
            raise

        entry["return_code"] = result.returncode
        entry["finished_at_unix"] = time.time()
        if result.returncode == 0:
            entry["status"] = "completed"
        else:
            entry["status"] = "failed"
        print(f"\n{separator}", flush=True)
        print(
            f"SWEEP CONFIG {position}/{len(resolved_configs)} FINISHED: "
            f"{mirror_dir.name} ({entry['status']})",
            flush=True,
        )
        print(f"{separator}\n", flush=True)
        state["active_config"] = None
        state["updated_at_unix"] = time.time()
        write_json(state, state_path)
        if result.returncode != 0 and not args.continue_on_error:
            raise SystemExit(result.returncode)

    state["status"] = (
        "completed"
        if all(
            state["runs"].get(key, {}).get("status") == "completed"
            for key, _, _, _ in resolved_configs
        )
        else "incomplete"
    )
    state["active_config"] = None
    state["updated_at_unix"] = time.time()
    write_json(state, state_path)
    print(f"Sweep status: {state['status']}; state saved to {state_path}")


if __name__ == "__main__":
    main()
