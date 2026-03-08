import os
import argparse
import importlib.util

DEFAULT_CONFIG_PATH = "config.py"
config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)


spec = importlib.util.spec_from_file_location("config", config_path)
if spec is None or spec.loader is None:
    raise FileNotFoundError(f"Could not load config module from path: {config_path}")

config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
cfg = config_module.cfg

RUN_DIR = cfg["run_dir"]

def main(run_dir: str | None = None) -> None:
    resolved_run_dir = run_dir or RUN_DIR
    flag_path = os.path.join(resolved_run_dir, "stop.flag")
    os.makedirs(resolved_run_dir, exist_ok=True)

    tmp_path = f"{flag_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("stop\n")
    os.replace(tmp_path, flag_path)
    print(f"[request_shutdown] Stop flag created: {flag_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Request graceful shutdown by creating stop.flag")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory to place stop.flag in. Overrides run_dir from config.",
    )
    args = parser.parse_args()
    main(run_dir=args.run_dir)