import os

from config import cfg

RUN_DIR = cfg.paths.run_dir
FLAG_PATH = os.path.join(RUN_DIR, "stop.flag")

def main() -> None:
    tmp_path = f"{FLAG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("stop\n")
    os.replace(tmp_path, FLAG_PATH)
    print(f"[request_shutdown] Stop flag created: {FLAG_PATH}")


if __name__ == "__main__":
    main()