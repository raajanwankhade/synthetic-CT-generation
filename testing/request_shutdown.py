import os
import time

# Request a graceful stop by creating a flag file.
# Environment overrides:
#   GRACEFUL_STOP_FLAG  -> path to flag file (default: run/stop.flag)

FLAG_PATH = os.environ.get("GRACEFUL_STOP_FLAG", os.path.join("run", "stop.flag"))


def main() -> None:
    os.makedirs(os.path.dirname(FLAG_PATH), exist_ok=True)
    tmp_path = f"{FLAG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("stop\n")
    os.replace(tmp_path, FLAG_PATH)

    print(f"[request_shutdown] Stop flag created: {FLAG_PATH}")


if __name__ == "__main__":
    main()
