"""Read-only machine snapshot; does not change clocks or power settings."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"unavailable": str(exc)}


def containers(format_string):
    try:
        query = subprocess.run(
            ["docker", "ps", "--format", format_string], capture_output=True, text=True
        )
    except OSError as exc:
        return {"unavailable": str(exc)}
    if query.returncode:
        return {"unavailable": query.stderr.strip()}
    return query.stdout.splitlines()


def gpu_devfreq():
    thor = Path("/sys/class/devfreq/gpu-gpc-0")
    return thor if thor.is_dir() else Path("/sys/class/devfreq/17000000.gpu")


def sample():
    power = subprocess.run(["/usr/sbin/nvpmodel", "-q"], capture_output=True, text=True)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "l4t": read("/etc/nv_tegra_release"),
        "meminfo": read("/proc/meminfo"),
        "gpu_devfreq_hz": {
            name: read(gpu_devfreq() / name) for name in ["cur_freq", "min_freq", "max_freq"]
        },
        "power_mode_query": {
            "returncode": power.returncode,
            "stdout": power.stdout,
            "stderr": power.stderr,
        },
        "background_containers": containers("{{.Names}}"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(sample(), ensure_ascii=False, indent=2))
