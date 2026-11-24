"""Write an isolated LIBERO config from an installed package or Pro2 simulator-source."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro", type=Path, help="Checked-out LIBERO-Pro2.0 root")
    parser.add_argument(
        "--output", type=Path, required=True, help="New LIBERO_CONFIG_PATH directory"
    )
    args = parser.parse_args()
    if args.pro:
        root = args.pro.resolve() / "simulator-source/libero/libero"
    else:
        spec = importlib.util.find_spec("libero")
        if not spec:
            raise RuntimeError("Install the pinned hf-libero package first")
        root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    paths = {
        "benchmark_root": root,
        "assets": root / "assets",
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
    }
    if not args.pro and not paths["assets"].is_dir():
        paths["assets"] = Path.home() / ".cache/libero/assets"
    for path in paths.values():
        if not path.is_dir():
            raise FileNotFoundError(path)
    paths["datasets"] = args.output.resolve() / "datasets"
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "config.yaml").open("x") as config:
        for key, value in paths.items():
            config.write(f"{key}: {json.dumps(str(value))}\n")


if __name__ == "__main__":
    main()
