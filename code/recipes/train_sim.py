"""Train either LIBERO or the selected Pro2 episodes from an explicit VLM export."""

import argparse
import json
from pathlib import Path
import sys

CONFIG = Path(__file__).resolve().parents[1] / "configs/libero_full.json"


def train_options(args):
    opts = json.loads(CONFIG.read_text())
    frozen = args.tune == "action"
    opts.update(
        {
            "policy.freeze_vision_encoder": frozen,
            "policy.train_expert_only": frozen,
            "policy.vlm_model_name": str(args.vlm.resolve()),
            "dataset.root": str(args.data.resolve()),
            "output_dir": str(args.output.resolve()),
            "job_name": args.output.name,
        }
    )
    if frozen:
        opts["policy.scheduler_decay_steps"] = 32000
    if args.dataset == "pro":
        opts.update(
            {
                "dataset.repo_id": "phicoltan/libero_pro2",
                "dataset.revision": "e527a3276cc33258e9979e6c322b703bc8c0ef62",
                "dataset.video_backend": "torchcodec",
                "steps": 360000,
                "save_freq": 120000,
                "policy.scheduler_decay_steps": 360000,
            }
        )
    return opts


def as_flags(opts):
    return [
        "--"
        + key
        + "="
        + (
            value
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":"))
        )
        for key, value in opts.items()
    ]


def install_arrow_reader(cache):
    from datasets import Dataset
    from lerobot.datasets import dataset_reader, io_utils
    import pyarrow.dataset as arrow

    def load_nested_dataset(pq_dir, features=None, episodes=None):
        files = sorted(Path(pq_dir).glob("*/*.parquet"))
        if not files:
            raise FileNotFoundError(pq_dir)
        filters = (
            arrow.field("episode_index").isin(episodes)
            if episodes is not None
            else None
        )
        return Dataset.from_parquet(
            [str(p) for p in files],
            features=features,
            filters=filters,
            cache_dir=str(cache),
            keep_in_memory=False,
            batch_size=128,
            writer_batch_size=128,
            fragment_scan_options=arrow.ParquetFragmentScanOptions(pre_buffer=False),
        )

    dataset_reader.load_nested_dataset = io_utils.load_nested_dataset = (
        load_nested_dataset
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["libero", "pro"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune", choices=["action", "full"], required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/arrow"))
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    opts = train_options(args)
    if args.print_config:
        print(json.dumps(opts, indent=2))
        return
    from lerobot.scripts import lerobot_train

    install_arrow_reader(args.cache.resolve())
    if args.dataset == "pro":
        from pro_data import install

        install(lerobot_train)
    lerobot_train.get_step_checkpoint_dir = (
        lambda out, total, step: Path(out) / "checkpoints" / f"{step//4:06d}"
    )
    sys.argv = [sys.argv[0], *as_flags(opts)]
    lerobot_train.main()


if __name__ == "__main__":
    main()
