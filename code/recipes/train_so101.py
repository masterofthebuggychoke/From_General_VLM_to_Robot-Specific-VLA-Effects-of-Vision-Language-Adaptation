"""Run one of the four recorded SO-101 real-data adaptations."""

import argparse
import json
from pathlib import Path
import sys


def options(args):
    if args.branch == "direct":
        return {
            "policy.path": str(args.init.resolve()),
            "dataset.repo_id": "DONGDA/pick_and_place_real_merged",
            "dataset.root": str(args.data.resolve()),
            "rename_map": {
                "observation.images.top": "observation.images.camera1",
                "observation.images.wrist": "observation.images.camera2",
            },
            "batch_size": 16,
            "accelerator.gradient_accumulation.steps": 4,
            "accelerator.mixed_precision": "bf16",
            "steps": 30000,
            "save_freq": 10000,
            "log_freq": 100,
            "num_workers": 8,
            "policy.device": "cuda",
            "policy.freeze_vision_encoder": False,
            "policy.train_expert_only": False,
            "policy.push_to_hub": False,
            "wandb.enable": False,
            "seed": 1000,
            "output_dir": str(args.output.resolve()),
            "job_name": args.output.name,
        }
    frozen = args.branch in ("original", "aligned")
    opts = {
        "policy.path": str(args.init.resolve()),
        "policy.vlm_model_name": str(args.vlm.resolve()),
        "policy.load_vlm_weights": True,
        "policy.freeze_vision_encoder": frozen,
        "policy.train_expert_only": frozen,
        "policy.train_state_proj": True,
        "policy.device": "cuda",
        "policy.use_amp": True,
        "policy.push_to_hub": False,
        "policy.scheduler_warmup_steps": 1000,
        "policy.scheduler_decay_steps": 30000,
        "dataset.repo_id": "local/so101_selected_1000",
        "dataset.root": str(args.data.resolve()),
        "dataset.eval_split": 0.0,
        "dataset.image_transforms.enable": False,
        "output_dir": str(args.output.resolve()),
        "job_name": args.output.name,
        "seed": 0,
        "steps": 30000,
        "batch_size": 16,
        "accelerator.gradient_accumulation.steps": 1,
        "accelerator.mixed_precision": "bf16",
        "save_freq": 5000,
        "log_freq": 100,
        "num_workers": 4,
        "prefetch_factor": 2,
        "env_eval_freq": 0,
        "eval_steps": 0,
        "wandb.enable": False,
    }
    if not frozen:
        opts["policy.optimizer_lr"] = 1e-4
    return opts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=["original", "aligned", "curriculum", "direct"],
        required=True,
    )
    parser.add_argument(
        "--init",
        type=Path,
        required=True,
        help="Starting pretrained_model directory for this branch",
    )
    parser.add_argument(
        "--vlm",
        type=Path,
        help="Corresponding VLM export for one of the 30k transfer branches",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Raw LeRobot dataset for direct, prepare_so101 output for the 30k branches",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved LeRobot overrides without training",
    )
    args = parser.parse_args()
    if (args.branch == "direct") == bool(args.vlm):
        parser.error(
            "--vlm is required for the three 30k branches and unused for direct"
        )
    opts = options(args)
    if args.print_config:
        print(json.dumps(opts, indent=2))
        return
    from lerobot.scripts import lerobot_train
    from train_sim import as_flags

    if args.branch != "direct":
        from so101_data import make_datasets

        lerobot_train.make_train_eval_datasets = make_datasets
        lerobot_train.get_step_checkpoint_dir = (
            lambda out, total, step: Path(out) / "checkpoints" / f"{step:06d}"
        )
    if args.branch == "curriculum":
        make_policy = lerobot_train.make_policy

        def make_trainable(*a, **kw):
            policy = make_policy(*a, **kw)
            policy.requires_grad_(True)
            return policy

        lerobot_train.make_policy = make_trainable
    sys.argv = [sys.argv[0], *as_flags(opts)]
    lerobot_train.main()


if __name__ == "__main__":
    main()
