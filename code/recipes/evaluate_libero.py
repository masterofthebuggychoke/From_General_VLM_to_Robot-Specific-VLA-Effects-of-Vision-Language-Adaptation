"""Run the original four-suite protocol with a local pretrained_model directory."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from lerobot.scripts import lerobot_eval
    from train_sim import as_flags

    options = {
        "policy.path": str(args.checkpoint.resolve()),
        "policy.device": "cuda",
        "policy.use_amp": True,
        "policy.n_action_steps": 1,
        "env.type": "libero",
        "env.task": "libero_spatial,libero_object,libero_goal,libero_10",
        "env.init_states": True,
        "env.hard_reset": True,
        "env.control_mode": "relative",
        "env.observation_height": 256,
        "env.observation_width": 256,
        "env.max_parallel_tasks": 1,
        "eval.batch_size": 10,
        "eval.max_episodes_rendered": 0,
        "eval.recording": False,
        "eval.n_episodes": 50,
        "seed": 0,
        "output_dir": str(args.output.resolve()),
    }
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    sys.argv = [sys.argv[0], *as_flags(options)]
    lerobot_eval.main()


if __name__ == "__main__":
    main()
