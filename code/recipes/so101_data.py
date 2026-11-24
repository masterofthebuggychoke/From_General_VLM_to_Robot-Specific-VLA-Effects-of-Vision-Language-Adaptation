"""Episode-boundary-safe adapter for the pinned 1000-episode selection."""

import bisect, json, pathlib
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class SelectedDataset(Dataset):
    def __init__(self, split="train", chunk_size=50, data_dir=None):
        DEST = Path(data_dir)
        obj = json.loads((DEST / "prepared.json").read_text())
        self.rows = [r for r in obj["episodes"] if r["split"] == split]
        self.chunk_size = chunk_size
        self.arrays = [
            dict(np.load(DEST / "episodes" / f'{r["id"]:04d}.npz')) for r in self.rows
        ]
        lengths = [r["frames"] for r in self.rows]
        self.ends = np.cumsum(lengths).tolist()
        self.starts = [0] + self.ends[:-1]
        self.num_frames = self.ends[-1]
        self.num_episodes = len(self.rows)
        self.episodes = None
        self.absolute_to_relative_idx = None
        self.image_transforms = None
        features = {
            k: {"dtype": "float32", "shape": (6,), "names": NAMES}
            for k in ["action", "observation.state"]
        }
        for v in ["top", "wrist"]:
            features["observation.images." + v] = {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            }
        self.meta = SimpleNamespace(
            features=features,
            stats=obj["stats"],
            episodes={"dataset_from_index": self.starts, "dataset_to_index": self.ends},
            camera_keys=["observation.images.top", "observation.images.wrist"],
            depth_keys=[],
            has_language_columns=False,
            fps=30,
            robot_type="so101_follower",
        )

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        from lerobot.datasets.video_utils import decode_video_frames

        e = bisect.bisect_right(self.ends, index)
        offset = index - self.starts[e]
        r = self.rows[e]
        a = self.arrays[e]
        n = r["frames"]
        q = np.arange(offset, offset + self.chunk_size)
        pad = q >= n
        q = np.minimum(q, n - 1)
        out = {
            "observation.state": torch.from_numpy(a["state"][offset].copy()),
            "action": torch.from_numpy(a["action"][q].copy()),
            "action_is_pad": torch.from_numpy(pad),
            "task": r["task"],
            "episode_index": e,
            "index": index,
        }
        for v, item in r["videos"].items():
            t = float(item["start"] + a["timestamp"][offset])
            out["observation.images." + v] = decode_video_frames(
                item["local_path"],
                [t],
                tolerance_s=0.02,
                backend="torchcodec",
                return_uint8=True,
            )[0]
        return out


class EvaluationSubset(torch.utils.data.Subset):
    def __init__(self, dataset):
        # Deterministic coverage across all held-out episodes, five positions per episode.
        indices = [
            s + int(x)
            for s, r in zip(dataset.starts, dataset.rows)
            for x in np.linspace(0, r["frames"] - 1, 5)
        ]
        super().__init__(dataset, indices)
        self.meta = dataset.meta


def make_datasets(cfg):
    train = SelectedDataset("train", cfg.policy.chunk_size, cfg.dataset.root)
    meta = json.loads((Path(cfg.dataset.root) / "prepared.json").read_text())
    evaluation = (
        EvaluationSubset(
            SelectedDataset("validation", cfg.policy.chunk_size, cfg.dataset.root)
        )
        if meta["validation_episodes"]
        else None
    )
    return train, evaluation
