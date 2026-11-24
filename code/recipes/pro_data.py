"""Adapt pinned public Pro2 videos to the existing 8D SmolVLA input contract."""

import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from lerobot.processor.env_processor import LiberoProcessorStep

ROOT = Path(__file__).resolve().parents[1] / "data/pro"
REPO = "phicoltan/libero_pro2"
REVISION = "e527a3276cc33258e9979e6c322b703bc8c0ef62"
CAMERAS = {
    "observation.images.agentview": "observation.images.image",
    "observation.images.wrist": "observation.images.image2",
}


def state8(state):
    return torch.cat(
        [
            state[..., :3],
            LiberoProcessorStep()._quat2axisangle(state[..., 3:7]),
            state[..., 7:9],
        ],
        dim=-1,
    )


class AdaptedDataset(Dataset):
    def __init__(self, reference):
        self.reference = reference
        self.meta = copy.copy(reference.meta)
        self.meta.info = copy.deepcopy(reference.meta.info)
        features = {
            CAMERAS.get(k, k): copy.deepcopy(v)
            for k, v in reference.meta.features.items()
            if k in CAMERAS or k in ("observation.state", "action")
        }
        features["observation.state"]["shape"] = (8,)
        features["observation.state"]["names"] = [
            "eef_x",
            "eef_y",
            "eef_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "gripper_left",
            "gripper_right",
        ]
        self.meta.info.features = features
        stats = json.loads((ROOT / "selected_stats.json").read_text())
        self.meta.stats = {
            k: {s: np.asarray(v) for s, v in values.items()}
            for k, values in stats.items()
        }

    def __getattr__(self, key):
        return getattr(self.reference, key)

    def __len__(self):
        return len(self.reference)

    @staticmethod
    def adapt(row):
        row = dict(row)
        row["observation.state"] = state8(row["observation.state"])
        # Public conversion rotates native HDF5 images by 180 degrees. Undo
        # this for consistency with the existing native Pro evaluator/VQA.
        for source, target in CAMERAS.items():
            row[target] = torch.flip(row.pop(source), dims=(-2, -1))
        row.pop("observation.joint_pos", None)
        row.pop("observation.joint_vel", None)
        return row

    def __getitem__(self, index):
        return self.adapt(self.reference[index])

    def __getitems__(self, indices):
        return [self.adapt(row) for row in self.reference.__getitems__(indices)]


def install(trainer, *, smoke=False):
    from lerobot.datasets import video_utils

    video_utils._default_decoder_cache = video_utils.VideoDecoderCache(max_size=128)
    original = trainer.make_train_eval_datasets
    selected = json.loads((ROOT / "selected_episodes.json").read_text())

    def make(cfg):
        cfg.dataset.repo_id = REPO
        cfg.dataset.revision = REVISION
        cfg.dataset.episodes = selected[:2] if smoke else selected
        # Built-in torchcodec LRU avoids reopening long AV1 files per frame.
        # 128 entries cover the pinned release's video shards; decoding uses
        # one FFmpeg thread per decoder by default.
        cfg.dataset.video_backend = "torchcodec"
        cfg.dataset.video_decoder_cache_size = None
        reference, evaluation = original(cfg)
        assert evaluation is None
        wrapped = AdaptedDataset(reference)
        if not smoke:
            assert wrapped.num_frames == 4115692 and wrapped.num_episodes == 18751
        return wrapped, None

    trainer.make_train_eval_datasets = make
