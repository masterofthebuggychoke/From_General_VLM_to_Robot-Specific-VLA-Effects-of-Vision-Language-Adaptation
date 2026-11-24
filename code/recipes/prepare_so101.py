"""Prepare local SO-101 demonstration tables and camera pointers for action training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_parquet_dir(folder):
    return pd.concat(
        [pd.read_parquet(path) for path in sorted(folder.rglob("*.parquet"))]
    )


def prepare(source, output):
    source = source.resolve()
    info = json.loads((source / "meta/info.json").read_text())
    episodes = read_parquet_dir(source / "meta/episodes")
    frames = read_parquet_dir(source / "data")
    assert info["fps"] == 30 and len(episodes) == info["total_episodes"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    records = []
    values = {"action": [], "observation.state": []}
    for ep in episodes.to_dict("records"):
        idx = int(ep["episode_index"])
        frame = frames.loc[frames.episode_index == idx].sort_values("frame_index")
        action = np.stack(frame.action).astype("<f4")
        state = np.stack(frame["observation.state"]).astype("<f4")
        times = frame.timestamp.to_numpy(dtype="float64")
        assert action.shape == state.shape == (ep["length"], 6)
        np.savez(
            output / "episodes" / f"{idx:04d}.npz",
            action=action,
            state=state,
            timestamp=times - times[0],
        )
        values["action"].append(action)
        values["observation.state"].append(state)
        videos = {}
        for name in ("top", "wrist"):
            camera = "observation.images." + name
            key = "videos/" + camera
            path = source / info["video_path"].format(
                video_key=camera,
                chunk_index=int(ep[key + "/chunk_index"]),
                file_index=int(ep[key + "/file_index"]),
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            videos[name] = {
                "local_path": str(path),
                "start": float(ep[key + "/from_timestamp"]),
                "camera": camera,
            }
        records.append(
            {
                "id": idx,
                "episode_index": idx,
                "frames": len(frame),
                "task": ep["tasks"][0],
                "split": "train",
                "videos": videos,
                "action_state_sha256": hashlib.sha256(
                    action.tobytes() + state.tobytes()
                ).hexdigest(),
            }
        )
    stats = {}
    for key, parts in values.items():
        array = np.concatenate(parts).astype("float64")
        stats[key] = {
            "mean": array.mean(0).tolist(),
            "std": np.maximum(array.std(0), 1e-6).tolist(),
            "min": array.min(0).tolist(),
            "max": array.max(0).tolist(),
            "count": [len(array)],
        }
    result = {
        "episodes": sorted(records, key=lambda ep: ep["id"]),
        "stats": stats,
        "train_episodes": len(records),
        "validation_episodes": 0,
        "validation_ids": [],
    }
    (output / "prepared.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local LeRobot SO-101 dataset with two cameras",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
