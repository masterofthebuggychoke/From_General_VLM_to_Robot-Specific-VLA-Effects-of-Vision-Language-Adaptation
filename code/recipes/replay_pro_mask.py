"""Replay one indexed Pro2 frame and verify its visible instance boxes and RGB gates."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
import mujoco
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from prepare_pro_qa import read_index


# NumPy-safe segmentation readback; adapted from robosuite (third_party/Robosuite-LICENSE).
def _fixed_read_pixels(self, width, height, depth=False, segmentation=False):
    """Work around robosuite's uint8 * 256 overflow with newer NumPy."""
    viewport = mujoco.MjrRect(0, 0, width, height)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    depth_img = np.empty((height, width), dtype=np.float32) if depth else None
    mujoco.mjr_readPixels(rgb=rgb, depth=depth_img, viewport=viewport, con=self.con)
    result = rgb
    if segmentation:
        channels = rgb.astype(np.int32)
        ids = channels[:, :, 0] + channels[:, :, 1] * 256 + channels[:, :, 2] * 65536
        ids[ids >= self.scn.ngeom + 1] = 0
        geom_ids = np.full((self.scn.ngeom + 1, 2), -1, dtype=np.int32)
        for index in range(self.scn.ngeom):
            geom = self.scn.geoms[index]
            if geom.segid != -1:
                geom_ids[geom.segid + 1] = [geom.objtype, geom.objid]
        result = geom_ids[ids]
    return (result, depth_img) if depth else result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="pro2_raw_b0d66b62f4143b6e_f0035")
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Original native Pro2 training archive",
    )
    p.add_argument("--simulator", required=True, type=Path)
    args = p.parse_args()
    from libero.libero import benchmark
    from libero.libero.envs import SegmentationRenderEnv
    from libero_remote.environment.libero import (
        _set_perturbation_seed,
        _settle_environment,
        _begin_perturbation_rollout,
        _set_controller_mode,
    )
    import torch

    MjRenderContextOffscreen.read_pixels = _fixed_read_pixels
    scene = next((s for s in read_index() if s["scene_id"] == args.scene), None)
    if scene is None:
        p.error("Unknown accepted scene")
    suite, task, case, _ = scene["episode_id"].split("/")
    tid = int(task.split("_")[-1])
    manifest = json.loads((args.simulator / "assets/task_manifest.json").read_text())[
        "tasks"
    ]
    spec = next(
        x
        for x in manifest
        if (x["suite"], x["base_task_id"], x["case_id"]) == (suite, tid, case)
    )
    bddl = args.simulator / "assets" / spec["bddl_path"]
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != spec["bddl_sha256"]:
        raise ValueError("BDDL hash mismatch")
    with h5py.File(args.source / scene["hdf5_path"]) as h5:
        episode = h5[scene["hdf5_group"]]
        meta = json.loads(episode.attrs["source_metadata"])
        original = np.asarray(episode["obs/agentview_image"][scene["frame"]])
        actions = np.asarray(episode["actions"][: scene["frame"]])
    if hashlib.sha256(original.tobytes()).hexdigest() != scene["native_pixel_sha256"]:
        raise ValueError("Recorded frame hash mismatch")
    with torch.serialization.safe_globals(
        [
            (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.ndarray,
            np.dtype,
            type(np.dtype("float64")),
        ]
    ):
        states = benchmark.get_benchmark_dict()[suite]().get_task_init_states(tid)
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
    )
    try:
        seed = meta["seed"]
        env.seed(seed)
        _set_perturbation_seed(env, seed)
        env.reset()
        env.set_init_state(states[meta["init_state_id"]])
        _settle_environment(env)
        _set_perturbation_seed(env, seed)
        obs = _begin_perturbation_rollout(env)
        _set_controller_mode(env, meta["control_mode"])
        for action in actions:
            obs, _, _, _ = env.step(action)
        rendered = np.asarray(obs["agentview_image"])
        seg = np.asarray(obs["agentview_segmentation_instance"])[:, :, 0]
        delta = np.abs(original.astype(float) - rendered.astype(float))
        mae = float(delta.mean())
        cor = float(np.corrcoef(original.ravel(), rendered.ravel())[0, 1])
        assert mae <= 3 and np.isfinite(cor) and cor >= 0.97, (mae, cor)
        for instance, (_, expected) in scene["objects"].items():
            yy, xx = np.where(seg == env.instance_to_id[instance])
            box = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
            assert [coord * 4 for coord in box] == expected, instance
            x1, y1, x2, y2 = box
            region = delta[y1:y2, x1:x2]
            assert region.mean() <= 5 and np.percentile(region, 95) <= 25, (
                instance,
                mae,
                cor,
                float(region.mean()),
                float(np.percentile(region, 95)),
            )
        print(
            json.dumps(
                {
                    "scene": args.scene,
                    "frame_mae": mae,
                    "correlation": cor,
                    "boxes_matched": len(scene["objects"]),
                }
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
