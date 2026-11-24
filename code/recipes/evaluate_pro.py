"""Local Pro2 workers sharing one batched policy; native RGB, paired trial noise.

Smoke output is explicitly unscored. Exceptions abort instead of becoming failures.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def worker(pipe):
    import numpy as np
    import torch
    from libero_remote.environment.libero import LiberoEnvironmentAdapter

    env = LiberoEnvironmentAdapter()
    try:
        while True:
            command, value = pipe.recv()
            if command == "close":
                break
            if command == "create":
                env.close()
                with torch.serialization.safe_globals(
                    [
                        (
                            np._core.multiarray._reconstruct,
                            "numpy.core.multiarray._reconstruct",
                        ),
                        np.ndarray,
                        np.dtype,
                        type(np.dtype("float64")),
                    ]
                ):
                    from robosuite.utils.errors import RandomizationError

                    for attempt in range(12):
                        try:
                            result = env.create(value)
                            break
                        except RandomizationError as exc:
                            # Retry constructor-only incidental placement. Never retry
                            # the seeded benchmark reset or change its seed/init-state.
                            import sys

                            frames = traceback.extract_tb(sys.exc_info()[2])
                            if any(
                                f.name == "reset"
                                and f.filename.endswith("/environment/libero.py")
                                for f in frames
                            ):
                                raise
                            print(
                                json.dumps(
                                    {
                                        "event": "constructor_placement_retry",
                                        "suite": value["suite"],
                                        "task": value["task_id"],
                                        "variant": value["variant"],
                                        "seed": value["seed"],
                                        "attempt": attempt + 1,
                                    }
                                ),
                                flush=True,
                            )
                            env.close()
                            if attempt == 11:
                                raise

            else:
                assert command == "step"
                result = env.step(value)
            pipe.send(("ok", result))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        env.close()
        pipe.close()


def receive(pipe):
    if not pipe.poll(180):
        raise TimeoutError("Pro environment worker did not reply within 180 seconds")
    status, result = pipe.recv()
    if status != "ok":
        raise RuntimeError(result)
    return result


def trials(repeats):
    PRO = Path(os.environ["LIBERO_PRO_ROOT"]).resolve()
    split = json.loads((PRO / "assets/benchmark_split_v1.json").read_text())["splits"][
        "test"
    ]
    cases = json.loads((PRO / "assets/task_manifest.json").read_text())["tasks"]
    cases = [c for c in cases if c["base_task_id"] in split[c["suite"]]]
    assert len(cases) == 339
    result = []
    for c in cases:
        path = PRO / "assets" / c["bddl_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == c["bddl_sha256"]
        for repeat in range(repeats):
            key = f"{c['suite']}/{c['base_task_id']}/{c['case_id']}/{repeat}"
            result.append(
                dict(
                    id=key,
                    case_kind=c["case_kind"],
                    spec=dict(
                        suite=c["suite"],
                        task_id=c["base_task_id"],
                        variant=c["case_id"],
                        _bddl_path=str(path),
                        camera_names=["agentview", "robot0_eye_in_hand"],
                        camera_height=128,
                        camera_width=128,
                        max_steps=c["max_steps"],
                        init_state_id=repeat,
                        seed=repeat,
                        control_mode="relative",
                    ),
                )
            )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--repeats", type=int, choices=[3, 10, 30], default=3)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    assert a.batch_size > 0
    a.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.processor.env_processor import LiberoProcessorStep

    torch.set_num_threads(4)
    cfg = SmolVLAConfig.from_pretrained(a.checkpoint)
    cfg.device = "cuda"
    cfg.n_action_steps = 1
    assert cfg.n_obs_steps == 1 and cfg.num_steps == 10
    policy = SmolVLAPolicy.from_pretrained(a.checkpoint, config=cfg).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=a.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    quaternion = LiberoProcessorStep()
    todo = trials(a.repeats)
    if a.smoke:
        todo = todo[: a.batch_size]
    plan = dict(
        checkpoint=str(a.checkpoint.resolve()),
        batch_size=a.batch_size,
        trials=len(todo),
        repeats=a.repeats,
        smoke=a.smoke,
        scored=not a.smoke,
        denoising_steps=10,
        n_action_steps=1,
        native_rgb=True,
        precision="bf16",
        noise="independent CPU generator seeded by trial ID; shared across model arms",
    )
    plan_path = a.output / "plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Evaluation plan changed"
    else:
        write(plan_path, plan)
    context = mp.get_context("spawn")
    pipes, processes = [], []
    results = (
        [
            json.loads(line)
            for line in (a.output / "episodes.jsonl").read_text().splitlines()
        ]
        if (a.output / "episodes.jsonl").exists()
        else []
    )
    completed = {r["id"] for r in results}
    assert len(completed) == len(results), "Duplicate saved trial IDs"
    assert completed <= {r["id"] for r in todo}, "Unknown saved trials"
    assert all(r["scored"] == (not a.smoke) for r in results)
    print(f"RESUME: {len(results)}/{len(todo)} saved trials", flush=True)
    start = time.perf_counter()
    try:
        for _ in range(min(a.batch_size, len(todo))):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child,))
            process.start()
            child.close()
            pipes.append(parent)
            processes.append(process)
        for offset in range(0, len(todo), a.batch_size):
            group = todo[offset : offset + a.batch_size]
            if all(t["id"] in completed for t in group):
                continue
            for pipe, trial in zip(pipes, group):
                pipe.send(("create", trial["spec"]))
            states = [receive(pipe) for pipe in pipes[: len(group)]]
            languages = [s["language_instruction"] for s in states]
            generators = [
                torch.Generator().manual_seed(
                    int(hashlib.sha256(t["id"].encode()).hexdigest()[:15], 16)
                )
                for t in group
            ]
            active = list(range(len(group)))
            policy.reset()
            step = 0
            while active:
                # Keep batch shape fixed across arms even when success occurs at
                # different times. Finished slots remain inert padding.
                observations = [s["observation"] for s in states]
                raw = {}
                for name, source in [
                    ("image", "agentview_image"),
                    ("image2", "robot0_eye_in_hand_image"),
                ]:
                    raw["observation.images." + name] = (
                        torch.from_numpy(np.stack([o[source] for o in observations]))
                        .permute(0, 3, 1, 2)
                        .float()
                        / 255
                    )
                xyz = torch.from_numpy(
                    np.stack([o["robot0_eef_pos"] for o in observations])
                )
                rot = quaternion._quat2axisangle(
                    torch.from_numpy(
                        np.stack([o["robot0_eef_quat_xyzw"] for o in observations])
                    )
                )
                grip = torch.from_numpy(
                    np.stack([o["robot0_gripper_qpos"] for o in observations])
                )
                raw.update(
                    {
                        "observation.state": torch.cat([xyz, rot, grip], -1),
                        "task": languages,
                    }
                )
                batch = pre(raw)
                noise = torch.stack(
                    [
                        torch.randn(cfg.chunk_size, cfg.max_action_dim, generator=g)
                        for g in generators
                    ]
                ).cuda()
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    actions = (
                        post(policy.predict_action_chunk(batch, noise=noise)[:, 0])
                        .float()
                        .cpu()
                        .numpy()
                    )
                assert np.isfinite(actions).all() and actions.shape == (len(group), 7)
                for i in active:
                    pipes[i].send(("step", actions[i]))
                step += 1
                remaining = []
                for i in active:
                    states[i] = receive(pipes[i])
                    s = states[i]
                    if s["terminated"] or s["truncated"] or (a.smoke and step >= 3):
                        row = dict(
                            id=group[i]["id"],
                            case_kind=group[i]["case_kind"],
                            steps=step,
                            success=bool(s["success"]) if not a.smoke else None,
                            scored=not a.smoke,
                        )
                        if row["id"] not in completed:
                            results.append(row)
                            completed.add(row["id"])
                            with (a.output / "episodes.jsonl").open("a") as f:
                                f.write(json.dumps(row) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                    else:
                        remaining.append(i)
                active = remaining
            write(
                a.output / "progress.json",
                dict(
                    completed=len(results),
                    total=len(todo),
                    seconds=time.perf_counter() - start,
                ),
            )
        write(
            a.output / "result.json",
            dict(
                status="smoke_passed" if a.smoke else "complete",
                episodes=len(results),
                seconds=time.perf_counter() - start,
                success_rate=(
                    None
                    if a.smoke
                    else sum(r["success"] for r in results) / len(results)
                ),
            ),
        )
    finally:
        for pipe in pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == "__main__":
    main()
