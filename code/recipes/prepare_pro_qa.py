"""Reconstruct the accepted Pro QA release from its compact geometry index and native HDF5 frames."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "data/pro/qa_geometry.jsonl.gz"
SYSTEM_MESSAGE = (
    "Answer the robot vision question using the image and task context. "
    "Spatial answers describe the current image, not the desired goal state."
)


def questions(scene):
    objects = scene["objects"]
    for item in scene["qa"]:
        name, box = objects[item[0]]
        if len(item) == 1:
            prompt = f"Give the visible bounding box of the {name} as [xmin,ymin,xmax,ymax] on a 0-1000 scale."
            answer = json.dumps(
                [round(v * 1000 / 512) for v in box], separators=(",", ":")
            )
            yield "object_bbox", prompt, answer
            continue
        other, box_b = objects[item[1]]
        axis = item[2]
        first, second = ("left", "right") if axis == 0 else ("above", "below")
        delta = (box[axis] + box[axis + 2] - box_b[axis] - box_b[axis + 2]) / 2
        if abs(delta) < 51.2:
            raise ValueError("Ambiguous relation at " + scene["scene_id"])
        prompt = (
            f"In image coordinates, is the visible bounding-box center of the {name} "
            f"{first} or {second} of that of the {other}? Answer {first} or {second}."
        )
        yield "relative_position", prompt, second if delta > 0 else first


def read_index():
    with gzip.open(INDEX, "rt", encoding="utf8") as index:
        return [json.loads(line) for line in index]


def check(records):
    counts, kinds = Counter(), Counter()
    splits = {}
    for item in records:
        split = item["split"]
        suite, task, _, _ = item["episode_id"].split("/")
        for key in (f"{suite}/{task}", item["episode_id"], item["native_pixel_sha256"]):
            if splits.setdefault(key, split) != split:
                raise ValueError("Split overlap at " + key)
        for kind, _, _ in questions(item):
            counts[split] += 1
            kinds[kind] += 1
    if counts != {"train": 43939, "val": 11478, "test": 4583} or kinds != {
        "object_bbox": 30597,
        "relative_position": 29403,
    }:
        raise ValueError("Index does not match the historical split")
    print(
        json.dumps(
            {"scenes": len(records), "qa_per_split": counts, "qa_per_type": kinds}
        )
    )


def build(records, source, output):
    from contextlib import ExitStack
    import h5py
    import numpy as np
    from PIL import Image

    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    # Multiple scenes from one condition share the same HDF5. Limit open handles.
    from collections import OrderedDict

    opened = OrderedDict()
    try:
        with ExitStack() as stack:
            streams = {
                k: stack.enter_context(
                    (output / f"{k}.jsonl").open("w", encoding="utf8")
                )
                for k in ("train", "val", "test")
            }
            for scene in records:
                dataset_path = source / scene["hdf5_path"]
                if dataset_path not in opened:
                    if len(opened) == 8:
                        opened.popitem(last=False)[1].close()
                    opened[dataset_path] = h5py.File(dataset_path, "r")
                opened.move_to_end(dataset_path)
                episode = opened[dataset_path][scene["hdf5_group"]]
                rgb = np.asarray(episode["obs/agentview_image"][scene["frame"]])
                if (
                    hashlib.sha256(rgb.tobytes()).hexdigest()
                    != scene["native_pixel_sha256"]
                ):
                    raise ValueError("Native pixel mismatch at " + scene["scene_id"])
                path = Path("images") / (scene["scene_id"] + ".png")
                Image.fromarray(rgb).resize((512, 512), Image.Resampling.BICUBIC).save(
                    output / path
                )
                instruction = json.loads(episode.attrs["source_metadata"])[
                    "instruction"
                ]
                for i, (kind, question, answer) in enumerate(questions(scene)):
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": f"Task instruction: {instruction}\n{question}",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        },
                    ]
                    record = {
                        "id": f"{scene['scene_id']}_{i:02d}",
                        "scene_id": scene["scene_id"],
                        "episode_id": scene["episode_id"],
                        "type": kind,
                        "split": scene["split"],
                        "image_size": [512, 512],
                        "image_path": str(path),
                        "messages": messages,
                        "answer_text": answer,
                    }
                    streams[scene["split"]].write(json.dumps(record) + "\n")
    finally:
        for handle in opened.values():
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        help="Pinned LIBERO-Pro2.0-Train snapshot with data/ HDF5",
    )
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--limit-scenes", type=int, help="Materialize a prefix for a quick check"
    )
    args = p.parse_args()
    if bool(args.source) != bool(args.output):
        p.error("--source and --output must be supplied together")
    if args.limit_scenes is not None and (not args.output or args.limit_scenes < 1):
        p.error("--limit-scenes needs --output and a positive count")
    records = read_index()
    check(records)
    if args.output:
        build(records[: args.limit_scenes], args.source, args.output)


if __name__ == "__main__":
    main()
