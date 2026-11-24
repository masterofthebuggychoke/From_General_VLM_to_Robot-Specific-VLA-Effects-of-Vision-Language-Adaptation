"""Overlay the released policy's VLM tensors on the full SmolVLM2 language model."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText
    from vqa_data import processor_load

    args.output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.vlm, torch_dtype=torch.float32, local_files_only=True
    )
    prefix = "model.vlm_with_expert.vlm."
    with safe_open(
        args.policy / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        tensors = {
            k[len(prefix) :]: source.get_tensor(k)
            for k in source.keys()
            if k.startswith(prefix)
        }
    loaded = model.load_state_dict(tensors, strict=False)
    if len(tensors) <= 100 or loaded.unexpected_keys:
        raise ValueError("Incompatible policy or VLM export")
    model.save_pretrained(args.output)
    processor_load(args.vlm).save_pretrained(args.output)


if __name__ == "__main__":
    main()
