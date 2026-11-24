"""Answer-only SFT using a prepared JSONL manifest and its local images."""

import argparse
import json
import math
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--libero-final",
        action="store_true",
        help="One epoch at 5e-6 from the mixed VLM",
    )
    args = parser.parse_args()
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        get_cosine_schedule_with_warmup,
    )
    from vqa_data import AnswerOnlyCollator, processor_load, token_contract

    random.seed(0)
    torch.manual_seed(0)
    processor = processor_load(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32
    )
    model.requires_grad_(True).train().to("cuda")
    model.model.connector.register_forward_hook(
        lambda mod, inputs, out: out.to(model.get_input_embeddings().weight.dtype)
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    rows = []
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        if row.get("split") != "train":
            raise ValueError("Expected only training records")
        row["image_path"] = str(args.manifest.parent / row["image_path"])
        if "token_contract" not in row:
            row["token_contract"] = token_contract(processor, row)
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    collate = AnswerOnlyCollator(processor)
    epochs = 1 if args.libero_final else 2
    total = math.ceil(len(rows) / 32) * epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-6 if args.libero_final else 1e-5, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, math.ceil(total * 0.05)), total
    )
    for epoch in range(epochs):
        random.Random(epoch).shuffle(rows)
        for start in range(0, len(rows), 32):
            group = rows[start : start + 32]
            tokens = sum(
                r["token_contract"]["label_end"] - r["token_contract"]["prompt_tokens"]
                for r in group
            )
            optimizer.zero_grad(set_to_none=True)
            for off in range(0, len(group), 2):
                batch = collate(group[off : off + 2]).to("cuda")
                weight = int((batch["labels"] != -100).sum()) / tokens
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                (loss * weight).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
    model.config.use_cache = True
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
