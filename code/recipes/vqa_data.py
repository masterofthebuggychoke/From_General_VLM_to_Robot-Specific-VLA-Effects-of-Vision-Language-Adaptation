"""SmolVLM answer-span supervision with one unsplit 512px image."""

import hashlib
import json
from PIL import Image


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def processor_load(path):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"height": 512, "width": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}
    processor.tokenizer.padding_side = "right"
    return processor


def render(processor, row, prompt=False):
    return processor.apply_chat_template(
        row["messages"][:-1] if prompt else row["messages"],
        tokenize=False,
        add_generation_prompt=prompt,
        processor_kwargs={"do_image_splitting": False},
    )


def text_ids(processor, text):
    # Exactly the pinned processor's single-image expansion. Every real collator
    # invocation below cross-checks these IDs against the actual image processor.
    tokenizer = processor.tokenizer
    expansion = (
        tokenizer.fake_image_token
        + tokenizer.global_image_token
        + tokenizer.image_token * processor.image_seq_len
        + tokenizer.fake_image_token
    )
    assert text.count(tokenizer.image_token) == 1
    text = text.replace(tokenizer.image_token, expansion)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def token_contract(processor, row):
    prefix = text_ids(processor, render(processor, row, prompt=True))
    full = text_ids(processor, render(processor, row))
    if full[: len(prefix)] != prefix:
        raise ValueError("Prompt is not a token prefix of full conversation")
    eos = processor.tokenizer.eos_token_id
    end = max(i for i, token in enumerate(full) if token == eos) + 1
    answer = processor.tokenizer.decode(
        full[len(prefix) : end - 1], skip_special_tokens=False
    ).strip()
    if answer != row["answer_text"] or end <= len(prefix):
        raise ValueError("Answer-only label span is incorrect")
    if len(full) > 1024:
        raise ValueError(
            "Sequence exceeds configured limit; never silently truncate image tokens"
        )
    return {
        "prompt_tokens": len(prefix),
        "label_end": end,
        "sequence_tokens": len(full),
        "token_ids_sha256": hashlib.sha256(canonical(full).encode()).hexdigest(),
    }


class AnswerOnlyCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows):
        import torch

        images = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                if list(image.size) != row["image_size"]:
                    raise ValueError("Original image dimensions changed")
                images.append([image.convert("RGB")])
        batch = self.processor(
            text=[render(self.processor, r) for r in rows],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        if tuple(batch["pixel_values"].shape[1:]) != (1, 3, 512, 512):
            raise ValueError("Unexpected image split or spatial transform")
        if not bool(batch["pixel_attention_mask"].all()):
            raise ValueError("Spatial padding would invalidate the coordinate contract")
        labels = torch.full_like(batch["input_ids"], -100)
        for i, row in enumerate(rows):
            actual = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
            contract = row["token_contract"]
            if (
                hashlib.sha256(canonical(actual).encode()).hexdigest()
                != contract["token_ids_sha256"]
            ):
                raise ValueError(
                    "Actual processor tokens differ from prepared token contract"
                )
            start, end = contract["prompt_tokens"], contract["label_end"]
            labels[i, start:end] = batch["input_ids"][i, start:end]
        image_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.tokenizer.image_token
        )
        assert not bool(((labels != -100) & (batch["input_ids"] == image_id)).any())
        assert bool((labels != -100).any(dim=1).all())
        assert not bool(((labels != -100) & ~batch["attention_mask"].bool()).any())
        batch["labels"] = labels
        return batch
