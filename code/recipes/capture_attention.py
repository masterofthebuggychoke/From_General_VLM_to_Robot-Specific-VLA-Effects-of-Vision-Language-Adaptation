"""Read-only attention diagnostics on recorded frames; no training or rollouts."""

import io, json, types, gc
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies import make_pre_post_processors


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="Base-initialized 8k checkpoint"
    )
    parser.add_argument(
        "--finetuned", type=Path, required=True, help="VQA-initialized 8k checkpoint"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Pinned original LIBERO training snapshot",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    data = args.data
    shard = sorted((data / "data").glob("*/*.parquet"))[0]
    table = pq.ParquetFile(shard).read_row_group(0)
    tasks = pq.read_table(data / "meta/tasks.parquet").to_pandas()
    records = []
    groups = ["camera0", "camera1", "language", "state", "prefix_padding", "action"]
    for arm in ["base", "finetuned"]:
        path = getattr(args, arm)
        cfg = SmolVLAConfig.from_pretrained(path)
        cfg.device = "cuda"
        cfg.use_amp = False
        policy = (
            SmolVLAPolicy.from_pretrained(path, config=cfg).float().to("cuda").eval()
        )
        pre, _ = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        v = policy.model.vlm_with_expert
        ctx = {}
        counts = []
        layout = {}
        captured = []
        maps = []
        assert not cfg.add_image_special_tokens
        original_embed = v.embed_image
        original_prefix = policy.model.embed_prefix
        original_att = v.eager_attention_forward

        def embed(self, img):
            y = original_embed(img)
            counts.append(y.shape[1])
            return y

        def prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
            y = original_prefix(images, img_masks, lang_tokens, lang_masks, state)
            n = sum(counts)
            text = lang_tokens.shape[1]
            total = y[0].shape[1]
            layout.update(
                bounds=[0, counts[0], n, n + text, n + text + 1, total], prefix=total
            )
            return y

        def wrapper(original, mode):
            def call(self, *a, **kw):
                ctx.update(
                    layer=kw.get("layer_idx", a[2] if len(a) > 2 else None), mode=mode
                )
                try:
                    return original(*a, **kw)
                finally:
                    ctx.clear()

            return call

        def attention(self, mask, b, hd, q, k, val):
            if q.shape[1] == cfg.chunk_size and layout:
                kk = k.repeat_interleave(
                    self.num_attention_heads // self.num_key_value_heads, dim=2
                )
                scores = (
                    q.float().transpose(1, 2)
                    @ kk.float().transpose(1, 2).transpose(-2, -1)
                    * hd**-0.5
                )
                prob = torch.softmax(
                    scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min),
                    -1,
                )
                assert torch.allclose(
                    prob.sum(-1), torch.ones_like(prob.sum(-1)), atol=1e-5
                )
                bounds = layout["bounds"] + [k.shape[1]]
                masses = torch.stack(
                    [
                        prob[0, :, :, a:z].sum(-1)
                        for a, z in zip(bounds[:-1], bounds[1:])
                    ],
                    -1,
                )
                captured.append((ctx["layer"], ctx["mode"], masses.cpu().numpy()))
                maps.append(prob[0, :, 0, : sum(counts)].mean(0).cpu().numpy())
            return original_att(mask, b, hd, q, k, val)

        v.embed_image = types.MethodType(embed, v)
        policy.model.embed_prefix = types.MethodType(prefix, policy.model)
        for name, mode in [
            ("forward_attn_layer", "SA"),
            ("forward_cross_attn_layer", "CA"),
        ]:
            setattr(v, name, types.MethodType(wrapper(getattr(v, name), mode), v))
        v.eager_attention_forward = types.MethodType(attention, v)
        for frame in [20, 70, 120]:
            counts.clear()
            layout.clear()
            captured.clear()
            maps.clear()
            row = table.slice(frame, 1).to_pylist()[0]
            task = (
                str(tasks.index[int(row["task_index"])])
                if isinstance(tasks.index[0], str)
                else str(tasks.iloc[int(row["task_index"])]["task"])
            )
            keys = ["observation.images.image", "observation.images.image2"]
            images = [
                np.array(Image.open(io.BytesIO(row[k]["bytes"])).convert("RGB"))
                for k in keys
            ]
            raw = {
                k: torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 255
                for k, im in zip(keys, images)
            }
            raw.update(
                {
                    "observation.state": torch.tensor(
                        row["observation.state"], dtype=torch.float32
                    ),
                    "task": task,
                }
            )
            batch = pre(raw)
            noise = torch.randn(
                1,
                cfg.chunk_size,
                cfg.max_action_dim,
                generator=torch.Generator().manual_seed(0),
            ).cuda()
            with torch.inference_mode():
                action = policy.predict_action_chunk(batch, noise=noise.clone())
            assert len(captured) == cfg.num_steps * cfg.num_vlm_layers, (
                len(captured),
                cfg.num_vlm_layers,
            )
            masses = np.stack([x[2] for x in captured])
            assert np.allclose(masses.sum(-1), 1, atol=1e-5)
            np.savez_compressed(
                OUT / f"{arm}_{frame}.npz",
                mass=masses,
                image_first_query=np.stack(maps),
                actions=action.cpu().numpy(),
            )
            if frame == 20:
                v.eager_attention_forward = original_att
                with torch.inference_mode():
                    reference = policy.predict_action_chunk(batch, noise=noise.clone())
                assert torch.equal(reference, action), "Capture changed actions"
                v.eager_attention_forward = types.MethodType(attention, v)
            layer_records = []
            for layer in range(cfg.num_vlm_layers):
                ix = [i for i, x in enumerate(captured) if x[0] == layer]
                m = masses[ix].mean(axis=(0, 1, 2))
                layer_records.append(
                    {
                        "layer": layer,
                        "mode": captured[ix[0]][1],
                        **dict(zip(groups, m.tolist())),
                    }
                )
            records.append(
                {
                    "arm": arm,
                    "frame": int(row["frame_index"]),
                    "episode": int(row["episode_index"]),
                    "task": task,
                    "checkpoint": str(path),
                    "layers": layer_records,
                }
            )
            for cam, im in enumerate(images):
                Image.fromarray(im).save(OUT / f"frame{frame}_camera{cam}.png")
            print("DONE", arm, frame, flush=True)
        del policy, v, pre
        gc.collect()
        torch.cuda.empty_cache()
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "groups": groups,
                "records": records,
                "capture_preserves_actions_exactly": True,
                "limitations": "Three recorded frames from one shard, no success labels; attention mass is not causal attribution; current mixed-SA/CA checkpoints, not SA-only ablation.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
