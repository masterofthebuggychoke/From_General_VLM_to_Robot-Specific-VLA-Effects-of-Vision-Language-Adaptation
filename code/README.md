# Grounded SmolVLA experiments

**Author:** Handonghyun

This directory can be copied into its own GitHub repository. Run commands from this directory. The analysis inputs travel with the code. The recipes use the team's LIBERO, Pro2 and SO-101 training and evaluation settings. Model weights, full datasets and real-robot recordings are separate inputs.

## Reproduce the reported analyses on CPU

The first three commands need Python 3.10+ and its standard library. The last two install figure dependencies and regenerate the learning curves and two-camera attention figure. The sensitivity table omits one suite or one task at a time. The box table re-scores 331 SO-101 image-question predictions against reviewed pseudo-labels at intersection-over-union (IoU) thresholds of 0.25, 0.50 and 0.75. These are perception scores on real imagery.

```bash
python data/leave_one_out.py --out sensitivity-recomputed.csv
python analysis/score_boxes.py data/real-box-pairs.csv > real-box-recomputed.csv
python recipes/prepare_pro_qa.py
python -m pip install -c environment-constraints.txt matplotlib Pillow numpy
python analysis/build_study_figures.py --out figures
```

Compare with `data/sensitivity.csv` and `data/real-box-thresholds.csv`. The Pro index checker verifies the split counts, task/episode/pixel isolation across splits, box answer coordinates and directional relation margins. Both camera views in the attention figure have source attribution in `data/figure-attribution.txt`.

## Training and simulation environment

The recorded stack used Linux x86-64, Python 3.12.15, a CUDA 12.8 PyTorch wheel, bfloat16, a working NVIDIA driver, FFmpeg shared libraries for TorchCodec, and EGL for original LIBERO. Pro's OSMesa renderer additionally needs system OSMesa and ImageMagick libraries. The constraints file pins key packages observed in the research environment. The installation commands assume a fresh Python 3.12 environment and a compatible driver.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
mkdir -p deps
git clone https://github.com/huggingface/lerobot.git deps/lerobot
git -C deps/lerobot checkout 146d8474932c8b566717eb2738ba459337ab71c4
git -C deps/lerobot apply ../../third_party/lerobot.patch
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
python -m pip install -c environment-constraints.txt -e 'deps/lerobot[dataset,training,smolvla,libero]' h5py matplotlib
```

Pro video training and native HDF5 QA use different source snapshots. Download only the inputs required for the selected commands. These full dataset downloads are large.

```bash
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 --local-dir assets/smolvlm
hf download lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --local-dir assets/smolvla-base
hf download HuggingFaceVLA/libero --repo-type dataset --revision 86958911c0f959db2bbbdb107eb3e17c5f9c798e --local-dir assets/libero
hf download phicoltan/libero_pro2 --repo-type dataset --revision e527a3276cc33258e9979e6c322b703bc8c0ef62 --local-dir assets/pro-video
hf download zhouxueyang/LIBERO-Pro2.0-Train --repo-type dataset --revision 6c4c5524676815a88880a7025badd17fb724f7f4 --include 'data/*' --local-dir assets/pro-hdf5
```

## VLM initialization and answer-only SFT

The simulation Base condition uses the published SmolVLA VLM weights wherever present in the full SmolVLM2 model. VLA training initializes its action expert afresh. VLM SFT uses two epochs at 1e-5, groups of 32 examples, microbatches of two, and answer token weighting. The mixed LIBERO condition continues with one LIBERO epoch at 5e-6. The standalone Pro training set uses 43,939 QA of 60,000 total. Its validation and test rows are exported separately.

```bash
python recipes/export_base.py --vlm assets/smolvlm --policy assets/smolvla-base --output runs/base-vlm
python recipes/prepare_pro_qa.py --source assets/pro-hdf5 --output runs/pro-qa
python recipes/train_vlm.py --model runs/base-vlm --manifest runs/pro-qa/train.jsonl --output runs/pro-vqa-vlm
```

The Pro index contains the frame IDs, native pixel hashes, accepted boxes, prompt selection and split assignments. The builder reads the recorded instruction and native camera frame, checks its hash, upsizes it from 128 to 512 pixels with bicubic interpolation, and reconstructs the messages in recorded order. It uses recorded labels. The optional independent scene replay below tests how well a new rendering recovers those boxes. Neither the entire simulator archive nor its images are embedded in the index.

The original mixed LIBERO + RoboInter image/annotation releases and trained VQA VLM weights are external inputs. With the exact manifests and images available, use the same trainer with the mixed manifest first and the LIBERO manifest second.

```bash
python recipes/train_vlm.py --model runs/base-vlm --manifest assets/mixed-qa/train.jsonl --output runs/mixed-vlm
python recipes/train_vlm.py --model runs/mixed-vlm --manifest assets/libero-qa/train.jsonl --libero-final --output runs/mixed-libero-vlm
```

A manifest is JSONL with one record per single-image chat, containing `split`, `image_path` (relative to the manifest or absolute), original `image_size`, `messages` and `answer_text`. The collator verifies the tokenized answer span and masks the prompt and padding. `prepare_pro_qa.py` produces an example of this schema.

## VLA training and evaluation

Select an initialization with `--vlm` and an update scope with `--tune`. For original LIBERO, the full recipe runs 12,000 optimizer updates and saves every 4,000. The frozen-VLM optimizer uses the historical 8k decay schedule and its floor through 12k. The original Base and mixed frozen arms extended the saved 8k training state. This portable launcher starts a fresh uninterrupted run with the same schedule. For Pro the target is 90,000 updates, with checkpoints every 30,000. These targets specify schedules independently of which results were measured. Both use an effective batch of 64 (16 × 4). `--print-config` prints the resolved flags without loading a model, dataset or GPU.

```bash
python recipes/train_sim.py --dataset libero --data assets/libero --vlm runs/base-vlm --tune full --output runs/libero-base-full
python recipes/train_sim.py --dataset libero --data assets/libero --vlm runs/mixed-libero-vlm --tune full --output runs/libero-mixed-full
python recipes/train_sim.py --dataset pro --data assets/pro-video --vlm runs/base-vlm --tune action --output runs/pro-base-action
python recipes/train_sim.py --dataset pro --data assets/pro-video --vlm runs/pro-vqa-vlm --tune action --output runs/pro-vqa-action
```

The Pro adapter selects the recorded 18,751 episode IDs (4,115,692 frames), uses statistics from that subset, restores the native camera orientation and converts quaternion orientation to an axis–angle vector for the 8-D state. Its actions have seven dimensions. Keep the VLM directory available wherever saved checkpoints refer to its tokenizer.

Original LIBERO evaluates all four suites, 10 tasks per suite, 50 episodes per task, 2,000 episodes per checkpoint. It uses relative control, fixed initial states, hard resets, one executed action per inference, and 10 parallel instances of a single task.

```bash
hf download lerobot/libero-assets --repo-type dataset --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --local-dir "$HOME/.cache/libero/assets"
python recipes/configure_sim.py --output .cache/libero-config
export LIBERO_CONFIG_PATH="$PWD/.cache/libero-config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python recipes/evaluate_libero.py --checkpoint runs/libero-base-full/checkpoints/012000/pretrained_model --output runs/eval-base-full-12k
```

To inspect attention on frames 20, 70 and 120 of the first original LIBERO shard, supply two actual 8k checkpoints from the same training scope. The capture saves post-softmax mass by group, layer, head, action query and denoising step, plus first-query image attention for rendering. It checks probability conservation and, on frame 20, exact agreement with the uninstrumented attention kernel. The two-camera figure in the project uses frozen-VLM checkpoints.

```bash
python recipes/capture_attention.py --data assets/libero --base "$BASE_8K" --finetuned "$MIXED_8K" --output runs/attention
```

## Pro benchmark and replay diagnostic

Use the pinned Pro source in a separate shell from original LIBERO. Install Git LFS, OSMesa and ImageMagick as system prerequisites. Some vendored perturbation objects are stored as LFS pointers. The actual textures/meshes must be materialized for evaluation. Upstream asset checksums identify the required objects. The historical environment added the Pro simulator and its dependencies to the Python search path alongside LeRobot.

```bash
git clone https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0.git deps/pro
git -C deps/pro checkout e5500c690c1498eca1b7050350654964190cdf70
git -C deps/pro lfs pull
python -m pip install -c environment-constraints.txt gym==0.26.2 scikit-image==0.24.0 Wand==0.6.13
export LIBERO_PRO_ROOT="$PWD/deps/pro"
export PYTHONPATH="$LIBERO_PRO_ROOT/simulator-source:$LIBERO_PRO_ROOT"
python recipes/configure_sim.py --pro "$LIBERO_PRO_ROOT" --output .cache/pro-config
export LIBERO_CONFIG_PATH="$PWD/.cache/pro-config"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
python recipes/evaluate_pro.py --checkpoint runs/pro-base-action/checkpoints/030000/pretrained_model --output runs/eval-pro-base --batch-size 16 --repeats 10
python recipes/replay_pro_mask.py --source assets/pro-hdf5 --simulator "$LIBERO_PRO_ROOT"
```

The evaluator accepts 3, 10 or 30 repeats on 339 official test cases, saves trial-level outcomes and resumes matching runs. The repeat count above gives 3,390 trials. A three-repeat run gives 1,017. `--smoke` is an unscored execution check. The replay diagnostic applies the original frame/region RGB gates and compares fresh mask boxes to the accepted geometry. Under the locally verified OSMesa setup the example S13 frame passed the frame gate but one region exceeded the gate (MAE 6.74 vs ≤5, 95th percentile 25.85 vs ≤25). It returns a failure in this case. Fresh simulator-level reproduction of this label remains unresolved. The geometry index and RGB reconstruction can be checked separately.

## SO-101 real data

The local pick-and-place dataset contains 75 top-and-wrist demonstrations (39,000 frames, 30 Hz). All 75 were used for optimization. State and action have six motor-position channels. The completed branches used different initial policies and parameter update scopes.

| Branch | Real-stage initialization | Real-stage trainable components | Optimizer updates | Effective batch | Seed |
|---|---|---|---:|---:|---:|
| Original | Original VLM branch's external-data 30k policy | Action side, VLM frozen | 30,000 | 16 | 0 |
| Aligned | Aligned VLM branch's external-data 30k policy | Action side, VLM frozen | 30,000 | 16 | 0 |
| Curriculum | Curriculum external policy, then 1,045 extra updates on the former 100-episode validation split | Joint VLM and action side | 30,000 | 16 | 0 |
| Direct | Public `lerobot/smolvla_base` policy | Joint VLM and action side | 7,500 | 64 (16 × 4) | 1,000 |

The original and aligned branches used a 900/100 external train/validation split. The curriculum branch continued from an external 15k policy to the `external100/checkpoints/001045` policy before real adaptation; those 100 episodes became training data. Its real launcher also explicitly enables parameter gradients on the constructed model. All three 30k real branches start a fresh optimizer with real-data normalization and save every 5,000 updates. Their upstream initialization and tuning procedures differ; interpreting any differences requires both factors.

The direct branch runs the native LeRobot dataset loader, maps top/wrist images to camera1/camera2, and stores checkpoints by training-loop step. Its final `030000` directory corresponds to 7,500 Adam updates, confirmed from each of 489 recorded optimizer step counters. The 11/17 slide records the same demonstration count and update count; its rollout image has no checkpoint identifier. Neither the image nor the inference server log supplies a trial denominator.

```bash
python recipes/prepare_so101.py --source "$SO101_DEMOS" --output runs/so101-data
python recipes/train_so101.py --branch original --init "$ORIGINAL_EXTERNAL_30K" --vlm "$ORIGINAL_VLM" --data runs/so101-data --output runs/real-original
python recipes/train_so101.py --branch aligned --init "$ALIGNED_EXTERNAL_30K" --vlm "$ALIGNED_VLM" --data runs/so101-data --output runs/real-aligned
python recipes/train_so101.py --branch curriculum --init "$CURRICULUM_EXTERNAL100_1045" --vlm "$CURRICULUM_VLM_EPOCH4" --data runs/so101-data --output runs/real-curriculum
python recipes/train_so101.py --branch direct --init assets/smolvla-base --data "$SO101_DEMOS" --output runs/real-direct
```

Each command accepts `--print-config` to inspect the overrides on CPU. The first three require the corresponding policy and VLM exports. The direct branch requires the public base policy, access to its VLM tokenizer and the raw local dataset. A physical rollout additionally needs the matching arm, camera geometry, calibration, control setup and a recorded scoring protocol.

The public CPU re-scoring command above reproduces the real-image box metrics from anonymous numeric pairs. The table measures VLM perception. Physical policy success rates require trial records. This release contains neither the original real videos nor a robot control client.

## Source and verification notes

[Provenance and third-party notices](SOURCES.md) map the packaged recipes to the experiment files. Model architecture and simulator design belong to their upstream projects. The project's experiment code covers preparation and provenance of QA, VLM initialization and tuning, matched VLA experiments, native replay checks and attention diagnostics. Portability changes replace machine paths and queue orchestration with explicit CLI inputs.

For this release we compared both LIBERO modes' training options to saved checkpoint configs, regenerated the 75 real episode action/state hashes and normalization tables, and checked all 60,000 reconstructed QA questions/answers against the historical QA. One locally available HDF5 source scene also matched the original resized PNG hash and all four training messages. The two CPU re-scoring tables matched their recorded CSVs. Full archive image extraction, fresh-environment installation, GPU retraining, complete simulator rollouts and hardware deployment were not repeated for packaging.
