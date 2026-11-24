# Research recipe provenance

The names below identify the originating experiment files in the research workspace. SHA-256 identifies the exact source inspected when making this portable edition. Recipes in this folder replace host-specific paths, process management and run queues. The published geometry index retains the accepted sample order from the parallel QA builder. `prepare_pro_qa.py` is a release utility that reconstructs training content from those archived decisions.

| Experiment source | Originating file | Source SHA-256 |
|---|---|---|
| LIBERO flags, matched-arm VLM overlay and mixed SFT | `experiments/baseline_command.py` | `31d1284e4737fefb8d0645bfc21b712fd5d2285205278a53db44e016d03148d8` |
| LIBERO flags, matched-arm VLM overlay and mixed SFT | `experiments/libero_vlm_ablation_20261113/run.py` | `0594be817add0dc98b3d73a55bd2cc92808d88731604744094f272dcd66fcb0c` |
| LIBERO flags, matched-arm VLM overlay and mixed SFT | `experiments/libero_vlm_ablation_20261113/extended.py` | `7ce96b4a31f557879803924e40ada48851e068e3cf3fbfba16d720de6d1e11a5` |
| Joint VLA update scope and disk-mapped data | `experiments/libero_full_vla_20261115/run.py` | `607f68c4456ddf4fa06f0dfd34b6afcd60bb5ea31c4a819032fad7df9269e01b` |
| Joint VLA update scope and disk-mapped data | `experiments/smolvla_base_full_libero_20261104/loader.py` | `e778c04c9e5c52791b2faa87a2f7772147c7a7d3351823ea3f649ed3e8f90168` |
| Answer-only collation | `experiments/so101_vlm/smol_sft_data.py` | `c1697c237c4c94acf19379d7238684ba45f7886b079ddd0bdf5040c2f9528873` |
| Pro data, training contract and trial runner | `experiments/libero_pro2_action_20261115/public_data.py` | `6b37edc598545ad72109824a20ab221886add2fd1fc6f01ccf6525ace9413e06` |
| Pro data, training contract and trial runner | `experiments/libero_pro2_action_20261115/run.py` | `99d270c289994facf960efec3ec9a8c3e8aed1449ea6c262335c3885254adeac` |
| Pro data, training contract and trial runner | `experiments/libero_pro2_action_20261115/evaluate.py` | `65c7ba8d47824728e9b08e56ab38e98b97673212ed245b41ff42936c7c8c9d24` |
| Pro geometry and replay gates | `experiments/libero_bbox_relation_cpu_20261111/build.py` | `0c9d42a4893cd248a97db01499349c294f342b5621eb84fe9e4935c5b381c09b` |
| Pro geometry and replay gates | `experiments/libero_bbox_relation_cpu_20261111/pro2_60k_render.py` | `5786eb0b1663dce639a00a267319876e98f59f32c1f942b5f531aa36d4faf3db` |
| Pro geometry and replay gates | `experiments/libero_bbox_relation_cpu_20261111/pro2_raw_60k.py` | `0971fb85bd31c30b048a77a174fb06ce5c4223a41479d91acbbd729d1088cc72` |
| Pro geometry and replay gates | `experiments/libero_bbox_relation_cpu_20261111/pro2_raw_package.py` | `3fe5ed0a7f3820867a977d68d0d5b3864858099a3dbd7885c46c22adcd5ca3c2` |
| SO-101 frames, normalization and training | `experiments/so101_three_vla_20261110/setup.py` | `03ee3802b45aacc0936a440e5684834b398b1935f5e8e11b9c2d3db05381e585` |
| SO-101 frames, normalization and training | `experiments/so101_three_vla_20261110/data.py` | `4b3f483edf3978f8770d8fa025e93f1a078a7f50e598b8685576da82f2f46129` |
| SO-101 frames, normalization and training | `experiments/so101_three_vla_20261110/train.py` | `0ccf2cf8e1b1da5157f7b8cb6cade8d3b8831a7de462abece5f5a84e87a7499c` |
| SO-101 curriculum 30k and direct 7.5k real branches | `outputs/smolvla_real_20261108/train_full.sh` | `ea9644889834aed6a2a9a0596582d500777b10cdcfa3756961a8c5e12008df23` |
| SO-101 curriculum 30k and direct 7.5k real branches | `experiments/so101_validation_adapt_20261110/train.py` | `97001d448f038ab548da82523a86674d76d4e9a2b33b5cc72ca2e8c44b4881bb` |
| Attention instrumentation | `experiments/smol_attention_20261119/run.py` | `c7fc2175151e24830ed65dc7d1938a60a07cd8261c98a7d2ef68a1ee599586a9` |
| Attention instrumentation | `experiments/smol_attention_20261119/frozen.py` | `6492441683d73733f3887c47e62f01335e05f96905737dbbd6a057509ff6cfc8` |

## Upstream dependencies and data

- SmolVLA is Hugging Face's model and training framework. The public policy and instruction VLM are distributed by their owners. Their model cards and terms govern those weights. [SmolVLA policy](https://huggingface.co/lerobot/smolvla_base) · [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- `third_party/lerobot.patch` records local changes to LeRobot's final-batch, resumption and evaluation-video behavior at commit `146d8474932c8b566717eb2738ba459337ab71c4`. The corresponding license is `third_party/LeRobot-LICENSE`. [LeRobot source revision](https://github.com/huggingface/lerobot/tree/146d8474932c8b566717eb2738ba459337ab71c4)
- `recipes/replay_pro_mask.py` uses a modified robosuite readback that widens RGB channels before deriving segmentation IDs under NumPy 2. `third_party/Robosuite-LICENSE` accompanies that implementation. [robosuite](https://github.com/ARISE-Initiative/robosuite)
- LIBERO episode data, the converted dataset attribution, and license information for the two bundled camera frames are in `data/figure-attribution.txt`. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LeRobot-format dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- `data/pro/selected_episodes.json` and `selected_stats.json` identify the chosen episode IDs and normalization statistics for the public video conversion. The geometry index contains project-derived numeric boxes, question selection, source locators and hashes from the original native training set. Acquire source media and simulator meshes from upstream. [LIBERO-Pro2.0](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/tree/e5500c690c1498eca1b7050350654964190cdf70) · [Native HDF5 training set](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Train) · [Video conversion](https://huggingface.co/datasets/phicoltan/libero_pro2)
- `data/real-box-pairs.csv` contains de-identified numeric predictions and reviewed reference boxes used in the reported held-out real-image evaluation. Derived numeric tables and experiment recipes are the project team's work. The maintainers have not selected a public reuse license for their original materials in this package. Upstream licenses apply to the identified third-party components.
