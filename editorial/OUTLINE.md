# Study outline

The research question is whether visual instruction tuning changes action success when the underlying demonstrations are held fixed. The plan was developed before the report text. The web and print editions use the same evidence and distinct layouts.

| Section | Question | Evidence and media |
|---|---|---|
| Opening | What could a model learn from images before consuming action demonstrations? | September design motivation; a team photograph of the physical arm; paired 12k contrasts |
| Experimental design | Which approach came from SmolVLA, and which variables did the team control? | Public architecture, two real QA records from one frame, common action initialization and fitting scopes |
| LIBERO | When does VQA help, and on which tasks does the effect change? | 18 × 2,000 rollouts, full suite and task breakdown, task-exclusion sensitivity, single-seed limitation |
| Qualitative policy comparison | What does one matched success/failure disagreement look like? | Synchronized 4k Base and LIBERO-only videos at the same reset; the 50-reset task score for context |
| Physical SO-101 | What was observed on the actual arm, and what was scored on real images? | Dual-camera teleoperation footage, a separately documented physical rollout, paired box predictions, four collection targets |
| Attention | Does an action query allocate more weight to action tokens than to images? | Three frames from one episode; group masses across queries, a first-query map, the scope of the paper's Table 6 |
| Pro 2.0 | What has been built and what remains to be evaluated? | Matched 30k policy outcomes, per-suite coverage, corpus and subset counts, QA splits, saved checkpoints, and the replay diagnostic |
| Reproduction | What can a reader verify without the private hardware or models? | Portable code, numeric inputs, upstream credits, training commands and open measurements |

The September 22 presentation establishes the motivation; November 3 documents the early evaluation and hardware setup; November 17 documents the robot tasks and a rollout. Completed scores are drawn from evaluation records. The November 23 update defines collection targets. Physical success rates and a final Pro comparison require additional evidence.
