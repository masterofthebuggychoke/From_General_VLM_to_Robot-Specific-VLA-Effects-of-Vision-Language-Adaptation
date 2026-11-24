# Does visual grounding transfer to robot control?

**Author:** Handonghyun

An English-language research project on SmolVLA, LIBERO, LIBERO-Pro 2.0, and a physical SO-101 robot. Evidence snapshot: **23 November 2026**.

| Deliverable | Entry point |
|---|---|
| GitHub Pages project | [Website](https://masterofthebuggychoke.github.io/From_General_VLM_to_Robot-Specific-VLA-Effects-of-Vision-Language-Adaptation/) |
| Standalone paper | [Download the PDF](site/report.pdf) |
| Independent code repository | [Training, evaluation and analysis recipes](code/README.md) |
| Experimental evidence | [Numeric tables, media and source ledgers](site/evidence/) |
| Editorial record | [Study outline](editorial/OUTLINE.md) and [20 review passes](editorial/REVIEW.md) |

The site includes two videos, a 40-task results explorer, a two-camera attention map, the held-out real-image evaluation, and [Pro 30k trial outcomes](site/evidence/pro-30k-trials.csv). The paper is separately typeset for print and uses source frames from both videos. The project does not depend on an active experiment server.

## Preview the site

From the repository root run the command below, then open [localhost:8000](http://localhost:8000). No build tool, CDN, model download or API key is required to read the site. The two typefaces are self-hosted.

```sh
python -m http.server 8000 --directory site
```

## Publish on GitHub Pages

Extract the GitHub project ZIP and place its **contents** at the root of a GitHub repository. The ZIP contains the hidden `.github/` directory. The workflow watches `main` and publishes only `site/`. Under **Settings → Pages → Build and deployment**, choose **GitHub Actions**, then push to `main` or run the workflow. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

The `code/` directory is self-contained and can be published in a separate repository. Once that repository has a public address, set `codeRepository` in `site/config.js` to its HTTPS GitHub URL. If keeping one repository, use its `tree/main/code` URL. Both code links default to the working local ZIP.

## Data and reproduction

- Download [the standalone code ZIP](site/downloads/code.zip) to reproduce the task-exclusion analysis, box-threshold scoring and scientific figures. The code README gives the environment, data preparation, pinned dependency versions, launch commands, verification performed and remaining limits.
- [Simulation evidence](site/evidence/simulation.json) and [real-world evidence](site/evidence/real-world.json) identify the source records for each measurement. Raw evaluation paths are identifiers from the research workspace. Public tables and numeric inputs are included where redistribution is possible.
- The published record contains team-derived materials and identified third-party components. [Code provenance](code/SOURCES.md) and [media attribution](site/evidence/figure-attribution.txt) give upstream notices. Reuse terms for the original materials have not yet been chosen. Fonts are distributed with their accompanying OFL notices. [Instrument Serif](site/assets/fonts/InstrumentSerif-OFL.txt) · [Inter](site/assets/fonts/Inter-OFL.txt)

## Update results or add video

1. Preserve the original run and add a dated evidence record. For each policy evaluation, record checkpoint, task or condition, initialization rule, success criterion, successful trials and total trials. For Pro, a complete checkpoint evaluation covers 339 cases × 10 repeats. For the physical experiment track demonstration counts separately from scored autonomous trials.
2. Place new MP4 files, stills and posters in `site/assets/`. Web videos use H.264 with yuv420p and native player controls. Record task, operator or policy, recording rate, playback rate, selection rule and source hashes. Use synchronized frame indices for paired views. Add the clip to the relevant study section and matching stills to the paper.
3. Edit `site/index.html` and the independent manuscript under `report/`. Update each numerical table from its source. The 12k explorer stores task-level counts in the `task-values` JSON block, keyed by suite and task ID. Re-run the checks in [the review record](editorial/REVIEW.md).
4. Rebuild the paper with `python editorial/render_report.py`. Rebuild the ZIP for the code with `python editorial/package_code.py`. To generate a GitHub upload ZIP from this checkout, run `python editorial/package_release.py`.

## Open measurements

The four physical task conditions are still in collection. The original 75 demonstrations cannot yet be allocated across those targets. A physical rollout was documented, but its checkpoint identity, success criterion and completed trial sheet are pending. The 30k Base policies have 1,936 matched Pro trials in the 17:08 KST snapshot, with three frozen-policy successes and two joint-policy successes. A full 3,390-trial comparison and scored Pro-VQA results are pending. The release includes one fresh Pro mask replay whose object-level alignment exceeded tolerance. These limits remain visible in the study.
