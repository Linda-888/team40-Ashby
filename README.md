# Tissue Fingerprints: Delta code

Code for every step of the plan, 1 through 11 (heatmaps and the scorecard included).

## Do these first (Saturday morning)

The hackathon page says the organizers **pre-loaded the data (curated paired images and reports from TCGA and OpenPath) and the CONCH/MUSK weights** on the cluster. So:

1. **Find the pre-loaded CONCH and MUSK weights** (Slack, the Hackathon 2026 GitHub repo, or the resource outline). Put their paths in `TF_CONCH_CKPT` and `TF_MUSK_CKPT` in `config.sh`; then no Hugging Face access is needed. As a fallback, request access at [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch) and [xiangjx/musk](https://huggingface.co/xiangjx/musk).
2. **Find the pre-loaded TCGA data and check its format.** Are they whole slides (`.svs`) or pre-cut patches? Are TCGA barcodes kept (we need patient, site and sample type)? Is each report linked to a patient? Only the TCGA part is usable: OpenPath has no patient IDs, so it can't be used for re-identification.
3. Run `ls /projects` and `accounts` to confirm the project directory is `/projects/bfqi`. If it isn't, edit `config.sh`.
4. While waiting on 1–2, everything works with **PathGen-CLIP, QuiltNet and PLIP**, which need no access.

## Getting the code onto Delta

```bash
rsync -av ~/tissue-fingerprints/ <netid>@login.delta.ncsa.illinois.edu:/projects/bfqi/tissue-fingerprints/code/
```

On Delta, edit `config.sh`. At minimum set `TF_SLIDES_DIR`.

## Run order

All commands run from the `code/` directory. Create the log folder once with `mkdir -p logs`.

| Plan step | Command | Where it runs |
|---|---|---|
| 1 Environments | `bash setup_env.sh` | **login node** (needs internet, ~20 min) |
| 1 Load-test models | `sbatch slurm/check_models.sbatch` | GPU, 30 min |
| 2 Pick patients | `source activate.sh main; python 01_pick_patients.py --projects TCGA-LUAD TCGA-LUSC` | login node or Jupyter |
| 3 Tiling | `sbatch slurm/tile.sbatch` | CPU account `bfqi-delta-cpu`, 4 × 32 cores, no GPU hours |
| 4–5 Embedding | `sbatch slurm/embed.sbatch` (backups: `--array=3-4`) | GPU, one array task per model |
| 6–7 Test 1 / 2 | `python 05_tests.py --model conch` | CPU, seconds |
| 8 Scrambling + attack | `python 06_scramble.py --model conch` | CPU, a few minutes |
| 9 Noise / direction removal | `python 07_remove.py --model conch` | CPU, a few minutes |
| 10 Heatmaps | `python 09_heatmaps.py --model conch` | CPU, a minute |
| Test 2, learned | `python 10_linkage.py --model conch` (and `--split site`) | CPU, a few minutes |
| 11 Scorecard | `python 08_scorecard.py` | CPU |

For steps 6–10, run the commands once each for `--model` `conch`, `musk` and `pathgen` (plus `quilt` and `plip` if you embedded them). They only need numpy and sklearn, so the `tf-main` kernel in Jupyter works fine.

**For a quick first pass:** add `--max-patients 60` to `01_pick_patients.py`. That runs the whole pipeline on a small set; afterwards, rerun 01 without the flag. Steps 02, 03 and 04 skip anything already finished, so a killed job just needs resubmitting.

## Key design choices (be ready for judges' questions)

- **Pairing (step 2).** Slide A is always the tumor slide. When an FFPE diagnostic slide (DX) exists, A is that slide, which means Test 2 compares each report against the slide it describes. Slide B must be a different physical specimen:
  - `dx_vs_frozen` (default): tumor FFPE vs tumor frozen. Most patients have both.
  - `tumor_vs_normal`: tumor vs normal tissue (sample 11).

  TS and BS slides of the same frozen piece count as one specimen and are never paired together (`fingerprints/barcodes.py`).
- **Diagnosis.** Defaults to the TCGA project, so picking LUAD + LUSC gives the subtype classifier free labels. Passing `--clinical clinical.tsv` (exported from GDC) switches to primary_diagnosis instead.
- **Embedding space.** Every model uses its projected image-text shared space (CONCH `proj_contrast=True`, MUSK `with_head=True`, CLIP-style `encode_image` for the others), because Test 2 compares images and text directly.
- **Five models.** The plan's three (CONCH, MUSK, PathGen-CLIP), plus QuiltNet-B-32 and PLIP from the hackathon's model list. The last two are ungated and make the scorecard broader.
- **Two conda envs.** MUSK pins torch 2.0.1 / timm 0.9.8 and conflicts with CONCH. `tf-main` holds CONCH, PathGen, OpenSlide and all the analysis code; `tf-musk` holds only MUSK.
- **Fairness controls.** `05_tests.py` reports all four candidate sets: `all`, `site`, `dx`, `dx+site`. **The headline number is Test 2 `dx+site`.** Each result includes chance, a Wilson 95% CI, and a p-value from simulating random guessing.
- **Scrambling attack (`06_scramble.py`).** Test 1 and Test 2 each get two independent rotations. The script runs two attacks:
  - Plain Procrustes, as in the plan.
  - PCA + Procrustes, an extra I added. A rotation only rotates the principal axes, so after PCA only a few dimensions need aligning. That should take fewer known pairs, but it only clearly helped on Test 1 with synthetic data. On real data, compare the two curves before picking the stronger one.

  Every attack is scored only on patients the attacker does not know. The script also reports a "hospital reuses the same key" case, which is mathematically identical to no scrambling at all.
- **Test 2 as a learned-alignment attack (`10_linkage.py`).** On 200 real patients, plain cosine between a report and its slide was at chance for PathGen-CLIP and QuiltNet. The realistic attacker has public paired data (e.g. TCGA) embedded with the same public model, so they fit a report→slide map (orthogonal Procrustes or ridge) on those patients and re-identify a disjoint set. The test patients are fixed per split, so chance does not change with training size. Every learned method has a **shuffled-pairs control** (pairs shuffled within diagnosis: disease-level alignment survives, identity does not); beating it is the evidence that the map matches people. `--split site` puts training and test patients at different hospitals; `--split project` trains on one cancer type and tests on the other.
- **The attack curve can exceed the no-scrambling line.** Procrustes fitted on known pairs is effectively a learned image-text alignment, so it can beat raw cosine. That isn't a bug, and it's worth a sentence in the deck.
- **Fingerprint removal (`07_remove.py`).** `--removal mean` is the plan's method: take each patient's mean, subtract the diagnosis mean, run PCA. `--removal xcov` is an alternative that targets the directions where a patient's two views agree. Utility is 5-fold logistic regression balanced accuracy, with one slide per patient, so folds split cleanly by patient. **Sweet spot:** the highest-utility setting where both Test 1 and Test 2 are ≤ `--max-lift` (default 2) × chance.

## If there are no report texts

The public TCGA-Reports dataset (Kefeli et al., 2024) has OCR text for about 9.5k TCGA pathology reports. Convert it to a CSV with a patient id column and a `text` column, then point `TF_REPORTS_CSV` at it. `04_embed_reports.py` pulls the first `TCGA-XX-XXXX` it finds in the id column. If you shorten reports with GPT-OSS on Lumen, store the result in a separate column and run `--text-col summary --out-name reports_summary`. The analysis scripts then take `--reports-name reports_summary`.

## Output layout

```
$TF_ROOT/data/patients.csv, slides.csv     step 2
$TF_WORK/tiles/<slide>.npz                 step 3: tiles (200×224×224×3) + level-0 coordinates
$TF_ROOT/thumbs/<slide>.jpg (+ .txt downsample)   for heatmaps
$TF_ROOT/emb/<model>/tiles/<slide>.npy     step 4: per-tile embeddings (heatmaps)
$TF_ROOT/emb/<model>/slides.npz, reports.npz
$TF_ROOT/results/<model>/*.json, *.png     steps 6–9
$TF_ROOT/results/scorecard.md              step 11
```

## Verified locally / still to verify on Delta

`python tests/smoke_test.py` passes locally. It covers barcode parsing, specimen pairing, text chunking, tiling on a fake slide, steps 05–09 on synthetic embeddings with a planted fingerprint, and heatmaps.

**Real models, verified on CPU:** PathGen-CLIP, QuiltNet and PLIP were run for real through `00_check_models.py`, `03_embed_images.py` (including DataLoader workers) and `04_embed_reports.py`.

**Not yet run (Delta only):** CONCH and MUSK loading (gated or pre-loaded weights), `setup_env.sh`, and the sbatch scripts. CONCH and MUSK calls follow each repo's README; `check_models.sbatch` is the first real test.

## Judging criteria (from the hackathon page)

1. Innovation and creativity in using AI/ML → the cross-modal attack, and breaking key scrambling
2. Development progress, including effective use of compute and existing literature/tools → Delta batch jobs, five models, prior work as the baseline
3. Clinical impact ("relevance and potential clinical impact") → safe data sharing between hospitals, plus the scorecard as a pre-release check
4. Quality and clarity of the final summary and oral presentation

On 10/21 the presentations are **flash presentations**, probably much shorter than 20 slides. Confirm the time limit with the organizers.
