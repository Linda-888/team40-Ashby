# Tissue Fingerprints: Delta code

Code for every step of the plan, 1 through 11 (heatmaps and the scorecard included).

**Status (10/17):** ran end to end on Delta for 200 TCGA lung cancer patients (LUAD + LUSC) with
PathGen-CLIP, QuiltNet and PLIP. The results and the story are in [STORY.md](STORY.md).
CONCH and MUSK are not run yet (Hugging Face access pending).

## What is on Delta (checked 10/17)

- **Data:** the hackathon's shared folder (`/projects/bfqi/shared/datasets`) has **no TCGA whole slides and no
  reports**. It holds PathGen-2k (10 patients, one slide each), NCT-CRC-HE-100K and PathMMU, none of which
  supports patient-level matching. So `tcga_download.py` fetches open-access slides from the GDC and report text
  from TCGA-Reports.
- **Weights:** none in the shared folder. PathGen-CLIP, QuiltNet and PLIP download freely (`setup_env.sh`).
  For CONCH and MUSK, request access at [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch) and
  [xiangjx/musk](https://huggingface.co/xiangjx/musk), log in to Hugging Face on Delta, then run `bash setup_env.sh weights`.
- **Space:** everything lives on the team's NVMe work space (`/work/nvme/bfqi/$USER/tissue-fingerprints`, see `config.sh`);
  200 patients take about 165 GB of slides.
- **GPU budget:** 10 GPU-hours per person (`accounts`). The whole 200-patient run used well under one.

## Getting the code onto Delta

```bash
git clone git@github.com:Linda-888/team40-Ashby.git /work/nvme/bfqi/$USER/tissue-fingerprints/code
```

The repo is private: clone with your own GitHub access. On `lhuang17`'s Delta account there are two
deploy keys for this repo only: `~/.ssh/tf_deploy` (read-only, used for pulls) and `~/.ssh/tf_deploy_rw`
(write, used only by `sync_results.sh` to push `results/`). Remove the write key in GitHub's
Settings → Deploy keys when the hackathon is over.
Log in through Open OnDemand (Clusters → Delta Shell Access).

## Run order

All commands run from the `code/` directory. Create the log folder once with `mkdir -p logs`.

| Plan step | Command | Where it runs |
|---|---|---|
| 1 Environments | `bash setup_env.sh` | **login node** (needs internet, ~20 min) |
| 2a Data | `source config.sh; python3 tcga_download.py --n-patients 200` | **login node**, ~165 GB, 15–40 min |
| 2b Pick patients | `source activate.sh main; python 01_pick_patients.py --projects TCGA-LUAD TCGA-LUSC` | login node or Jupyter |
| 3–5 Tile + embed | `sbatch slurm/pipeline_gpu.sbatch` (what we used; `--export=ALL,MODELS=conch` for one model) | one A100 job, ~10 min for 400 slides |
| 3 Tiling, alternative | `sbatch slurm/tile.sbatch` | CPU account; the CPU queue was full on 10/17, so not used |
| 4–5 Embedding, alternative | `sbatch slurm/embed.sbatch` (MUSK: `--array=1`, needs the `tf-musk` env) | GPU, one array task per model |
| 6–7 Test 1 / 2 | `python 05_tests.py --model conch` | CPU, seconds |
| 8 Scrambling + attack | `python 06_scramble.py --model conch` | CPU, a few minutes |
| 9 Noise / direction removal | `python 07_remove.py --model conch` | CPU, a few minutes |
| 10 Heatmaps | `python 09_heatmaps.py --model conch` | CPU, a minute |
| Test 2, learned | `python 10_linkage.py --model conch` (and `--split site`) | CPU, a few minutes |
| Deck figures | `python 11_figures.py` (after the steps above, all models at once) | CPU, seconds |
| 11 Scorecard | `python 08_scorecard.py` | CPU |
| 05–11 as a batch job | `sbatch --export=ALL,MODELS=musk slurm/analyze.sbatch` (use this for 1024-dim models) | A100 node's 16 cores, not the login node |
| Freeze to GitHub | `bash sync_results.sh` (after 08 and 11) | login node |

For steps 6–10, run the commands once each for every embedded model (`pathgen`, `quilt`, `plip`, and later `conch`, `musk`). They only need numpy and sklearn, so the `tf-main` kernel in Jupyter works fine.

**For a quick first pass:** add `--max-patients 60` to `01_pick_patients.py`. That runs the whole pipeline on a small set; afterwards, rerun 01 without the flag. Steps 02, 03 and 04 skip anything already finished, so a killed job just needs resubmitting.

## Key design choices (be ready for judges' questions)

- **Pairing (step 2).** Slide A is always the tumor slide. When an FFPE diagnostic slide (DX) exists, A is that slide, which means Test 2 compares each report against the slide it describes. Slide B must be a different physical specimen:
  - `dx_vs_frozen` (default): tumor FFPE vs tumor frozen. Most patients have both.
  - `tumor_vs_normal`: tumor vs normal tissue (sample 11).

  TS and BS slides of the same frozen piece count as one specimen and are never paired together (`fingerprints/barcodes.py`).
- **Diagnosis.** Defaults to the TCGA project, so picking LUAD + LUSC gives the subtype classifier free labels. Passing `--clinical clinical.tsv` (exported from GDC) switches to primary_diagnosis instead.
- **Embedding space.** Every model uses its projected image-text shared space (CONCH `proj_contrast=True`, MUSK `with_head=True`, CLIP-style `encode_image` for the others), because Test 2 compares images and text directly.
- **Five models supported, three run.** The plan's three (CONCH, MUSK, PathGen-CLIP), plus QuiltNet-B-32 and PLIP from the hackathon's model list. PathGen-CLIP, QuiltNet and PLIP are ungated and have been run; CONCH and MUSK wait on access.
- **Two conda envs.** MUSK pins torch 2.0.1 / timm 0.9.8 and conflicts with CONCH. `tf-main` holds CONCH, PathGen, OpenSlide and all the analysis code; `tf-musk` holds only MUSK.
- **Fairness controls.** `05_tests.py` reports all four candidate sets: `all`, `site`, `dx`, `dx+site`. The plan's headline was Test 2 `dx+site`; on real data it is at chance, so the headline is now Test 1 (see STORY.md). Each result includes chance, a Wilson 95% CI, and a p-value from simulating random guessing.
- **Scrambling attack (`06_scramble.py`).** Test 1 and Test 2 each get two independent rotations. The script runs two attacks:
  - Plain Procrustes, as in the plan.
  - PCA + Procrustes, an extra I added. A rotation only rotates the principal axes, so after PCA only a few dimensions need aligning. On 200 real patients it performed about the same as plain Procrustes.

  Every attack is scored only on patients the attacker does not know. The script also reports a "hospital reuses the same key" case, which is mathematically identical to no scrambling at all.
- **Test 2 as a learned-alignment attack (`10_linkage.py`).** On 200 real patients, plain cosine between a report and its slide was at chance for PathGen-CLIP, QuiltNet and PLIP. The realistic attacker has public paired data (e.g. TCGA) embedded with the same public model, so they fit a report→slide map (orthogonal Procrustes or ridge) on those patients and re-identify a disjoint set. The test patients are fixed per split, so chance does not change with training size. Every learned method has a **shuffled-pairs control** (pairs shuffled within diagnosis: disease-level alignment survives, identity does not); beating it is the evidence that the map matches people. `--split site` puts training and test patients at different hospitals; `--split project` trains on one cancer type and tests on the other.
- **The attack curve can exceed the no-scrambling line.** Procrustes fitted on known pairs is effectively a learned image-text alignment, so it can beat raw cosine. That isn't a bug, and it's worth a sentence in the deck.
- **Fingerprint removal (`07_remove.py`).** `--removal mean` is the plan's method: take each patient's mean, subtract the diagnosis mean, run PCA. `--removal xcov` is an alternative that targets the directions where a patient's two views agree. It is **cross-fitted**: directions are fit on half the patients and scored on the other half. Utility is 5-fold logistic regression balanced accuracy, with one slide per patient, so folds split cleanly by patient. **Sweet spot:** the highest-utility setting where both Test 1 and Test 2 are ≤ `--max-lift` (default 2) × chance.

## If there are no report texts

The public TCGA-Reports dataset (Kefeli et al., 2024) has OCR text for about 9.5k TCGA pathology reports. Convert it to a CSV with a patient id column and a `text` column, then point `TF_REPORTS_CSV` at it. `04_embed_reports.py` pulls the first `TCGA-XX-XXXX` it finds in the id column. If you shorten reports with GPT-OSS on Lumen, store the result in a separate column and run `--text-col summary --out-name reports_summary`. The analysis scripts then take `--reports-name reports_summary`.

## Output layout

```
$TF_ROOT/data/patients.csv, slides.csv     step 2
$TF_WORK/tiles/<slide>.npz                 step 3: tiles (200×224×224×3) + level-0 coordinates
$TF_ROOT/thumbs/<slide>.jpg (+ .txt downsample)   for heatmaps
$TF_ROOT/emb/<model>/tiles/<slide>.npy     step 4: per-tile embeddings (heatmaps)
$TF_ROOT/emb/<model>/slides.npz, reports.npz
$TF_ROOT/results/<model>/*.json, *.png     steps 6–10 (tests, scramble, removal, linkage[_site], heatmaps)
$TF_ROOT/results/scorecard.md              step 11
$TF_ROOT/results/figures/                  deck figures + key_numbers.md (11_figures.py)
code/results/  (in this repo)              lean copy pushed by sync_results.sh: figures, JSON, summary plots
```

## What has actually run

- **On Delta with real data (200 patients):** `setup_env.sh`, `tcga_download.py`, `01_pick_patients.py`,
  `slurm/pipeline_gpu.sbatch` (tiling + embedding), `05`–`11`, for PathGen-CLIP, QuiltNet and PLIP.
- **Never run on Delta:** `slurm/check_models.sbatch`, `slurm/tile.sbatch`, `slurm/embed.sbatch` (`pipeline_gpu.sbatch`
  replaced them), and anything with CONCH or MUSK.
- **Synthetic only:** `python tests/smoke_test.py` (barcodes, pairing, chunking, tiling on a fake slide, steps 05–11
  on synthetic embeddings). Its numbers are not results.
- **Caveat:** the Test 2 rows of the heatmap summaries are noise, because Test 2 itself is at chance. Use only Test 1 heatmaps.

## Judging criteria (from the hackathon page)

1. Innovation and creativity in using AI/ML → an individual tissue fingerprint across specimens, breaking key scrambling, and a learned cross-modal attack with hospital-level controls
2. Development progress, including effective use of compute and existing literature/tools → Delta batch jobs, five models, prior work as the baseline
3. Clinical impact ("relevance and potential clinical impact") → safe data sharing between hospitals, plus the scorecard as a pre-release check
4. Quality and clarity of the final summary and oral presentation

On 10/21 the presentations are **flash presentations**, probably much shorter than 20 slides. Confirm the time limit with the organizers.
