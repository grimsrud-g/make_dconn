# make-dconn

Builds motion-censored, smoothed dense connectivity matrices (`.dconn.nii`, Pearson r)
from XCP-D `ses-combined` CIFTI outputs with Connectome Workbench, and
optionally z-scores them the way the template-matching pipeline expects.

## What it does

For each subject × condition (a *unit*):

1. **Censor** (`censor.py`), using the motion TSV and the run boundaries TSV:
   - keep frames with `framewise_displacement_filtered` ≤ 0.2 mm;
   - optionally drop frames whose across-grayordinate BOLD SD is a
     median-absolute-deviation outlier (Hermosillo et al. 2024; **off** by
     default to match pypfm);
   - drop surviving stretches shorter than 5 contiguous frames, counted
     within each run, so the end of one run never joins the start of the next.
2. **Smooth** the censored series: `wb_command -cifti-smoothing <in> 2.55 2.55
   COLUMN <out> -left-surface <L midthickness> -right-surface <R midthickness>
   -merged-volume`. This is the same call pypfm uses (MSC convention, sigma in mm),
   plus `-fix-zeros-surface` when there are missing vertices (see below).
3. **Correlate**: `wb_command -cifti-correlation` gives Pearson r, with no Fisher z.
4. **Z-score** (separate stage, run when needed): each of the 9 blocks (left
   cortex, right cortex and subcortex, paired with each other) is z-scored
   over all its elements. This reproduces `temp_match_pfm/code/zscore_dconn`
   (after Hermosillo et al. 2024) with the same maths, using about half the
   memory (float32, in place).

### Missing vertices

XCP-D leaves a fixed set of cortical vertices per subject NaN in every frame
(460–598 per subject, about 1% of cortex, in the inferior medial temporal
lobe). With `smoothing.fill_missing: true` (the default) they are set to 0
and smoothed with `-fix-zeros-surface`, so Workbench ignores them in every
average and fills them from their neighbours. The dconn therefore has no
NaN rows, which ReproTM can't handle. Values at every other vertex are
identical to smoothing without the fill. Without the fill, NaN would also
spread to ~300–550 neighbouring real vertices, as it does in pypfm.

The filled vertices carry **extrapolated, not measured** connectivity. Each
dconn has a `_filledgrayordinates.tsv` listing them (and the count is in the
sidecar), so you can exclude them when comparing methods. The code refuses
to fill if a vertex is NaN in only some frames, if a missing grayordinate is
subcortical, or if the real data contains exact zeros.

Every parameter lives in a config file. No input path is hardcoded.

## Layout

```
configs/base.yaml            shared paths, censoring, smoothing, Slurm resources
configs/task_concat.yaml     task-task (all task runs concatenated)
configs/rest.yaml            task-rest
configs/task_individual.yaml the 8 single tasks
configs/split_half.yaml      task/rest split-1 and split-2
src/make_dconn/              all source code
slurm/array.sbatch           generic array task (called by `make-dconn submit`)
tests/                       pytest suite (synthetic data)
```

## Setup

```bash
ml devel uv/0.10.8
uv sync --frozen --offline           # creates .venv from uv.lock, cache only
uv run --frozen --offline pytest     # unit tests
```

Package versions are pinned to what is already in the group uv cache. Always
pass `--offline` (and `--frozen`) so uv never downloads or changes a version.

```bash
alias mdc='uv run --frozen --offline make-dconn'   # optional shorthand
```

## Usage

```bash
# what exists / what is missing
uv run --frozen --offline make-dconn status configs/rest.yaml

# submit every unit that still needs a dconn, then its z-score (each z-score
# task waits only for its own dconn task)
uv run --frozen --offline make-dconn submit configs/rest.yaml --stage dconn
uv run --frozen --offline make-dconn submit configs/rest.yaml --stage both
uv run --frozen --offline make-dconn submit configs/rest.yaml --stage zscore   # for dconns that already exist
uv run --frozen --offline make-dconn submit configs/split_half.yaml --stage dconn --subjects s03 --tasks rest --dry-run

# run one unit in the current allocation (needs ~70 GB RAM for a dconn)
uv run --frozen --offline make-dconn run configs/task_individual.yaml --subject s03 --task nBack
```

## Reproducibility

- **Code:** `submit` refuses to run with uncommitted changes, and each job
  checks that the code is still at the commit it was submitted with.
- **Frozen runs:** `submit` writes a run directory under
  `<output_dir>/_runs/` holding the fully resolved config, the manifest, the
  code commit and the Slurm logs. Jobs read the frozen config, so editing a
  config after submitting can't change a run.
- **Sidecars:** every output has a JSON sidecar with the git commit,
  environment, Workbench version, input fingerprints (sha256 for the motion,
  boundary and surface files), the exact commands run, censoring counts per
  run, output checks and the resolved config. The sidecar is written last;
  only a sidecar with `status: complete` marks a unit done.
- **Parameter safety:** the output name's `desc` encodes the parameters
  (e.g. `fd0p2c5sm2p55fill`), and a parameter hash in the sidecar catches any
  other change. Outputs made with different parameters are never silently
  reused or overwritten (`--overwrite` is required).
- **Censoring record:** a per-frame `_censoring.tsv` records FD and why each
  frame was kept or dropped.

## Outputs

`<output_dir>/sub-<s>/ses-combined/func/`:

```
sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55fill_stat-pearsoncorrelation_boldmap.dconn.nii
sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55fill_stat-pearsoncorrelation_boldmap.json
sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55fill_censoring.tsv
sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55fill_filledgrayordinates.tsv
sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55fill_stat-zscored_boldmap.dconn.nii
sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55fill_stat-zscored_boldmap.json
```

Each dconn is about 33 GB. The default `output_dir` is on `$GROUP_SCRATCH`,
which is **purged after 90 days without modification**. Copy anything you
need to keep to Oak.
