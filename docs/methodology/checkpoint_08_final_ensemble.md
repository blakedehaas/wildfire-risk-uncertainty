# Phase 8 final ensemble training

The production command is `uv run wildfire-phase8 launch`. It creates one immutable
experiment under `artifacts/training/phase8/phase8-<UTC timestamp>/` and detaches a
supervisor. That supervisor trains all 25 models sequentially: five independently
initialized members for each of the five outer geographic studies. Launch does not
evaluate outer TEST.

The selected Optuna trials are 0, 1, 0, 1, and 2 for outer studies 1–5. These are
the highest frozen Phase 7 objective values among the first five successful trials
in each study. Outer 3 trial 19 is the winner of the separate 20-success pilot and
is preserved in the manifest as a sensitivity result, not used for the primary
ensemble. The runner rechecks the databases and stores their SHA-256 hashes,
selected configurations, objective values, and source/data fingerprints at launch.

Training uses all rows with the frozen outer TRAIN eligibility flag. The mmap
reader selects these rows before materializing features and targets; it never
delivers outer TEST rows to the trainer. Preprocessing is fitted on that complete
outer TRAIN support once per study. Each member uses the frozen selected depth,
width, batch size, AdamW learning rate and weight decay, 100 epochs, BF16
autocast, and the established BP/FLP losses and GradNorm algorithm. Initialization
seeds are `42000 + 1000 * outer + member` for members 0–4. Training order follows
the existing deterministic epoch/tile shuffle with seed 42.

Every epoch ends with an atomic full-state checkpoint (model, AdamW, GradNorm,
its Adam optimizer, RNG, completed epoch and global step), a SHA-256 sidecar,
and an atomic progress record. Earlier checkpoints remain as recovery points.
Resume verifies the frozen manifest and scans checkpoints newest first, using
the latest integrity-valid state. A damaged checkpoint is retained with a
`.corrupt` suffix when its epoch is replayed. Completed models have a separate
marker tied to the final checkpoint hash and are skipped on resume.

Operator commands, where `<id>` is the directory name printed by launch:

```sh
uv run wildfire-phase8 status <id>
uv run wildfire-phase8 pause <id>
uv run wildfire-phase8 resume <id>
```

Pause takes effect at the next epoch boundary. Resume also works after worker,
supervisor, container, or host shutdown once the repository and experiment
directory are available. The controller lock and PID identity block duplicate
active supervisors; a Linux parent-death signal ends a worker whose supervisor
disappears. A model gets at most three signal-terminated worker failures before
the controller stops, with failure details in
`ledger.json` and `worker.stderr.log`. RAM, swap, GPU memory, and worker RSS are
sampled in the ledger; crossing a resource threshold requests an epoch-boundary
pause. Container/host startup automation must invoke `resume` after a reboot.

## Initial launch audit and startup repair

Experiment `phase8-20261102T175040Z` failed before preprocessing or training.
Its worker stderr contains three identical Python startup errors:
`Error while finding module specification for '__main__' (ValueError:
__main__.__spec__ is None)`. The console entry point had launched the supervisor
as an importable module, but the supervisor used its runtime `__name__` to launch
workers. Under `python -m`, that name is `__main__`. The supervisor retried the
same deterministic exit three times. No epoch checkpoint, preprocessing state,
or completed model exists in that experiment.

The repaired controller launches both supervisor and worker by the explicit
importable module path and stops immediately on a positive worker exit code.
Only signal-terminated workers receive bounded retries. The original manifest
hashes the pre-repair source; the provenance check correctly rejects a resume
with repaired code. Preserve that failed directory as audit evidence and create
a new experiment with `uv run wildfire-phase8 launch` after approval.
