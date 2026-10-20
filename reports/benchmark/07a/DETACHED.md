# Detached full-epoch calibration

The launcher is `scripts/benchmark_07a_detached.py`. It stores each run under `artifacts/benchmark/07a_detached/<experiment_id>/` with `config.json`, atomic `status.json`, supervisor stdout/stderr logs, per-fold stdout/stderr logs, and completed `fold_<n>.json` metrics. Artifacts are ignored by Git and remain in the shared workspace for a later session.

```bash
.venv/bin/python scripts/benchmark_07a_detached.py launch
.venv/bin/python scripts/benchmark_07a_detached.py status EXPERIMENT_ID
.venv/bin/python scripts/benchmark_07a_detached.py results EXPERIMENT_ID
.venv/bin/python scripts/benchmark_07a_detached.py stop EXPERIMENT_ID
```

The calibration executes **one production `train_epoch`** for each of the five frozen inner TRAIN folds of outer study 3. It uses the validated mmap cache, fold-specific preprocessing states, 33 inputs, three hidden layers of width 128, BF16, AdamW with learning rate 0.001 and weight decay 0.000316227766, batch size 65,536, and seed 42. There are two concurrent fold workers, each with four PyTorch intra-op threads and partitioned CPU affinity. No validation or outer TEST rows are read. It is a throughput calibration, not a model-selection run.

The launcher returns after its detached supervisor reports a running worker. The supervisor creates its own POSIX session and outlives the launching CLI and Codex session. Worker processes set Linux `PR_SET_PDEATHSIG` so a killed supervisor cannot leave them running. `status` reconciles a disappeared supervisor into a persisted failure record. `stop` sends SIGTERM to the supervisor, which finishes active epochs, reaps workers, and does not start another fold. A resource gate ends the run if sampled GPU memory exceeds the prior 15,500 MiB gate or guest available RAM falls below 2 GiB. Worker failures retain exit codes and stderr paths. The reported throughput should use the production per-fold wall times and exclude fold startup as appropriate; results can be compared with the Phase 7A short-window projections.

Lifecycle verification on 2026-10-20: smoke experiment `07a-20261020T182954Z-16d6adf8` remained running with a supervisor whose parent was PID 1 and whose session ID matched its own PID after the launch CLI exited. Killing that supervisor made the worker disappear; a subsequent `status` call persisted a failed state. Smoke experiment `07a-20261020T183020Z-b206733c` accepted a graceful stop request, reaped its worker with exit code 0, and retained the result artifact. These smoke runs used no training data.
