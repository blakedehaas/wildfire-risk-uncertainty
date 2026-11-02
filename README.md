# Wildfire Risk Uncertainty

Research software for studying probabilistic wildfire risk prediction and uncertainty quantification.

The project uses:

- Dillon et al. probabilistic wildfire-risk components for CONUS.
- SILVIS Wildland-Urban Interface data.
- Distributional modeling and ensemble-based uncertainty analysis.

The current research phase is reproducible exploratory data analysis of the raw source datasets. Predictive modeling has not yet begun.

## Development

Python dependencies and commands are managed with `uv`.

Large source, extracted, and processed datasets are intentionally excluded from Git. Source provenance is recorded in [`data/SOURCES.md`](data/SOURCES.md).

## Phase 8 final ensemble

Launch the 25-model sequential final-training experiment with one command:

```sh
uv run wildfire-phase8 launch
```

The command prints an experiment ID. Use `uv run wildfire-phase8 status <id>`,
`pause <id>`, or `resume <id>` to inspect and control it. See the
[Phase 8 training protocol](docs/methodology/checkpoint_08_final_ensemble.md)
for the frozen HPO selections, checkpoint recovery, and provenance details.

Canonical dataset pilot construction and validation are documented in [docs/canonical_dataset_pilot.md](docs/canonical_dataset_pilot.md).
