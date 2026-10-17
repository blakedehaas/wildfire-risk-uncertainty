"""CLI entry point for Checkpoint 6B calibration."""

from pathlib import Path

from wildfire_risk_uncertainty.benchmark.training_throughput import run


def main() -> None:
    run(Path.cwd())


if __name__ == "__main__":
    main()
