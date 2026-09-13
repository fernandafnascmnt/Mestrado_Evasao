"""Command-line entry point: ``python -m dropout_prediction``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config, figures
from .experiment import ExperimentSettings, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dropout_prediction",
        description=(
            "Run the dropout-risk prediction experiment and write every "
            "artefact needed to reproduce the reported results."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/survey_responses.csv"),
        help="Semicolon-separated survey export (default: %(default)s)",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=Path("data/codebook.csv"),
        help="Variable codebook used to label the outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for the artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=config.DEFAULT_REPETITIONS,
        help="Cross-validation repetitions (default: %(default)s)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=config.DEFAULT_CV_FOLDS,
        help="Folds of the inner cross-validation (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel jobs for the grid search; -1 uses every core (default: %(default)s)",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help=(
            "Run a single repetition with collapsed grids to verify the "
            "installation. Results obtained this way must not be reported."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (matplotlib is then not required)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log warnings and errors only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = ExperimentSettings(
        data_path=args.data,
        output_dir=args.output,
        codebook_path=args.codebook if args.codebook.exists() else None,
        repetitions=args.repetitions,
        cv_folds=args.cv_folds,
        n_jobs=args.jobs,
        reduced=args.reduced,
    )
    run(settings)

    if not args.no_figures:
        figures.build_all(settings.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
