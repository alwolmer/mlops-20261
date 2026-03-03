from __future__ import annotations

import argparse
from pathlib import Path

from lifecycle.src.runner import CreditPipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the credit risk lifecycle pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Clean data, featurize it, and train the classifier.")
    train_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw CSV dataset.",
    )
    train_parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Directory where cleaned data, features, and model artifacts will be stored.",
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference with persisted pipeline artifacts.")
    infer_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw CSV dataset used for inference.",
    )
    infer_parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Directory containing persisted featurizer and model artifacts.",
    )
    infer_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for persisting predictions.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runner = CreditPipelineRunner()

    if args.command == "train":
        artifacts = runner.run_training_pipeline(args.input, args.artifacts)
        for name, path in artifacts.items():
            print(f"{name}: {path}")
        return

    prediction_frame = runner.run_inference_pipeline(
        args.input,
        args.artifacts,
        prediction_path=args.output,
    )
    print(prediction_frame)


if __name__ == "__main__":
    main()
