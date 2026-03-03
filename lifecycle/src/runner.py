from __future__ import annotations

from pathlib import Path

import pandas as pd

from .classifier import DecisionTreeCreditClassifier
from .cleaning import CreditDataCleaner
from .featurization import CreditDataFeaturizer


class CreditPipelineRunner:
    """Orchestrate dataset cleaning, featurization, training, and inference."""

    def __init__(
        self,
        cleaner: CreditDataCleaner | None = None,
        featurizer: CreditDataFeaturizer | None = None,
        classifier: DecisionTreeCreditClassifier | None = None,
    ) -> None:
        self.cleaner = cleaner or CreditDataCleaner()
        self.featurizer = featurizer or CreditDataFeaturizer()
        self.classifier = classifier or DecisionTreeCreditClassifier()

    def run_training_pipeline(
        self, raw_dataset_path: str | Path, artifact_dir: str | Path
    ) -> dict[str, Path]:
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

        raw_dataset = self.cleaner.load_dataset(raw_dataset_path)
        clean_dataset = self.cleaner.clean(raw_dataset)
        clean_dataset_path = self.cleaner.persist_dataset(
            clean_dataset,
            artifact_root / "clean" / "credit_clean.csv",
        )

        feature_set = self.featurizer.fit_transform(clean_dataset)
        raw_features_path, normalized_features_path = self.featurizer.persist_datasets(
            feature_set,
            artifact_root / "features" / "credit_features_raw.csv",
            artifact_root / "features" / "credit_features_normalized.csv",
        )
        featurizer_state_path = self.featurizer.persist_state(
            artifact_root / "features" / "featurizer.pkl",
        )

        if feature_set.target is None:
            raise ValueError(
                "Training dataset must contain the target column 'inadimplente'."
            )

        self.classifier.train(feature_set.raw_features, feature_set.target)
        model_path = self.classifier.persist_model(
            artifact_root / "models" / "decision_tree.pkl"
        )

        return {
            "clean_dataset": clean_dataset_path,
            "raw_features": raw_features_path,
            "normalized_features": normalized_features_path,
            "featurizer_state": featurizer_state_path,
            "model": model_path,
        }

    def run_inference_pipeline(
        self,
        raw_dataset_path: str | Path,
        artifact_dir: str | Path,
        prediction_path: str | Path | None = None,
    ) -> pd.DataFrame:
        artifact_root = Path(artifact_dir)
        model_path = artifact_root / "models" / "decision_tree.pkl"
        featurizer_state_path = artifact_root / "features" / "featurizer.pkl"

        raw_dataset = self.cleaner.load_dataset(raw_dataset_path)
        clean_dataset = self.cleaner.clean(raw_dataset)
        featurizer = CreditDataFeaturizer.load_state(featurizer_state_path)
        classifier = DecisionTreeCreditClassifier.load_model(model_path)
        feature_set = featurizer.transform(clean_dataset)

        predictions = classifier.run_inference(feature_set.raw_features)
        prediction_frame = clean_dataset.copy()
        prediction_frame[predictions.columns] = predictions

        if prediction_path is not None:
            destination = Path(prediction_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            prediction_frame.to_csv(destination, index=False)

        return prediction_frame
