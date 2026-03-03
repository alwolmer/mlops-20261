from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeCreditClassifier:
    """Train, persist, and load a decision tree credit classifier."""

    def __init__(self, max_depth: int = 3, random_state: int = 42) -> None:
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, random_state=random_state
        )
        self.feature_columns: list[str] = []

    def train(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "DecisionTreeCreditClassifier":
        self.feature_columns = features.columns.tolist()
        self.model.fit(features, target)
        return self

    def persist_model(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "feature_columns": self.feature_columns,
            "model": self.model,
        }
        with destination.open("wb") as file_handle:
            pickle.dump(payload, file_handle)
        return destination

    @classmethod
    def load_model(cls, path: str | Path) -> "DecisionTreeCreditClassifier":
        with Path(path).open("rb") as file_handle:
            payload = pickle.load(file_handle)

        instance = cls(
            max_depth=payload["max_depth"],
            random_state=payload["random_state"],
        )
        instance.feature_columns = payload["feature_columns"]
        instance.model = payload["model"]
        return instance

    def predict(self, features: pd.DataFrame) -> pd.Series:
        aligned_features = self._align_features(features)
        predictions = self.model.predict(aligned_features)
        return pd.Series(
            predictions, index=aligned_features.index, name="predicted_inadimplente"
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        aligned_features = self._align_features(features)
        probabilities = self.model.predict_proba(aligned_features)[:, 1]
        return pd.Series(
            probabilities, index=aligned_features.index, name="score_inadimplencia"
        )

    def run_inference(self, features: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        return pd.concat([predictions, probabilities], axis=1)

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns:
            return features.reindex(columns=self.feature_columns, fill_value=0)
        return features.copy()
