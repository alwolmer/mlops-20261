from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSet:
    raw_features: pd.DataFrame
    normalized_features: pd.DataFrame
    target: pd.Series | None


class CreditDataFeaturizer:
    """Build raw and normalized model features from cleaned credit data."""

    CATEGORICAL_COLUMNS = ["estado_civil", "canal_aquisicao"]
    IDENTIFIER_COLUMNS = ["cliente_id", "nome"]
    DATE_COLUMN = "data_cadastro"
    TARGET_COLUMN = "inadimplente"

    def __init__(self, reference_date: pd.Timestamp | None = None) -> None:
        self.reference_date = (reference_date or pd.Timestamp.today()).normalize()
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []

    def fit_transform(self, dataset: pd.DataFrame) -> FeatureSet:
        modelling_frame, target = self._prepare_modelling_frame(dataset)
        raw_features = self._encode_features(modelling_frame)

        self.feature_columns = raw_features.columns.tolist()
        self.scaler = StandardScaler()
        normalized_array = self.scaler.fit_transform(raw_features)
        normalized_features = pd.DataFrame(
            normalized_array,
            columns=self.feature_columns,
            index=raw_features.index,
        )
        return FeatureSet(
            raw_features=raw_features,
            normalized_features=normalized_features,
            target=target,
        )

    def transform(self, dataset: pd.DataFrame) -> FeatureSet:
        if self.scaler is None or not self.feature_columns:
            raise ValueError(
                "Featurizer state is not fitted. Call fit_transform or load_state first."
            )

        modelling_frame, target = self._prepare_modelling_frame(dataset)
        raw_features = self._encode_features(modelling_frame)
        raw_features = raw_features.reindex(columns=self.feature_columns, fill_value=0)

        normalized_array = self.scaler.transform(raw_features)
        normalized_features = pd.DataFrame(
            normalized_array,
            columns=self.feature_columns,
            index=raw_features.index,
        )
        return FeatureSet(
            raw_features=raw_features,
            normalized_features=normalized_features,
            target=target,
        )

    def persist_datasets(
        self,
        feature_set: FeatureSet,
        raw_path: str | Path,
        normalized_path: str | Path,
    ) -> tuple[Path, Path]:
        raw_destination = Path(raw_path)
        normalized_destination = Path(normalized_path)
        raw_destination.parent.mkdir(parents=True, exist_ok=True)
        normalized_destination.parent.mkdir(parents=True, exist_ok=True)

        raw_to_save = feature_set.raw_features.copy()
        normalized_to_save = feature_set.normalized_features.copy()
        if feature_set.target is not None:
            raw_to_save[self.TARGET_COLUMN] = feature_set.target
            normalized_to_save[self.TARGET_COLUMN] = feature_set.target

        raw_to_save.to_csv(raw_destination, index=False)
        normalized_to_save.to_csv(normalized_destination, index=False)
        return raw_destination, normalized_destination

    def persist_state(self, path: str | Path) -> Path:
        if self.scaler is None or not self.feature_columns:
            raise ValueError("Featurizer state is not fitted and cannot be persisted.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference_date": self.reference_date,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
        }
        with destination.open("wb") as file_handle:
            pickle.dump(payload, file_handle)
        return destination

    @classmethod
    def load_state(cls, path: str | Path) -> "CreditDataFeaturizer":
        with Path(path).open("rb") as file_handle:
            payload = pickle.load(file_handle)

        instance = cls(reference_date=payload["reference_date"])
        instance.feature_columns = payload["feature_columns"]
        instance.scaler = payload["scaler"]
        return instance

    def _prepare_modelling_frame(
        self, dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        frame = dataset.copy()
        frame[self.DATE_COLUMN] = pd.to_datetime(
            frame[self.DATE_COLUMN], errors="coerce"
        )
        frame["periodo_como_cliente"] = (
            self.reference_date - frame[self.DATE_COLUMN]
        ).dt.days

        target = None
        if self.TARGET_COLUMN in frame.columns:
            target = frame[self.TARGET_COLUMN].astype(int)
            frame = frame.drop(columns=[self.TARGET_COLUMN])

        columns_to_drop = [
            column
            for column in self.IDENTIFIER_COLUMNS + [self.DATE_COLUMN]
            if column in frame.columns
        ]
        modelling_frame = frame.drop(columns=columns_to_drop)
        return modelling_frame, target

    def _encode_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(
            dataset,
            columns=[
                column
                for column in self.CATEGORICAL_COLUMNS
                if column in dataset.columns
            ],
            drop_first=False,
            dtype=int,
        )
