from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class CreditDataCleaner:
    """Apply schema cleanup and KNN imputations to the credit dataset."""

    DATE_COLUMNS = ["data_cadastro", "ultima_compra"]
    CATEGORICAL_COLUMNS = ["estado_civil", "canal_aquisicao"]
    NUMERICAL_COLUMNS = [
        "idade",
        "renda_mensal",
        "numero_dependentes",
        "valor_emprestimo",
        "score_credito",
        "tempo_emprego_anos",
    ]
    TARGET_COLUMN = "inadimplente"
    DROPPED_COLUMNS = ["ultima_compra"]

    def load_dataset(self, path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def clean(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cleaned = dataset.copy()

        for column in self.DATE_COLUMNS:
            if column in cleaned.columns:
                cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

        for column in self.CATEGORICAL_COLUMNS:
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].astype("string").str.strip()

        for column in self.NUMERICAL_COLUMNS:
            if column in cleaned.columns:
                cleaned[column] = (
                    cleaned[column]
                    .astype("string")
                    .str.strip()
                    .str.replace('"', "", regex=False)
                    .replace({"?": pd.NA, "": pd.NA})
                )
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        if self.TARGET_COLUMN in cleaned.columns:
            cleaned[self.TARGET_COLUMN] = pd.to_numeric(
                cleaned[self.TARGET_COLUMN],
                errors="coerce",
            ).astype("Int64")

        cleaned = self._knn_impute_column(
            cleaned,
            target_column="idade",
            feature_columns=["estado_civil", "tempo_emprego_anos"],
            integer_output=True,
        )
        cleaned = self._knn_impute_column(
            cleaned,
            target_column="renda_mensal",
            feature_columns=["tempo_emprego_anos"],
            integer_output=False,
        )
        cleaned = self._knn_impute_column(
            cleaned,
            target_column="numero_dependentes",
            feature_columns=["estado_civil", "idade"],
            integer_output=True,
        )

        if "idade" in cleaned.columns:
            cleaned["idade"] = pd.to_numeric(cleaned["idade"], errors="coerce").astype(
                "Int64"
            )
        if "numero_dependentes" in cleaned.columns:
            cleaned["numero_dependentes"] = pd.to_numeric(
                cleaned["numero_dependentes"],
                errors="coerce",
            ).astype("Int64")

        drop_columns = [
            column for column in self.DROPPED_COLUMNS if column in cleaned.columns
        ]
        return cleaned.drop(columns=drop_columns)

    def persist_dataset(self, dataset: pd.DataFrame, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(destination, index=False)
        return destination

    def _knn_impute_column(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        n_neighbors: int = 3,
        integer_output: bool = False,
    ) -> pd.DataFrame:
        if target_column not in dataset.columns:
            return dataset.copy()

        feature_frame = pd.get_dummies(
            dataset[feature_columns],
            drop_first=False,
            dtype=float,
        )
        feature_frame = feature_frame.fillna(feature_frame.mean(numeric_only=True))

        known_mask = dataset[target_column].notna()
        missing_mask = ~known_mask

        if not missing_mask.any():
            return dataset.copy()

        scaler = StandardScaler()
        x_train = scaler.fit_transform(feature_frame.loc[known_mask])
        x_missing = scaler.transform(feature_frame.loc[missing_mask])
        y_train = pd.to_numeric(
            dataset.loc[known_mask, target_column], errors="coerce"
        ).astype(float)

        model = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(y_train)))
        model.fit(x_train, y_train)
        imputed_values = model.predict(x_missing)

        result = dataset.copy()
        result[target_column] = pd.to_numeric(
            result[target_column], errors="coerce"
        ).astype("Float64")

        if integer_output:
            result.loc[missing_mask, target_column] = np.rint(imputed_values)
        else:
            result.loc[missing_mask, target_column] = imputed_values.astype(float)

        return result
