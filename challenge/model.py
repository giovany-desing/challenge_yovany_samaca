import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
import xgboost as xgb

class DelayModel:

    def __init__(self):
        self._model = None
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def _compute_delay(self, data: pd.DataFrame) -> pd.Series:
        fecha_i = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S')
        fecha_o = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S')
        min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60.0
        delay = np.where(min_diff > 15, 1, 0)
        return pd.Series(delay, index=data.index)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        required_cols = ['OPERA', 'TIPOVUELO', 'MES']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        df = data.copy()
        opera_dummies = pd.get_dummies(df['OPERA'], prefix='OPERA')
        tipovuelo_dummies = pd.get_dummies(df['TIPOVUELO'], prefix='TIPOVUELO')
        mes_dummies = pd.get_dummies(df['MES'], prefix='MES')
        features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)

        for col in self.top_10_features:
            if col not in features.columns:
                features[col] = 0

        features = features[self.top_10_features]

        if target_column is not None:
            if target_column in df.columns:
                target = df[target_column].astype(int)
            else:
                if 'Fecha-I' in df.columns and 'Fecha-O' in df.columns:
                    target = self._compute_delay(df)
                else:
                    raise ValueError(f"Target column '{target_column}' not found and cannot generate from dates")
            target = pd.DataFrame(target, columns=[target_column])
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        target_series = target.iloc[:, 0]
        n_y0 = (target_series == 0).sum()
        n_y1 = (target_series == 1).sum()
        scale = n_y0 / n_y1 if n_y1 > 0 else 1.0

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale
        )
        self._model.fit(features, target_series)

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError("Model not trained. Call fit() first.")
        features = features[self.top_10_features]
        preds = self._model.predict(features)
        return preds.tolist()