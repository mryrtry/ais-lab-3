import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.means = None
        self.stds = None

    def fill_missing_values(self):
        df = self.df.copy()
        for col in df.columns:
            if df[col].dtype == "O":
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        self.df = df

    def encode_categorical(self):
        df = self.df.copy()
        for col in df.columns:
            if df[col].dtype == "O":
                unique = df[col].unique()
                if len(unique) == 2:
                    df[col] = df[col].map({unique[0]: 0, unique[1]: 1}).astype(float)
                else:
                    df = pd.get_dummies(df, columns=[col], prefix=col)
        self.df = df

    def to_numpy(self, target_col):
        df = self.df.copy()
        y = df[target_col].values.astype(float)
        X_df = df.drop(columns=[target_col])
        X = X_df.values.astype(float)
        return X_df, X, y

    def normalize_train_test(self, X_train, X_test):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1
        self.means = mean
        self.stds = std
        return (X_train - mean) / std, (X_test - mean) / std
