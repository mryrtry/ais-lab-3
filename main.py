import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def linreg_fit(X, y):
    Xb = add_bias(X)
    XtX = Xb.T @ Xb
    XtY = Xb.T @ y
    w = np.linalg.inv(XtX) @ XtY
    return w


def linreg_predict(X, w):
    Xb = add_bias(X)
    return Xb @ w


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    print("Размер:", df.shape)
    print(df.head())
    print("\nСтатистика (числа):")
    print(df.describe())
    print("\nСтатистика (все):")
    print(df.describe(include='all'))
    return df


def plot_histograms(df):
    df.hist(figsize=(12, 8), bins=20)
    plt.tight_layout()
    plt.show()


def fill_missing_values(df):
    """Заполняет пропущенные значения: модой для строк, средним для чисел."""
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].dtype == 'O':
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        else:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled


def encode_categorical_columns(df):
    """Преобразует строковые колонки в числовые."""
    df_encoded = df.copy()
    df_encoded["Extracurricular Activities"] = (
        df_encoded["Extracurricular Activities"]
        .map({"Yes": 1, "No": 0})
        .astype(int)
    )
    return df_encoded


def normalize_features(X_train, X_test):
    """
    Нормализует признаки: (X - mean) / std.
    """
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    return X_train_norm, X_test_norm


def split_data(X, y, train_ratio=0.8, seed=42):
    """Разбивает данные на обучающую и тестовую выборки."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    train_size = int(train_ratio * len(X))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_names=None, model_name="Модель"):
    """
    Обучает модель, делает предсказания и оценивает R².
    """
    w = linreg_fit(X_train, y_train)
    y_pred = linreg_predict(X_test, w)
    r2 = r2_score(y_test, y_pred)
    print(f"\nR2 {model_name}:", r2)
    print(f"Коэффициенты {model_name}:")
    print("  bias:", w[0])

    if feature_names is not None:
        for name, coef in zip(feature_names, w[1:]):
            print(f"  {name}: {coef}")
    return r2


def create_interaction_feature(df, col1, col2, new_col_name):
    """Создает новый признак как произведение двух существующих."""
    df_new = df.copy()
    df_new[new_col_name] = df_new[col1] * df_new[col2]
    return df_new


def prepare_features_and_target(df, target_col):
    """Разделяет датасет на признаки X и целевую переменную y."""
    X_df = df.drop(columns=[target_col])
    y = df[target_col].values.astype(float)
    X = X_df.values.astype(float)
    return X_df, X, y


def main():
    df = load_and_explore_data("data.csv")
    plot_histograms(df)

    df = fill_missing_values(df)
    df = encode_categorical_columns(df)

    target_col = "Performance Index"
    X_df, X, y = prepare_features_and_target(df, target_col)

    X_train_full, X_test_full, y_train, y_test = split_data(X, y)

    X_train_norm, X_test_norm = normalize_features(X_train_full, X_test_full)

    # Обучение и оценка Модели 1: ["Hours Studied", "Previous Scores"]
    m1_cols = ["Hours Studied", "Previous Scores"]
    X_m1_all = X_df[m1_cols].values.astype(float)

    np.random.seed(42)
    idx = np.random.permutation(len(X_m1_all))
    train_size = int(0.8 * len(X_m1_all))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    X_train_m1 = X_m1_all[train_idx]
    X_test_m1 = X_m1_all[test_idx]

    y_train_m1 = y[train_idx]
    y_test_m1 = y[test_idx]

    X_train_m1, X_test_m1 = normalize_features(X_train_m1, X_test_m1)

    r2_1 = train_and_evaluate_model(
        X_train_m1, y_train_m1, X_test_m1, y_test_m1, m1_cols, "Модель 1"
    )


    # Обучение и оценка Модели 2: все признаки
    r2_2 = train_and_evaluate_model(X_train_norm, y_train, X_test_norm, y_test, X_df.columns, "Модель 2")


    # Обучение и оценка Модели 3: все признаки + синтетический
    df_syn = create_interaction_feature(df, "Hours Studied", "Sample Question Papers Practiced", "Study_Papers_Combo")
    X_df_syn, X_syn, y_syn = prepare_features_and_target(df_syn, target_col)
    X_syn_vals = X_df_syn.values.astype(float)
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = split_data(X_syn_vals, y_syn)
    X_train_syn, X_test_syn = normalize_features(X_train_syn, X_test_syn)

    r2_3 = train_and_evaluate_model(X_train_syn, y_train_syn, X_test_syn, y_test_syn, X_df_syn.columns, "Модель 3")


    results = {
        "Модель 1": r2_1,
        "Модель 2": r2_2,
        "Модель 3": r2_3
    }
    print("\nИтог:")
    for name, val in results.items():
        print(f"{name}: {val:.4f}")


if __name__ == "__main__":
    main()