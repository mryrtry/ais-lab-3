from data_loader import DataLoader
from preprocessor import Preprocessor
from linear_regressor import LinearRegressor
from evaluator import Evaluator
from visualizer import Visualizer

import numpy as np


def train_test_split(X, y, ratio=0.8):
    n = len(X)
    idx = np.random.permutation(n)
    t = int(n * ratio)
    return X[idx[:t]], X[idx[t:]], y[idx[:t]], y[idx[t:]]


def main():
    print("=== LOADING DATA ===")
    loader = DataLoader("data.csv")
    df = loader.load()
    print(df.head())

    prep = Preprocessor(df)

    print("\n=== PREPROCESSING ===")
    prep.fill_missing_values()
    prep.encode_categorical()
    print("Done.")

    # Extract X/y
    target = "Performance Index"
    X_df, X, y = prep.to_numpy(target)

    results = {}

    # -------------------- Model 1 --------------------
    cols1 = [c for c in ["Hours Studied", "Previous Scores"] if c in X_df.columns]

    if cols1:
        print("\n=== MODEL 1 ===")
        X_m1 = X_df[cols1].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_m1, y)
        X_tr, X_te = prep.normalize_train_test(X_tr, X_te)

        model = LinearRegressor()
        w1 = model.fit(X_tr, y_tr)
        pred = model.predict(X_te, w1)
        r2 = Evaluator.r2_score(y_te, pred)

        print(f"Model 1 R² = {r2:.4f}")
        results["Model 1"] = (r2, cols1, w1)

    # -------------------- Model 2 --------------------
    print("\n=== MODEL 2 ===")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    X_tr, X_te = prep.normalize_train_test(X_tr, X_te)

    model = LinearRegressor()
    w2 = model.fit(X_tr, y_tr)
    pred = model.predict(X_te, w2)
    r2 = Evaluator.r2_score(y_te, pred)

    print(f"Model 2 R² = {r2:.4f}")
    results["Model 2"] = (r2, list(X_df.columns), w2)

    # -------------------- Model 3 (interaction) --------------------
    print("\n=== MODEL 3 ===")
    df3 = prep.df.copy()
    if "Hours Studied" in df3 and "Sample Question Papers Practiced" in df3:
        df3["Interaction"] = df3["Hours Studied"] * df3["Sample Question Papers Practiced"]

        prep3 = Preprocessor(df3)
        prep3.fill_missing_values()
        prep3.encode_categorical()

        X_df3, X3, y3 = prep3.to_numpy(target)

        X_tr, X_te, y_tr, y_te = train_test_split(X3, y3)
        X_tr, X_te = prep3.normalize_train_test(X_tr, X_te)

        w3 = model.fit(X_tr, y_tr)
        pred = model.predict(X_te, w3)
        r2 = Evaluator.r2_score(y_te, pred)

        print(f"Model 3 R² = {r2:.4f}")
        results["Model 3"] = (r2, list(X_df3.columns), w3)

    # -------------------- Summary --------------------
    print("\n=== RESULTS SUMMARY ===")
    for name, (r2, cols, w) in results.items():
        print(f"{name}: R²={r2:.4f}")

    # -------------------- Visualizations --------------------
    viz = Visualizer()

    print("\n=== HISTOGRAMS ===")
    viz.plot_histograms(df)


if __name__ == "__main__":
    main()
