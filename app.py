import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

from data_loader import DataLoader
from preprocessor import Preprocessor
from linear_regressor import LinearRegressor
from evaluator import Evaluator
from visualizer import Visualizer


class App:
    """Основное GUI-приложение"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Student Performance — Linear Regression Lab")
        self.root.geometry("1100x700")

        # ====== Панель управления ======
        top = ttk.Frame(root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(top, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_explore = ttk.Button(top, text="Explore", command=self.explore, state=tk.DISABLED)
        self.btn_explore.pack(side=tk.LEFT, padx=5)

        self.btn_preprocess = ttk.Button(top, text="Preprocess", command=self.preprocess, state=tk.DISABLED)
        self.btn_preprocess.pack(side=tk.LEFT, padx=5)

        self.btn_train = ttk.Button(top, text="Train & Evaluate", command=self.train, state=tk.DISABLED)
        self.btn_train.pack(side=tk.LEFT, padx=5)

        # ====== Лог ======
        left = ttk.Frame(root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.log = tk.Text(left, width=40)
        self.log.pack(fill=tk.Y)

        # ====== Графики ======
        right = ttk.Frame(root, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.visualizer = Visualizer(right)

        # ====== Состояние ======
        self.data_loader = None
        self.preprocessor = None
        self.df = None

    def log_msg(self, txt):
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)

    # ---------------------------- Load CSV ----------------------------
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        self.data_loader = DataLoader(path)
        self.df = self.data_loader.load()
        self.preprocessor = Preprocessor(self.df)

        self.log_msg(f"Loaded {path}, shape={self.df.shape}")
        self.btn_explore.config(state=tk.NORMAL)
        self.btn_preprocess.config(state=tk.NORMAL)

    # ---------------------------- Explore ----------------------------
    def explore(self):
        df = self.df
        self.log_msg("=== DATA HEAD ===")
        self.log_msg(df.head().to_string())
        self.log_msg("\n=== DESCRIBE ===")
        self.log_msg(df.describe(include="all").to_string())
        self.visualizer.plot_histograms(df)

    # ---------------------------- Preprocess ----------------------------
    def preprocess(self):
        self.log_msg("=== PREPROCESSING ===")
        self.preprocessor.fill_missing_values()
        if "Extracurricular Activities" in self.preprocessor.df.columns:
            self.preprocessor.encode_categorical({"Extracurricular Activities": {"Yes": 1, "No": 0}})
        else:
            self.preprocessor.encode_categorical()
        self.log_msg("Done.")
        self.btn_train.config(state=tk.NORMAL)

    # ---------------------------- Split ----------------------------
    def split(self, X, y, ratio=0.8):
        n = len(X)
        idx = np.random.permutation(n)
        t = int(ratio * n)
        return X[idx[:t]], X[idx[t:]], y[idx[:t]], y[idx[t:]]

    # ---------------------------- Train ----------------------------
    def train(self):
        target = "Performance Index"
        if target not in self.preprocessor.df.columns:
            messagebox.showerror("Ошибка", f"Нет колонки '{target}'")
            return

        X_df, X_all, y_all = self.preprocessor.to_numpy(target)

        results = {}

        # ----- Model 1 -----
        cols1 = [c for c in ["Hours Studied", "Previous Scores"] if c in X_df.columns]

        if cols1:
            X_m1 = X_df[cols1].values
            X_tr, X_te, y_tr, y_te = self.split(X_m1, y_all)
            X_tr, X_te = self.preprocessor.normalize_train_test(X_tr, X_te)
            model = LinearRegressor()
            w1 = model.fit(X_tr, y_tr)
            pred = model.predict(X_te, w1)
            r2 = Evaluator.r2_score(y_te, pred)
            results["Model 1"] = (r2, cols1, w1)
            self.log_msg(f"Model 1 R2={r2:.4f}")

        # ----- Model 2 -----
        X_tr, X_te, y_tr, y_te = self.split(X_all, y_all)
        X_tr, X_te = self.preprocessor.normalize_train_test(X_tr, X_te)
        model = LinearRegressor()
        w2 = model.fit(X_tr, y_tr)
        pred = model.predict(X_te, w2)
        r2 = Evaluator.r2_score(y_te, pred)
        results["Model 2"] = (r2, list(X_df.columns), w2)
        self.log_msg(f"Model 2 R2={r2:.4f}")

        # ----- Model 3 (synthetic feature) -----
        df_syn = self.preprocessor.df.copy()
        if "Hours Studied" in df_syn and "Sample Question Papers Practiced" in df_syn:
            df_syn["Study_Papers_Combo"] = df_syn["Hours Studied"] * df_syn["Sample Question Papers Practiced"]
            prep2 = Preprocessor(df_syn)
            prep2.fill_missing_values()
            prep2.encode_categorical()
            X_df3, X3, y3 = prep2.to_numpy(target)
            X_tr, X_te, y_tr, y_te = self.split(X3, y3)
            X_tr, X_te = prep2.normalize_train_test(X_tr, X_te)
            w3 = model.fit(X_tr, y_tr)
            pred = model.predict(X_te, w3)
            r2 = Evaluator.r2_score(y_te, pred)
            results["Model 3"] = (r2, list(X_df3.columns), w3)
            self.log_msg(f"Model 3 R2={r2:.4f}")

        # ----- Итог -----
        self.log_msg("\n=== RESULTS ===")
        for name, (r2, cols, w) in results.items():
            self.log_msg(f"{name}: R2={r2:.4f}")

        # Показать веса Model 2 по умолчанию
        cols, w = results["Model 2"][1:]
        self.visualizer.show_coefficients(cols, w, "Model 2 Coefficients")
