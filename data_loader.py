import pandas as pd


class DataLoader:
    def __init__(self, filepath="data.csv"):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        return self.df
