import matplotlib.pyplot as plt


class Visualizer:

    def plot_histograms(self, df):
        """Строит гистограммы всех числовых признаков как в исходной версии."""
        df.hist(figsize=(14, 10), bins=20, edgecolor='black')
        plt.suptitle("Гистограммы признаков", fontsize=16)
        plt.tight_layout()
        plt.show()
