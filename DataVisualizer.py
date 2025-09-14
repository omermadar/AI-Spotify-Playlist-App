import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURES


class DataVisualizer:
    def __init__(self):
        self.features = FEATURES

    def plot_feature_distributions(self, X_scaled, save_path=None):
        num_features = len(self.features)

        # Calculate the number of rows needed for a 3-column grid
        # Use integer division and add 1 if there's a remainder
        num_rows = (num_features + 2) // 3

        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for i, feature in enumerate(self.features):
            sns.histplot(data=X_scaled, x=feature, bins=20, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature.capitalize()}')

        # Hide any unused subplots
        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_correlation_matrix(self, X_scaled, save_path=None):
        plt.figure(figsize=(12, 10))
        correlation_matrix = X_scaled.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()