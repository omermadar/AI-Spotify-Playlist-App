import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import FEATURES


class DataVisualizer:
    def __init__(self, use_full_dataset: bool = True, data_path: str = 'tracks.csv'):
        """
        Visualizer capable of plotting either the provided (possibly sampled) data
        or the full dataset loaded from disk. By default, it uses the full dataset
        to ensure plots reflect all available data.

        :param use_full_dataset: If True, load, clean, and scale the entire dataset from data_path.
        :param data_path: Path to the CSV file with tracks data.
        """
        self.features = FEATURES
        self.use_full_dataset = use_full_dataset
        self.data_path = data_path
        self.X_scaled_full = None

        if self.use_full_dataset:
            try:
                from DataCleaning import DataCleaner
                df = pd.read_csv(self.data_path)
                cleaner = DataCleaner()
                X_full, _ = cleaner.clean_data(df)
                self.X_scaled_full = cleaner.scale_features(X_full)
            except Exception as e:
                # Fallback to using provided X_scaled in plotting calls
                print(f"Warning: Failed to prepare full dataset for visualization ({e}). Using provided data instead.")
                self.use_full_dataset = False

    def _resolve_data(self, X_scaled, max_rows=None):
        """Choose which data to visualize (full vs provided) and optionally downsample rows for speed."""
        data = None
        if self.use_full_dataset and self.X_scaled_full is not None:
            data = self.X_scaled_full
        else:
            data = X_scaled
        # Ensure DataFrame and restrict to known feature columns if present
        if isinstance(data, pd.DataFrame):
            missing = [f for f in self.features if f not in data.columns]
            if missing:
                # If columns missing, just use the intersection
                cols = [f for f in self.features if f in data.columns]
                data = data[cols]
            else:
                data = data[self.features]
        # Optional downsampling purely for plotting performance
        if max_rows is not None and hasattr(data, 'sample'):
            try:
                if len(data) > max_rows:
                    data = data.sample(n=max_rows, random_state=42)
            except Exception:
                pass
        return data

    def plot_feature_distributions(self, X_scaled=None, save_path=None, max_rows=None):
        num_features = len(self.features)
        data = self._resolve_data(X_scaled, max_rows=max_rows)

        # Calculate the number of rows needed for a 3-column grid
        num_rows = (num_features + 2) // 3
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for i, feature in enumerate(self.features):
            if feature in getattr(data, 'columns', []):
                sns.histplot(data=data, x=feature, bins=30, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature.capitalize()}')
            else:
                axes[i].axis('off')

        # Hide any unused subplots
        for i in range(num_features, len(axes)):
            try:
                fig.delaxes(axes[i])
            except Exception:
                axes[i].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_correlation_matrix(self, X_scaled=None, save_path=None, max_rows=None):
        data = self._resolve_data(X_scaled, max_rows=max_rows)
        plt.figure(figsize=(12, 10))
        # Compute correlation on DataFrame only (features columns ensured in _resolve_data)
        correlation_matrix = data.corr() if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=self.features).corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,  # Show correlation values in each cell
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