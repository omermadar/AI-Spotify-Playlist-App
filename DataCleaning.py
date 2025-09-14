import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config

class DataCleaner:
    def __init__(self):
        self.scaler = MinMaxScaler()  # Tool to normalize values to 0-1 range
        # Audio features used for recommendations
        self.features = config.FEATURES

    def clean_data(self, df):
        """Remove songs with missing names and extract features"""
        df.dropna(subset=['name'], inplace=True)  # Remove songs without names
        X = df[self.features].copy()  # Get audio feature columns
        songs = df[['name', 'artists']].copy()  # Get song info
        print("Data cleaned successfully.")
        return X, songs

    def scale_features(self, X):
        """Normalize feature values to 0-1 range for fair comparison"""
        scaled_features = self.scaler.fit_transform(X)  # Scale all features
        X_scaled = pd.DataFrame(scaled_features, columns=self.features)  # Convert back to DataFrame
        print("Features scaled successfully.")
        return X_scaled