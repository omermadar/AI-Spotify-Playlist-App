import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Load the data
df = pd.read_csv('tracks.csv')

# Drop rows with missing values in the 'name' column
# There were 71 missing values in the 'name' column before dropping rows
df.dropna(subset=['name'], inplace=True)

# Select the features you need for your model
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'valence']
X = df[features]

# Keep the song names and artists for later
songs = df[['name', 'artists']]

### Scaling

# Create an instance of the scaler
scaler = MinMaxScaler()

# Scale the features
# The fit_transform method fits the scaler to your data and then transforms it
scaled_features = scaler.fit_transform(X)

# Convert the scaled features back into a DataFrame
X_scaled = pd.DataFrame(scaled_features, columns=features)

### Visualize the data

# Create a figure with subplots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))

# Use a for loop to plot a histogram for each feature
for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    sns.histplot(data=X_scaled, x=feature, bins=20, kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {feature.capitalize()}')
    axes[row, col].set_xlabel(feature.capitalize())
    axes[row, col].set_ylabel('Count')

plt.tight_layout()
plt.show()