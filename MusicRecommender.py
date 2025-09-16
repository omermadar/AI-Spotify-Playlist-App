import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self, features_data, songs_data, n_clusters: int = 75, batch_size: int = 4096, random_state: int = 42):
        """
        Initialize recommender with clustering, then use cosine similarity within clusters.

        :param features_data: Scaled numeric feature DataFrame or array of shape (n_samples, n_features)
        :param songs_data: DataFrame containing at least ['name', 'artists'] columns aligned with features_data
        :param n_clusters: Number of clusters to partition the dataset (e.g., 150 for 500k tracks)
        :param batch_size: MiniBatchKMeans batch size for scalability
        :param random_state: Random seed for reproducibility
        """
        self.features_df = features_data if isinstance(features_data, pd.DataFrame) else pd.DataFrame(features_data)
        self.features = self.features_df.values  # ndarray for algorithms
        self.songs = songs_data.reset_index(drop=True)  # ensure aligned indices

        # Fit MiniBatchKMeans for scalable clustering
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
        self.cluster_labels = self.kmeans.fit_predict(self.features)

        # Build mapping from cluster id to list of member indices
        self.cluster_to_indices = {}
        for idx, c in enumerate(self.cluster_labels):
            self.cluster_to_indices.setdefault(int(c), []).append(idx)

        print(f"Recommender initialized with MiniBatchKMeans (k={n_clusters}). Cosine similarity is computed within clusters.")

    def get_recommendations(self, song_title, n_recommendations=10):
        try:
            # Find song in dataset
            song_index = self.songs[self.songs['name'].str.lower() == song_title.lower()].index[0]
        except IndexError:
            return f"Song '{song_title}' not found in the dataset."

        # Identify the song's cluster
        cluster_id = int(self.cluster_labels[song_index])
        cluster_members = self.cluster_to_indices.get(cluster_id, [])

        # Exclude the song itself
        candidate_indices = [i for i in cluster_members if i != song_index]
        if not candidate_indices:
            return pd.DataFrame(columns=self.songs.columns)  # No other songs in the cluster

        # Compute cosine similarity between the song and candidates within the same cluster
        query_vec = self.features[song_index].reshape(1, -1)
        candidate_matrix = self.features[candidate_indices]
        sims = cosine_similarity(query_vec, candidate_matrix).flatten()

        # Rank candidates by similarity (descending)
        ranked = sorted(zip(candidate_indices, sims), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:n_recommendations]]
        return self.songs.iloc[top_indices].reset_index(drop=True)

    def search_songs(self, keyword, limit=5):
        """
        Searches for songs that contain a keyword in their title or artist name.
        """
        keyword_lower = keyword.lower()
        search_results = self.songs[
            self.songs['name'].str.lower().str.contains(keyword_lower, na=False) |
            self.songs['artists'].str.lower().str.contains(keyword_lower, na=False)
        ]
        return search_results.head(limit).reset_index(drop=True)