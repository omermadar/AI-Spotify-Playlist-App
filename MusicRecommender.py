import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import os, joblib
import config


class MusicRecommender:
    def __init__(self, features_data, songs_data, n_clusters: int = 18, batch_size: int = 4096, random_state: int = 42,
                 language_weight: float = 0.2, artist_weight: float = 0.1):
        """
        Initialize recommender with clustering, then use cosine similarity within clusters.
        """
        self.features = features_data
        self.songs = songs_data
        self.language_weight = language_weight
        self.artist_weight = artist_weight

        # Check for existing saved model
        model_path = f'models/kmeans_k{n_clusters}.joblib'
        os.makedirs('models', exist_ok=True)

        if os.path.exists(model_path):
            print(f"Loading existing KMeans model from {model_path}")
            self.kmeans = joblib.load(model_path)
            self.cluster_labels = self.kmeans.predict(self.features.values)  # Use .values to remove feature names
        else:
            print(f"Training new KMeans model (k={n_clusters})...")
            self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
            self.cluster_labels = self.kmeans.fit_predict(self.features)

            # Save the trained model
            joblib.dump(self.kmeans, model_path)
            print(f"Saved KMeans model to {model_path}")

        print(f"Recommender initialized with {len(self.features)} tracks and {n_clusters} clusters")

    def get_recommendations(self, song_title: str, n_recommendations: int = 10):
        """Get song recommendations based on similarity within the same cluster + language/artist filtering."""
        try:
            # Find the song in the dataset
            song_matches = self.songs[self.songs['name'].str.contains(song_title, case=False, na=False, regex=False)]

            if song_matches.empty:
                print(f"Song '{song_title}' not found.")
                return pd.DataFrame()

            # Use the first match and cast to int
            query_idx = int(song_matches.index[0])
            query_cluster = self.cluster_labels[query_idx]

            # Get query song info for language/artist filtering
            query_song = self.songs.iloc[query_idx]
            query_language = query_song.get('language', 'unknown')
            query_artists = str(query_song.get('artists', '')).lower()

            # FIX: Use .iloc to get row at index, then .values to get numpy array
            query_features = self.features.iloc[query_idx].values.reshape(1, -1)

            # Get all songs in the same cluster
            cluster_indices = np.where(self.cluster_labels == query_cluster)[0]
            cluster_features = self.features.iloc[cluster_indices].values  # FIX: Use .iloc and .values

            # Compute cosine similarity within the cluster
            audio_similarities = cosine_similarity(query_features, cluster_features).flatten()

            # FIXED: Create a composite score that stays within [0, 1] range
            total_similarities = []

            for i, original_idx in enumerate(cluster_indices):
                if original_idx == query_idx:
                    total_similarities.append(0.0)  # Skip the query song itself
                    continue

                candidate_song = self.songs.iloc[original_idx]

                # Start with the base audio similarity (0-1 range)
                base_similarity = audio_similarities[i]

                # Calculate bonus multipliers instead of additive bonuses
                language_multiplier = 1.0
                artist_multiplier = 1.0

                # Language bonus - prefer same language songs
                if 'language' in candidate_song and self.language_weight > 0:
                    candidate_language = candidate_song.get('language', 'unknown')
                    if candidate_language == query_language and query_language != 'unknown':
                        language_multiplier = 1.0 + self.language_weight

                # Artist similarity bonus - prefer similar artists
                if 'artists' in candidate_song and self.artist_weight > 0:
                    candidate_artists = str(candidate_song.get('artists', '')).lower()
                    if candidate_artists in query_artists or query_artists in candidate_artists:
                        artist_multiplier = 1.0 + self.artist_weight

                # Combine using weighted average to keep in [0, 1] range
                # The bonuses enhance the similarity but don't push it above 1.0
                enhanced_similarity = base_similarity * language_multiplier * artist_multiplier

                # Normalize to ensure we don't exceed 1.0
                # The maximum possible multiplier is (1 + language_weight) * (1 + artist_weight)
                max_multiplier = (1.0 + self.language_weight) * (1.0 + self.artist_weight)
                normalized_similarity = min(1.0, enhanced_similarity / max_multiplier * 1.0)

                total_similarities.append(normalized_similarity)

            total_similarities = np.array(total_similarities)

            # QUALITY FILTER: Only recommend songs with reasonable similarity
            min_similarity_threshold = 0.3  # Adjust this threshold as needed

            # Get top similar songs (excluding the query song itself)
            valid_indices = []
            query_pos_in_cluster = np.where(cluster_indices == query_idx)[0]

            for i, similarity in enumerate(total_similarities):
                original_idx = cluster_indices[i]
                if original_idx != query_idx and similarity >= min_similarity_threshold:
                    valid_indices.append(i)

            if not valid_indices:
                print(f"No similar songs found for '{song_title}' above similarity threshold")
                return pd.DataFrame()

            # Sort by total similarity and take top N
            sorted_indices = sorted(valid_indices, key=lambda i: total_similarities[i], reverse=True)
            top_indices_in_cluster = sorted_indices[:n_recommendations]

            # Map back to original indices
            top_indices = cluster_indices[top_indices_in_cluster]

            # Get the recommended songs
            recommended_songs = self.songs.iloc[top_indices].copy().reset_index(drop=True)
            recommended_songs['similarity_score'] = total_similarities[top_indices_in_cluster]
            recommended_songs['audio_similarity'] = audio_similarities[top_indices_in_cluster]

            return recommended_songs.reset_index(drop=True)

        except Exception as e:
            print(f"Error in get_recommendations for '{song_title}': {e}")
            return pd.DataFrame()

    def get_recommendations_with_language_filter(self, song_title: str, n_recommendations: int = 10,
                                                 preferred_language: str = 'en'):
        """
        Enhanced recommendation method with explicit language filtering.
        This ensures we get songs primarily in the preferred language.
        """
        try:
            # Find the song in the dataset
            song_matches = self.songs[self.songs['name'].str.contains(song_title, case=False, na=False, regex=False)]

            if song_matches.empty:
                print(f"Song '{song_title}' not found.")
                return pd.DataFrame()

            # Use the first match and cast to int
            query_idx = int(song_matches.index[0])
            query_cluster = self.cluster_labels[query_idx]

            # Get query song info
            query_song = self.songs.iloc[query_idx]
            query_features = self.features.iloc[query_idx].values.reshape(1, -1)

            # Get all songs in the same cluster
            cluster_indices = np.where(self.cluster_labels == query_cluster)[0]

            # LANGUAGE PRE-FILTER: First filter by language preference
            language_filtered_indices = []
            for idx in cluster_indices:
                if idx == query_idx:
                    continue
                song = self.songs.iloc[idx]
                song_language = song.get('language', 'unknown')

                # Prefer songs in the target language
                if song_language == preferred_language or song_language == 'unknown':
                    language_filtered_indices.append(idx)

            if not language_filtered_indices:
                print(f"No songs found in language '{preferred_language}' in the same cluster")
                # Fallback to original method without language filter
                return self.get_recommendations(song_title, n_recommendations)

            # Calculate similarities only for language-filtered songs
            filtered_features = self.features.iloc[language_filtered_indices].values
            similarities = cosine_similarity(query_features, filtered_features).flatten()

            # Sort by similarity and take top N
            top_indices_in_filtered = similarities.argsort()[-n_recommendations:][::-1]
            final_indices = [language_filtered_indices[i] for i in top_indices_in_filtered]

            # Get the recommended songs
            recommended_songs = self.songs.iloc[final_indices].copy().reset_index(drop=True)
            recommended_songs['similarity_score'] = similarities[top_indices_in_filtered]
            recommended_songs['audio_similarity'] = similarities[top_indices_in_filtered]

            return recommended_songs.reset_index(drop=True)

        except Exception as e:
            print(f"Error in get_recommendations_with_language_filter for '{song_title}': {e}")
            return pd.DataFrame()

    def find_similar_to_features(self, target_features: dict, n_recommendations: int = 200):
        """
        Find songs similar to given audio features using cosine similarity.
        """
        # Create target vector from features dict
        feature_order = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness',
                         'speechiness', 'tempo', 'loudness']

        target_vector = []
        for col in feature_order:
            if col in target_features:
                value = target_features[col]
                # Normalize tempo and loudness to 0-1 scale like other features
                if col == 'tempo':
                    value = min(1.0, max(0.0, (value - 60) / 180))
                elif col == 'loudness':
                    value = min(1.0, max(0.0, (value + 60) / 60))
                target_vector.append(value)
            else:
                target_vector.append(0.5)  # Default middle value

        # Match the number of features in your data
        n_features = self.features.shape[1]
        while len(target_vector) < n_features:
            target_vector.append(0.5)
        target_vector = target_vector[:n_features]

        target_vector = np.array(target_vector).reshape(1, -1)

        # Compute cosine similarity with all songs
        similarities = cosine_similarity(target_vector, self.features.values).flatten()  # FIX: Use .values

        # Get top similar songs
        top_indices = similarities.argsort()[-n_recommendations:][::-1]

        # Return results
        results = self.songs.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        results['audio_similarity'] = similarities[top_indices]
        results['language_score'] = 0.0
        results['artist_score'] = 0.0
        results['total_similarity'] = similarities[top_indices]

        return results.reset_index(drop=True)