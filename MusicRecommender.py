import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import os, joblib, re
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

    def get_recommendations(self, seed_song: dict, n_recommendations: int = 10, user_request: str = "", use_trust_score: bool = True):
        """Get song recommendations based on a seed song object containing artist and title."""
        title = seed_song.get('title', '[Unknown Title]')
        try:
            artist = seed_song.get('artist', '')

            if not title or not artist:
                print(f"Seed song is missing title or artist: {seed_song}")
                return pd.DataFrame()

            # --- Artist-Qualified Seed Search ---
            title_matches = self.songs['name'].str.contains(title, case=False, na=False, regex=False)
            artist_matches = self.songs['artists'].str.contains(artist, case=False, na=False, regex=False)
            song_matches = self.songs[title_matches & artist_matches]

            if song_matches.empty:
                print(f"Song '{title}' by '{artist}' not found.")
                return pd.DataFrame()
            
            query_idx = int(song_matches.index[0])
            # --- End of Artist-Qualified Seed Search ---

            query_cluster = self.cluster_labels[query_idx]
            query_song = self.songs.iloc[query_idx]
            query_artists = str(query_song.get('artists', '')).lower()

            query_features = self.features.iloc[query_idx].values.reshape(1, -1)

            cluster_indices = np.where(self.cluster_labels == query_cluster)[0]
            cluster_features = self.features.iloc[cluster_indices].values

            audio_similarities = cosine_similarity(query_features, cluster_features).flatten()

            final_recommendations = []

            for i, original_idx in enumerate(cluster_indices):
                if original_idx == query_idx:
                    continue

                candidate_song = self.songs.iloc[original_idx].copy()
                audio_similarity = audio_similarities[i]
                
                if use_trust_score:
                    trust_score = 1.0
                    song_name_lower = str(candidate_song.get('name', '')).lower()
                    artist_name_lower = str(candidate_song.get('artists', '')).lower()
                    
                    # 1. Junk Filter
                    junk_words = ['remix', 'live', 'version', 'mix', 'workout', 'various artists', 'instrumental']
                    if any(word in song_name_lower for word in junk_words):
                        trust_score -= 0.3
                    if any(word in artist_name_lower for word in junk_words):
                        trust_score -= 0.3

                    # 2. Robust Language Filter (Character-based)
                    non_latin_chars = 0
                    if len(song_name_lower) > 0:
                        for char in song_name_lower:
                            if ord(char) > 127: # Check for characters outside basic ASCII
                                non_latin_chars += 1
                        # Penalize if more than 25% of characters are non-latin
                        if (non_latin_chars / len(song_name_lower)) > 0.25:
                            trust_score -= 0.5

                    # 3. Artist Mismatch Penalty
                    primary_query_artist = re.findall(r"'(.*?)'", query_artists)[0] if query_artists.startswith("[") else query_artists
                    if primary_query_artist not in artist_name_lower:
                        trust_score -= 0.2

                    # 4. Low Language Confidence Penalty
                    if candidate_song.get('language_confidence', 1.0) <= 0.5:
                        trust_score -= 0.2

                    final_score = (trust_score * 0.7) + (audio_similarity * 0.3)
                else:
                    final_score = audio_similarity

                if final_score > 0.5:
                    candidate_song['similarity_score'] = final_score
                    candidate_song['audio_similarity'] = audio_similarity
                    final_recommendations.append(candidate_song)

            if not final_recommendations:
                print(f"No similar songs found for '{title}' that passed the filters.")
                return pd.DataFrame()

            sorted_recs = sorted(final_recommendations, key=lambda x: x['similarity_score'], reverse=True)
            recommended_songs_df = pd.DataFrame(sorted_recs[:n_recommendations])

            return recommended_songs_df.reset_index(drop=True)

        except Exception as e:
            print(f"Error in get_recommendations for '{title}': {e}")
            return pd.DataFrame()

    def search_songs(self, query: str) -> pd.DataFrame:
        """Search for songs in the database by name."""
        if not query or len(query) < 2:
            return pd.DataFrame()
            
        song_matches = self.songs[self.songs['name'].str.contains(query, case=False, na=False, regex=False)]
        return song_matches[['name', 'artists']]

    def get_artist_songs_in_cluster(self, seed_song: dict):
        """Finds all songs by the same artist within the same cluster and calculates their similarity."""
        title = seed_song.get('title', '[Unknown Title]')
        try:
            artist = seed_song.get('artist', '')
            if not title or not artist:
                return pd.DataFrame()

            # Find the seed song
            title_matches = self.songs['name'].str.contains(title, case=False, na=False, regex=False)
            artist_matches = self.songs['artists'].str.contains(artist, case=False, na=False, regex=False)
            song_matches = self.songs[title_matches & artist_matches]

            if song_matches.empty:
                print(f"Song '{title}' by '{artist}' not found.")
                return pd.DataFrame()
            
            query_idx = int(song_matches.index[0])
            query_song = self.songs.iloc[query_idx]
            query_cluster = self.cluster_labels[query_idx]
            query_features = self.features.iloc[query_idx].values.reshape(1, -1)

            # Extract the primary artist
            primary_artist = str(query_song.get('artists', ''))
            if primary_artist.startswith("["):
                primary_artist = re.findall(r"'(.*?)'", primary_artist)[0]

            print(f"Analyzing cluster {query_cluster} for other songs by '{primary_artist}'...")

            # Get all songs in the same cluster
            cluster_indices = np.where(self.cluster_labels == query_cluster)[0]
            
            same_artist_songs = []
            for i in cluster_indices:
                if i == query_idx:
                    continue
                
                row = self.songs.iloc[i]
                if primary_artist in str(row.get('artists', '')):
                    same_artist_songs.append(row)

            if not same_artist_songs:
                return pd.DataFrame()

            same_artist_df = pd.DataFrame(same_artist_songs)
            
            # Calculate cosine similarity
            other_artist_song_features = self.features.loc[same_artist_df.index].values
            similarities = cosine_similarity(query_features, other_artist_song_features).flatten()
            
            same_artist_df['audio_similarity'] = similarities
            
            return same_artist_df[['name', 'artists', 'audio_similarity']].sort_values(by='audio_similarity', ascending=False)

        except Exception as e:
            print(f"Error in get_artist_songs_in_cluster for '{title}': {e}")
            return pd.DataFrame()

    def find_similar_to_features(self, target_features: dict, n_recommendations: int = 200):
        """
        Find songs similar to given audio features using cosine similarity.
        """
        feature_order = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness',
                         'speechiness', 'tempo', 'loudness']

        target_vector = []
        for col in feature_order:
            if col in target_features:
                value = target_features[col]
                if col == 'tempo':
                    value = min(1.0, max(0.0, (value - 60) / 180))
                elif col == 'loudness':
                    value = min(1.0, max(0.0, (value + 60) / 60))
                target_vector.append(value)
            else:
                target_vector.append(0.5)

        n_features = self.features.shape[1]
        while len(target_vector) < n_features:
            target_vector.append(0.5)
        target_vector = target_vector[:n_features]
        target_vector = np.array(target_vector).reshape(1, -1)

        similarities = cosine_similarity(target_vector, self.features.values).flatten()

        top_indices = similarities.argsort()[-n_recommendations:][::-1]

        results = self.songs.iloc[top_indices].copy()

        for i, song in results.iterrows():
            if 'artist' not in song or pd.isna(song['artist']):
                if 'artists' in song and not pd.isna(song['artists']):
                    if isinstance(song['artists'], list):
                        results.at[i, 'artist'] = song['artists'][0] if song['artists'] else 'Unknown Artist'
                    else:
                        results.at[i, 'artist'] = str(song['artists'])
                else:
                    results.at[i, 'artist'] = 'Unknown Artist'

        results['similarity_score'] = similarities[top_indices]
        results['audio_similarity'] = similarities[top_indices]
        results['language_score'] = 0.0
        results['artist_score'] = 0.0
        results['total_similarity'] = similarities[top_indices]

        return results.reset_index(drop=True)
