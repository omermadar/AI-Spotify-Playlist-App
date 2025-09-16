import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self, features_data, songs_data, n_clusters: int = 75, batch_size: int = 4096, random_state: int = 42,
                 language_weight: float = 0.2, artist_weight: float = 0.1):
        """
        Initialize recommender with clustering, then use cosine similarity within clusters.

        Parameters
        - features_data: Scaled numeric feature DataFrame or array of shape (n_samples, n_features)
        - songs_data: DataFrame containing at least ['name', 'artists'] columns aligned with features_data
        - n_clusters: Number of clusters to partition the dataset (e.g., 150 for 500k tracks)
        - batch_size: MiniBatchKMeans batch size for scalability
        - random_state: Random seed for reproducibility
        - language_weight: Weight of language bonus in final similarity (default 0.2)
        - artist_weight: Weight of artist bonus in final similarity (default 0.1)

        Final similarity per candidate is computed as:
            total = (1 - language_weight - artist_weight) * audio_cosine
                    + language_weight * language_score
                    + artist_weight * artist_score
        where
            language_score = min(query_conf, cand_conf) if language matches and not 'unknown' else 0
            artist_score   = 1.0 if any artist token matches (case-insensitive) else 0
        """
        self.features_df = features_data if isinstance(features_data, pd.DataFrame) else pd.DataFrame(features_data)
        self.features = self.features_df.values  # ndarray for algorithms
        self.songs = songs_data.reset_index(drop=True)  # ensure aligned indices

        # Store weights (validated to remain in [0,1] and sum <= 1)
        self.language_weight = float(max(0.0, min(1.0, language_weight)))
        self.artist_weight = float(max(0.0, min(1.0, artist_weight)))
        if self.language_weight + self.artist_weight > 0.99:
            # Slightly rescale to avoid negative weight for audio component
            total = self.language_weight + self.artist_weight
            self.language_weight /= total
            self.artist_weight /= total

        # Fit MiniBatchKMeans for scalable clustering
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
        self.cluster_labels = self.kmeans.fit_predict(self.features)

        # Build mapping from cluster id to list of member indices
        self.cluster_to_indices = {}
        for idx, c in enumerate(self.cluster_labels):
            self.cluster_to_indices.setdefault(int(c), []).append(idx)

        print(f"Recommender initialized with MiniBatchKMeans (k={n_clusters}). Cosine similarity is computed within clusters.")

    def _compute_language_scores(self, query_idx: int, candidate_indices: list[int]):
        import numpy as np
        # Fallbacks if columns missing
        if 'language' not in self.songs.columns or 'language_confidence' not in self.songs.columns:
            return np.zeros(len(candidate_indices), dtype=float)
        # Normalize text to lowercase
        q_lang = str(self.songs.at[query_idx, 'language']).strip().lower()
        q_conf = float(self.songs.at[query_idx, 'language_confidence']) if pd.notna(self.songs.at[query_idx, 'language_confidence']) else 0.0
        cand_langs = self.songs.loc[candidate_indices, 'language'].astype(str).str.strip().str.lower()
        cand_conf = self.songs.loc[candidate_indices, 'language_confidence'].fillna(0.0).astype(float).values
        match_mask = (cand_langs.values == q_lang) & (cand_langs.values != 'unknown') & (q_lang != 'unknown')
        # min(query_conf, cand_conf) when matches else 0
        mins = np.minimum(q_conf, cand_conf)
        return mins * match_mask.astype(float)

    def _compute_artist_scores(self, query_idx: int, candidate_indices: list[int]):
        import numpy as np
        # Fallback if column missing
        if 'artists' not in self.songs.columns:
            return np.zeros(len(candidate_indices), dtype=float)
        def split_artists(s: str):
            return {a.strip().lower() for a in str(s or '').split(',') if a.strip()}
        q_set = split_artists(self.songs.at[query_idx, 'artists'])
        scores = []
        for idx in candidate_indices:
            cand_set = split_artists(self.songs.at[idx, 'artists'])
            scores.append(1.0 if q_set & cand_set else 0.0)
        return np.array(scores, dtype=float)

    def get_recommendations(self, song_title, n_recommendations=10):
        """Return top-N recommendations for a given song title, ranked by blended similarity.

        The ranking blends audio cosine similarity with optional language and artist bonuses.
        See class docstring for the formula and semantics.
        """
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

        # Compute bonus terms
        import numpy as np
        lang_scores = self._compute_language_scores(song_index, candidate_indices)
        artist_scores = self._compute_artist_scores(song_index, candidate_indices)
        base_weight = max(0.0, 1.0 - self.language_weight - self.artist_weight)
        total_scores = base_weight * sims + self.language_weight * lang_scores + self.artist_weight * artist_scores

        # Rank candidates by total similarity (descending)
        ranked_pairs = sorted(zip(candidate_indices, total_scores, sims, lang_scores, artist_scores), key=lambda x: x[1], reverse=True)
        top = ranked_pairs[:n_recommendations]
        top_indices = [idx for idx, *_ in top]
        result = self.songs.iloc[top_indices].reset_index(drop=True).copy()
        # Attach scores for transparency
        result['audio_similarity'] = [s for _, _, s, _, _ in top]
        result['language_score'] = [ls for _, _, _, ls, _ in top]
        result['artist_score'] = [ascore for _, _, _, _, ascore in top]
        result['total_similarity'] = [ts for _, ts, _, _, _ in top]
        return result

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