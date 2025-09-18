import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self, features_data, songs_data, n_clusters: int = 18, batch_size: int = 4096, random_state: int = 42,
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

    def export_cluster_language_artist_distribution(
        self,
        csv_path: str = 'cluster_language_artist_distribution.csv',
        plot: bool = True,
        max_languages: int = 10,
        top_artists_overall: int = 10,
        artist_min_count: int = 5,
        unknown_label: str = 'unknown',
        plot_lang_path: str = 'K-Decision-Pictures/Cluster-Language-Distribution.png',
        plot_artist_path: str = 'K-Decision-Pictures/Cluster-Artist-Heatmap.png',
    ):
        """
        Build a CSV summarizing, for each cluster, the distribution of languages and
        the frequency of artists, and optionally create plots.

        Output CSV columns (tidy format):
          - cluster_id: int
          - type: 'language' or 'artist'
          - label: language name or artist name (lowercased, stripped)
          - count: occurrences in the cluster
          - percent: count / cluster_size
          - cluster_size: total rows in the cluster
          - global_count: (artist rows only) total occurrences of the artist across the whole dataset

        Parameters
        - csv_path: where to save the tidy CSV report
        - plot: whether to generate plots
        - max_languages: how many top languages (overall) to include in the stacked bar plot
        - top_artists_overall: number of top artists overall to include in the heatmap
        - artist_min_count: minimum total count across all clusters to consider an artist in the heatmap
        - unknown_label: normalization label for missing/unknown language
        - plot_lang_path: path to save the language stacked-bar plot
        - plot_artist_path: path to save the artist heatmap plot
        """
        import numpy as np
        import pandas as pd
        import os
        import matplotlib as mpl
        # Force a non-interactive backend to ensure PNGs render in headless environments
        try:
            mpl.use('Agg')
        except Exception:
            pass

        df = self.songs.copy()
        df = df.reset_index(drop=True)
        df['cluster_id'] = pd.Series(self.cluster_labels, index=df.index).astype(int)

        # Normalize language column
        if 'language' in df.columns:
            langs = df['language'].astype(str).str.strip().str.lower()
            # Replace empty or nan-like with unknown
            langs = langs.replace({'': unknown_label})
            langs = langs.fillna(unknown_label)
        else:
            langs = pd.Series([unknown_label] * len(df))
        df['language_norm'] = langs

        # Prepare cluster sizes
        cluster_sizes = df.groupby('cluster_id').size().rename('cluster_size')

        # Language distribution per cluster (counts and percents)
        lang_counts = (
            df.groupby(['cluster_id', 'language_norm']).size().rename('count').reset_index()
        )
        lang_counts = lang_counts.merge(cluster_sizes.reset_index(), on='cluster_id', how='left')
        lang_counts['percent'] = lang_counts['count'] / lang_counts['cluster_size']
        lang_counts['type'] = 'language'
        lang_counts.rename(columns={'language_norm': 'label'}, inplace=True)

        # Artist frequencies: split comma-separated artists and explode
        if 'artists' in df.columns:
            # split -> list of lowercased, stripped names; drop empty
            artists_series = (
                df['artists']
                .fillna('')
                .astype(str)
                .apply(lambda s: [a.strip().lower() for a in s.split(',') if a.strip()])
            )
            df_art = df[['cluster_id']].copy()
            df_art['artist'] = artists_series
            df_art = df_art.explode('artist')
            df_art = df_art[df_art['artist'].notna() & (df_art['artist'] != '')]
            # Per-cluster counts
            artist_counts = (
                df_art.groupby(['cluster_id', 'artist']).size().rename('count').reset_index()
            )
            artist_counts = artist_counts.merge(cluster_sizes.reset_index(), on='cluster_id', how='left')
            artist_counts['percent'] = artist_counts['count'] / artist_counts['cluster_size']
            artist_counts['type'] = 'artist'
            artist_counts.rename(columns={'artist': 'label'}, inplace=True)
            # Global (dataset-wide) artist counts
            total_artist_counts = df_art.groupby('artist').size().rename('global_count').reset_index()
            total_artist_counts.rename(columns={'artist': 'label'}, inplace=True)
            artist_counts = artist_counts.merge(total_artist_counts, on='label', how='left')
        else:
            artist_counts = pd.DataFrame(columns=['cluster_id', 'label', 'count', 'cluster_size', 'percent', 'type', 'global_count'])

        # Combine and save tidy CSV
        tidy = pd.concat([lang_counts, artist_counts], ignore_index=True)
        # Ensure global_count exists for all rows (NaN for non-artist rows)
        if 'global_count' not in tidy.columns:
            tidy['global_count'] = np.nan
        # Sort for readability
        tidy.sort_values(['type', 'cluster_id', 'count'], ascending=[True, True, False], inplace=True)
        tidy.to_csv(csv_path, index=False)
        print(f"Saved cluster language/artist distribution CSV to: {csv_path}")

        if not plot:
            return tidy

        # Plotting: import heavy libs lazily
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except Exception as e:
            print(f"Plotting libraries not available ({e}). Skipping plots.")
            return tidy

        # 1) Stacked bar of language percentages per cluster (top-N languages overall)
        try:
            total_lang_counts = lang_counts.groupby('label')['count'].sum().sort_values(ascending=False)
            top_langs = list(total_lang_counts.head(max_languages).index)
            # Group other languages into 'other'
            lang_plot = lang_counts.copy()
            lang_plot['label_plot'] = lang_plot['label'].where(lang_plot['label'].isin(top_langs), other='other')
            # Re-aggregate after grouping others
            lang_plot = (
                lang_plot.groupby(['cluster_id', 'label_plot'])[['count']].sum().reset_index()
                .merge(cluster_sizes.reset_index(), on='cluster_id', how='left')
            )
            lang_plot['percent'] = lang_plot['count'] / lang_plot['cluster_size']
            pivot = lang_plot.pivot(index='cluster_id', columns='label_plot', values='percent').fillna(0.0)
            # Order columns: top_langs then 'other' if exists
            cols = [c for c in top_langs if c in pivot.columns]
            if 'other' in pivot.columns:
                cols.append('other')
            pivot = pivot[cols]

            plt.figure(figsize=(max(10, len(pivot) * 0.5), 6))
            pivot.plot(kind='bar', stacked=True, colormap='tab20')
            plt.title('Language distribution by cluster (percentage)')
            plt.xlabel('Cluster ID')
            plt.ylabel('Percentage within cluster')
            plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_lang_path) or '.', exist_ok=True)
            plt.savefig(plot_lang_path, dpi=150, bbox_inches='tight')
            print(f"Saved language distribution plot to: {plot_lang_path}")
            plt.close()
        except Exception as e:
            print(f"Failed to create language distribution plot: {e}")

        # 2) Heatmap of artist counts across clusters (top-N artists overall with min-count filter)
        try:
            if not artist_counts.empty and top_artists_overall > 0:
                total_artist_counts = artist_counts.groupby('label')['count'].sum()
                eligible = total_artist_counts[total_artist_counts >= max(1, artist_min_count)]
                top_artists = list(eligible.sort_values(ascending=False).head(top_artists_overall).index)
                if top_artists:
                    aplot = artist_counts[artist_counts['label'].isin(top_artists)].copy()
                    # Rename columns with global counts for readability in the plot
                    label_to_global = total_artist_counts.to_dict()
                    aplot['label_pretty'] = aplot['label'].map(lambda s: f"{s} ({label_to_global.get(s, 0)})")
                    heat = aplot.pivot(index='cluster_id', columns='label_pretty', values='count').fillna(0)
                    # Sort clusters by size for readability
                    heat = heat.loc[cluster_sizes.sort_values(ascending=False).index]
                    plt.figure(figsize=(min(24, 2 + 0.5 * len(top_artists)), max(6, 0.35 * len(heat))))
                    sns.heatmap(heat, cmap='YlOrRd', linewidths=0.2)
                    plt.title('Top artists count by cluster (x-axis shows global totals)')
                    plt.xlabel('Artist (global count)')
                    plt.ylabel('Cluster ID')
                    plt.tight_layout()
                    os.makedirs(os.path.dirname(plot_artist_path) or '.', exist_ok=True)
                    plt.savefig(plot_artist_path, dpi=150, bbox_inches='tight')
                    print(f"Saved artist heatmap plot to: {plot_artist_path}")
                    plt.close()
                else:
                    print("No artists met the min-count criteria; skipping artist heatmap.")
            else:
                print("Artist data not available or disabled; skipping artist heatmap.")
        except Exception as e:
            print(f"Failed to create artist heatmap plot: {e}")

        return tidy