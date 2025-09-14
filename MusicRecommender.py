import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self, features_data, songs_data):
        self.features = features_data  # Scaled audio features
        self.songs = songs_data  # Song names and artists
        self.similarity_matrix = cosine_similarity(self.features)  # Calculate song similarities
        print("Recommender initialized.")

    def get_recommendations(self, song_title, n_recommendations=10):
        try:
            # Find song in dataset
            song_index = self.songs[self.songs['name'].str.lower() == song_title.lower()].index[0]
        except IndexError:
            return f"Song '{song_title}' not found in the dataset."

        # Get similarity scores for this song
        similarity_scores = list(enumerate(self.similarity_matrix[song_index]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
        top_indices = [score[0] for score in sorted_scores[1:n_recommendations + 1]]  # Skip the song itself

        return self.songs.iloc[top_indices]  # Return most similar songs


    def search_songs(self, keyword, limit=5):
        """
        Searches for songs that contain a keyword in their title or artist name.
        """
        keyword_lower = keyword.lower()
        search_results = self.songs[
            self.songs['name'].str.lower().str.contains(keyword_lower) |
            self.songs['artists'].str.lower().str.contains(keyword_lower)
            ]
        return search_results.head(limit)