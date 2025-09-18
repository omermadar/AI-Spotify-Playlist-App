import pandas as pd
from DataCleaning import DataCleaner
from MusicRecommender import MusicRecommender
from PlaylistCreator import PlaylistCreator


def main():
    """
    Main function to execute the music recommendation system pipeline.
    """
    print("--- Music Recommendation System Pipeline ---")

    # 1. Load and Clean Data
    print("Loading and cleaning dataset...")
    cleaner = DataCleaner()
    df = pd.read_csv('tracks_added_languages.csv')  # Load raw music data
    X, songs = cleaner.clean_data(df)  # Remove missing values, extract features

    # 2. Scale Features
    print("Scaling features...")
    X_scaled = cleaner.scale_features(X)  # Normalize feature values 0-1

    # 3. Build and Run Recommender
    print("Initializing recommender system...")
    recommender = MusicRecommender(features_data=X_scaled, songs_data=songs, language_weight=0.2, artist_weight=0.1)

    # Save songs with cluster assignments for persistence
    songs_with_clusters = songs.copy()
    songs_with_clusters['cluster_id'] = recommender.cluster_labels
    songs_with_clusters.to_csv('data/songs_with_clusters.csv', index=False)
    print("Saved songs with cluster assignments to data/songs_with_clusters.csv")

    # 4. Smart Playlist Creation Test - NEW APPROACH
    print("\n--- Smart Playlist Creation Test (Ollama Text-to-Features) ---")

    # Initialize playlist creator
    playlist_creator = PlaylistCreator(music_recommender=recommender)

    # Test the NEW approach: Text → Audio Features → Similarity Search
    user_request = "modern pop-rap songs for workout, i like drake, dave and more"

    # Use the new method
    playlist_data = playlist_creator.create_playlist_from_description(user_request, final_playlist_size=50)

    print(f"\nPlaylist '{playlist_data['name']}' created!")
    print(f"Total tracks: {playlist_data['track_count']}")
    print("\nFirst 10 songs:")
    for i, song in enumerate(playlist_data['songs'][:50]):
        print(f"  {i + 1}. {song['name']} by {song['artist']} (score: {song['score']:.3f})")


if __name__ == "__main__":
    main()