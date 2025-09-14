import pandas as pd
from DataCleaning import DataCleaner
from DataVisualizer import DataVisualizer
from MusicRecommender import MusicRecommender


def main():
    """
    Main function to execute the music recommendation system pipeline.
    """
    print("--- Music Recommendation System Pipeline ---")

    # 1. Load and Clean Data
    print("Loading and cleaning dataset...")
    cleaner = DataCleaner()
    df = pd.read_csv('tracks.csv')  # Load raw music data
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    X, songs = cleaner.clean_data(df)  # Remove missing values, extract features

    # 2. Scale Features
    print("Scaling features...")
    X_scaled = cleaner.scale_features(X)  # Normalize feature values 0-1

    # 3. Create Visualizations
    print("Creating visualizations...")
    # visualizer = DataVisualizer()
    # visualizer.plot_feature_distributions(X_scaled)  # Show feature histograms
    # visualizer.plot_correlation_matrix(X_scaled)  # Show feature relationships

    # 4. Build and Run Recommender
    print("Initializing recommender system...")
    recommender = MusicRecommender(features_data=X_scaled, songs_data=songs)  # Create similarity matrix

    # 5. Get a recommendation using the new search functionality
    search_keyword = 'love'
    print(f"\nSearching for songs with '{search_keyword}' in the title...")
    search_results = recommender.search_songs(search_keyword)

    if not search_results.empty:
        print("\nFound these songs:")
        print(search_results)

        # Get recommendations for the first song in the search results
        first_song_title = search_results.iloc[0]['name']
        print(f"\nGetting recommendations for '{first_song_title}'...")
        recommendations = recommender.get_recommendations(first_song_title, n_recommendations=5)

        print("\nRecommended songs:")
        print(recommendations)
    else:
        print(f"\nNo songs found with '{search_keyword}'.")


if __name__ == "__main__":
    main()