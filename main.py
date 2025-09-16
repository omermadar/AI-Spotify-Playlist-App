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
    X, songs = cleaner.clean_data(df)  # Remove missing values, extract features

    # 2. Scale Features
    print("Scaling features...")
    X_scaled = cleaner.scale_features(X)  # Normalize feature values 0-1

    # 3. Create Visualizations
    # print("Creating visualizations...")
    # visualizer = DataVisualizer()
    # visualizer.plot_feature_distributions(X_scaled)  # Show feature histograms
    # visualizer.plot_correlation_matrix(X_scaled)  # Show feature relationships

    # 4. Build and Run Recommender
    print("Initializing recommender system...")
    recommender = MusicRecommender(features_data=X_scaled, songs_data=songs)  # Create similarity matrix

    # 5. Interactive search and recommendations
    try:
        search_keyword = input("\nEnter a text prompt (word in title or artist) to search for songs: ").strip()
    except EOFError:
        search_keyword = ""

    if not search_keyword:
        print("No input given. Using default keyword 'love'.")
        search_keyword = 'love'

    print(f"\nSearching for songs with '{search_keyword}' in the title or artist...")
    search_results = recommender.search_songs(search_keyword, limit=10)

    if search_results.empty:
        print(f"\nNo songs found with '{search_keyword}'. Exiting.")
        return

    print("\nFound these songs (choose one by number):")
    for i, row in search_results.iterrows():
        print(f"  [{i}] {row['name']} — {row['artists']}")

    # Ask user to choose a song by index with validation
    while True:
        choice = input("Enter the number of the song you want recommendations for (or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Exiting without generating recommendations.")
            return
        if choice.isdigit():
            idx = int(choice)
            if idx in search_results.index:
                chosen_title = search_results.loc[idx, 'name']
                break
        print("Invalid selection. Please enter one of the numbers shown above, or 'q' to quit.")

    print(f"\nGetting recommendations for '{chosen_title}'...")
    recommendations = recommender.get_recommendations(chosen_title, n_recommendations=5)

    print("\nRecommended songs:")
    print(recommendations)


if __name__ == "__main__":
    main()