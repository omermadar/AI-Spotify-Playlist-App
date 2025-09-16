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
    df = pd.read_csv('tracks_added_languages.csv')  # Load raw music data
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
    # Initialize recommender with MiniBatchKMeans, also has default parameters
    recommender = MusicRecommender(features_data=X_scaled, songs_data=songs, language_weight= 0.2, artist_weight=0.1)

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
    search_results = search_results.reset_index(drop=True)
    for display_i, row in enumerate(search_results.itertuples(index=False), start=1):
        print(f"  [{display_i}] {row.name} — {row.artists}")

    # Ask user to choose a song by index with validation (1..N)
    n = len(search_results)
    while True:
        choice = input("Enter the number of the song you want recommendations for (or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Exiting without generating recommendations.")
            return
        if choice.isdigit():
            sel = int(choice)
            if 1 <= sel <= n:
                chosen_title = search_results.iloc[sel - 1]['name']
                break
        print(f"Invalid selection. Please enter a number between 1 and {n}, or 'q' to quit.")

    print(f"\nGetting recommendations for '{chosen_title}'...")
    recommendations = recommender.get_recommendations(chosen_title, n_recommendations=10)

    # Handle non-DataFrame responses
    if isinstance(recommendations, str):
        print(recommendations)
        return

    if recommendations is None or getattr(recommendations, 'empty', False):
        print("No recommendations available.")
        return

    # Choose score column: prefer blended total_similarity, else audio_similarity
    score_col = 'total_similarity' if 'total_similarity' in recommendations.columns else (
        'audio_similarity' if 'audio_similarity' in recommendations.columns else None
    )

    print("\nTop 10 recommended songs:")
    for i, row in enumerate(recommendations.itertuples(index=False), start=1):
        name = getattr(row, 'name', 'Unknown')
        artists = getattr(row, 'artists', 'Unknown')
        if score_col:
            score_val = getattr(row, score_col, None)
            try:
                score_txt = f"{float(score_val):.3f}"
            except Exception:
                score_txt = "N/A"
        else:
            score_txt = "N/A"
        print(f"  [{i}] {name} — {artists}  (similarity: {score_txt})")


if __name__ == "__main__":
    main()