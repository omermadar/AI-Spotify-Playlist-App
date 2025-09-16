import pandas as pd
import re
from langdetect import detect


def detect_language(songs_df):
    """
    Adds a 'language' column to the songs DataFrame based on various cues.
    """
    songs_df['language'] = 'Unknown'

    # Fill any missing values with an empty string to prevent the TypeError
    songs_df['name'] = songs_df['name'].fillna('')
    songs_df['artists'] = songs_df['artists'].fillna('')

    # Regex for character sets
    arabic_regex = re.compile(r'[\u0600-\u06FF]')
    hebrew_regex = re.compile(r'[\u0590-\u05FF]')
    cyrillic_regex = re.compile(r'[\u0400-\u04FF]')

    for index, row in songs_df.iterrows():
        title = row['name']
        artist = row['artists']

        # Direct signals from song title
        if arabic_regex.search(title) or arabic_regex.search(artist):
            songs_df.loc[index, 'language'] = 'Arabic'
        elif hebrew_regex.search(title) or hebrew_regex.search(artist):
            songs_df.loc[index, 'language'] = 'Hebrew'
        elif cyrillic_regex.search(title) or cyrillic_regex.search(artist):
            songs_df.loc[index, 'language'] = 'Russian/Cyrillic'

        # Simple keyword checks for common languages
        elif any(word in title.lower() for word in ['la', 'el', 'un', 'una', 'qué', 'que', 'mi', 'de', 'tu']):
            songs_df.loc[index, 'language'] = 'Spanish'
        elif any(word in title.lower() for word in ['le', 'la', 'un', 'une', 'je', 'tu', 'il', 'elle', 'mon']):
            songs_df.loc[index, 'language'] = 'French'

        # Fallback to language detection library
        else:
            try:
                lang = detect(title)
                if lang in ['en', 'es', 'fr', 'ar', 'ru', 'he']:  # Limit to common ones
                    songs_df.loc[index, 'language'] = lang.upper()
            except:
                pass  # Keep as 'Unknown' if detection fails

    return songs_df


def save_songs_with_language(songs_df, filename='tracks_added_languages.csv'):
    """
    Saves the songs DataFrame with the added language column to a CSV file.
    """
    print(f"\nSaving songs with languages to {filename}...")
    songs_df.to_csv(filename, index=False)
    print(f"File saved successfully! You can now check it for language information.")


def main():
    """
    Main function to load the data, detect languages, and save to a new file.
    """
    print("--- Language Detection and CSV Export ---")

    # 1. Load Data
    print("Loading raw music data...")
    df = pd.read_csv('tracks.csv')

    # Create a new DataFrame for songs with only relevant metadata
    songs_df = df.copy()

    # 2. Detect Languages
    print("Detecting languages based on song titles and artists...")
    songs_with_language = detect_language(songs_df)

    # 3. Save to a new CSV file
    save_songs_with_language(songs_with_language)


if __name__ == "__main__":
    main()