import pandas as pd
import re
from typing import Optional, Tuple

# Optional language detection backends
_LANGID = None
_CLD3 = None
_LANGDETECT = None

try:
    import langid  # fast, no internet, ISO codes
    _LANGID = langid
except Exception:
    _LANGID = None

try:
    import cld3  # Compact Language Detector v3
    _CLD3 = cld3
except Exception:
    _CLD3 = None

try:
    from langdetect import detect as _ld_detect
    _LANGDETECT = _ld_detect
except Exception:
    _LANGDETECT = None


ISO_TO_NAME = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'ar': 'Arabic', 'ru': 'Russian', 'he': 'Hebrew',
    'de': 'German', 'pt': 'Portuguese', 'it': 'Italian', 'nl': 'Dutch', 'sv': 'Swedish', 'no': 'Norwegian',
    'da': 'Danish', 'fi': 'Finnish', 'pl': 'Polish', 'uk': 'Ukrainian', 'tr': 'Turkish', 'el': 'Greek',
    'fa': 'Persian', 'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'ur': 'Urdu',
    'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'zh-cn': 'Chinese', 'zh-tw': 'Chinese',
}

# Regex ranges for quick script detection (high precision)
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
HEBREW_RE = re.compile(r'[\u0590-\u05FF]')
CYRILLIC_RE = re.compile(r'[\u0400-\u04FF]')
CJK_RE = re.compile(r'[\u3040-\u30FF\u3400-\u9FFF]')  # Japanese Hiragana/Katakana + CJK


def _best_lang_backend(text: str) -> Tuple[Optional[str], float, str]:
    """Try multiple detectors and return (lang_code, confidence, backend_name)."""
    text = (text or '').strip()
    if not text:
        return None, 0.0, 'none'

    # cld3 provides probability and reliability
    if _CLD3 is not None:
        try:
            res = _CLD3.get_language(text)
            if res and res.is_reliable and res.probability >= 0.6 and res.language:
                code = res.language.lower()
                # Normalize Chinese variants
                if code.startswith('zh'):
                    code = 'zh'
                return code, float(res.probability), 'cld3'
        except Exception:
            pass

    # langid provides a score
    if _LANGID is not None:
        try:
            code, score = _LANGID.classify(text)
            if score >= -10:  # langid returns log-prob; accept if not extremely low
                return code.lower(), max(0.0, min(1.0, 1.0 + score / 20.0)), 'langid'
        except Exception:
            pass

    # langdetect (no score)
    if _LANGDETECT is not None:
        try:
            code = _LANGDETECT(text)
            return code.lower(), 0.5, 'langdetect'
        except Exception:
            pass

    return None, 0.0, 'none'


def _iso_to_name(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.lower()
    return ISO_TO_NAME.get(code, code)


def detect_language(songs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'language' and auxiliary language metadata columns to the songs DataFrame using:
    - Strong script regex (Arabic, Hebrew, Cyrillic, CJK)
    - Multi-backend language id on (title, artists, combined)
    - Spotify audio features to flag Wordless (instrumental) and Spoken Word via speechiness

    New columns:
      - language: human-friendly language or 'Wordless'/'Spoken Word'/'Unknown'
      - language_source: 'script'|'audio_features'|'cld3'|'langid'|'langdetect'|'heuristic'|'none'
      - language_confidence: [0..1] confidence score (heuristic mapping)
      - has_words: boolean inference based on speechiness/instrumentalness
    """
    df = songs_df.copy()

    # Ensure text fields are strings
    for col in ['name', 'artists']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            df[col] = ''

    # Initialize columns
    df['language'] = 'Unknown'
    df['language_source'] = 'none'
    df['language_confidence'] = 0.0

    # Audio feature heuristics
    speech = df.get('speechiness', pd.Series([None] * len(df)))
    instr = df.get('instrumentalness', pd.Series([None] * len(df)))

    # Infer has_words using speechiness thresholds from Spotify definition
    has_words = (
        (speech.notna()) & (
            (speech >= 0.33) |  # substantial spoken content
            ((speech < 0.33) & (instr.notna()) & (instr < 0.6))  # low speech but also not highly instrumental
        )
    )
    df['has_words'] = has_words.fillna(False)

    # Flag Spoken Word and Wordless using stronger thresholds
    # - Spoken Word: speechiness > 0.66
    # - Wordless: speechiness < 0.12 and instrumentalness >= 0.75
    spoken_mask = (speech.notna()) & (speech > 0.66)
    wordless_mask = (speech.notna()) & (speech < 0.12) & (instr.notna()) & (instr >= 0.75)

    df.loc[spoken_mask, ['language', 'language_source', 'language_confidence']] = ['Spoken Word', 'audio_features', 0.95]
    df.loc[wordless_mask, ['language', 'language_source', 'language_confidence']] = ['Wordless', 'audio_features', 0.95]

    # For remaining rows, try script detection first (high precision)
    undecided = df.index[~spoken_mask & ~wordless_mask]
    if len(undecided) > 0:
        titles = df.loc[undecided, 'name']
        artists = df.loc[undecided, 'artists']
        scripts = []
        for t, a in zip(titles, artists):
            s = 'latin'
            txt = f"{t} {a}"
            if ARABIC_RE.search(txt):
                s = 'Arabic'
            elif HEBREW_RE.search(txt):
                s = 'Hebrew'
            elif CYRILLIC_RE.search(txt):
                s = 'Russian'
            elif CJK_RE.search(txt):
                s = 'Chinese'
            scripts.append(s)
        df.loc[undecided, 'language_script_hint'] = scripts
        # Apply script hint directly for non-latin scripts
        for lang_name in ['Arabic', 'Hebrew', 'Russian', 'Chinese']:
            mask = (df.index.isin(undecided)) & (df['language_script_hint'] == lang_name) & (df['language'] == 'Unknown')
            df.loc[mask, ['language', 'language_source', 'language_confidence']] = [lang_name, 'script', 0.98]

    # For still-unknown and latin script, use language id backends on title, artist, combined
    still_unknown = df.index[df['language'] == 'Unknown']
    for idx in still_unknown:
        title = df.at[idx, 'name']
        artist = df.at[idx, 'artists']
        combined = (title + ' ' + artist).strip()

        # Try title
        code, conf, src = _best_lang_backend(title)
        best = (code, conf, src, 'title')
        # Try artist
        code2, conf2, src2 = _best_lang_backend(artist)
        if conf2 > best[1]:
            best = (code2, conf2, src2, 'artist')
        # Try combined
        code3, conf3, src3 = _best_lang_backend(combined)
        if conf3 > best[1]:
            best = (code3, conf3, src3, 'combined')

        code = best[0]
        conf = best[1]
        src = best[2]

        if code:
            lang_name = _iso_to_name(code)
            if isinstance(lang_name, str) and lang_name:
                df.at[idx, 'language'] = lang_name
                df.at[idx, 'language_source'] = src
                df.at[idx, 'language_confidence'] = float(min(1.0, max(0.0, conf)))
        # If no code, leave as Unknown; if has_words is False and not yet Wordless, fallback to Wordless (so it doesn't pollute)
        if df.at[idx, 'language'] == 'Unknown' and not df.at[idx, 'has_words']:
            df.at[idx, 'language'] = 'Wordless'
            df.at[idx, 'language_source'] = 'heuristic'
            df.at[idx, 'language_confidence'] = 0.6

    # Clean up helper column if existed
    if 'language_script_hint' in df.columns:
        df.drop(columns=['language_script_hint'], inplace=True)

    return df


def save_songs_with_language(songs_df: pd.DataFrame, filename: str = 'tracks_added_languages_(Old-Spanish-Problem).csv') -> None:
    """Saves the songs DataFrame with the added language columns to a CSV file."""
    print(f"\nSaving songs with languages to {filename}...")
    songs_df.to_csv(filename, index=False)
    print("File saved successfully! You can now check it for language information.")


def main():
    """Load data, detect languages, and save to a new file."""
    print("--- Language Detection and CSV Export ---")
    print("Loading raw music data...")
    df = pd.read_csv('tracks.csv')

    songs_df = df.copy()

    print("Detecting languages using text + audio features...")
    songs_with_language = detect_language(songs_df)

    save_songs_with_language(songs_with_language)


if __name__ == "__main__":
    main()