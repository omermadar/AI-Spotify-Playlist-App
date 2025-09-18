import json
import requests
import re
from typing import List, Dict
import pandas as pd


class PlaylistCreator:
    def __init__(self, music_recommender, ollama_base_url="http://localhost:11434"):
        self.music_recommender = music_recommender
        self.ollama_base_url = ollama_base_url

    def get_seed_songs_from_llm(self, user_request: str, n_seeds: int = 10) -> List[str]:
        """
        Use Ollama to suggest seed songs based on the user's request.
        Returns list of song titles that we can search for.
        """
        prompt = f"""
        You are a hit-song seeding assistant for a recommender that uses cosine similarity and k-means over a large catalog of ~600k tracks.
        
        Task:
        - Given a natural-language playlist request, return EXACTLY {n_seeds} distinct SONG TITLES (no artists) as a JSON array.
        - Output must be the final, trimmed array only, no commentary.
        - Prefer globally popular, mainstream originals that are widely available in large catalogs (avoid remixes, live/sped-up/cover versions).
        - Prefer songs released 2000–2024 unless the request explicitly asks for earlier decades.
        - Keep titles as they appear on Spotify (proper casing, punctuation), but do NOT include the artist name.
        - Avoid duplicate titles or near-duplicates.
        - If the request is ambiguous, pick one coherent, popular interpretation and stay consistent.
        
        Bias rules (apply when relevant):
        - For energy/workout/gym/upbeat/party: pick high-energy hip-hop/pop bangers that are well-known and broadly popular.
        - For sad/mellow/acoustic/chill/study: pick widely-known, slower, acoustic or mellow tracks.
        - For decade/era/genre constraints: honor them first, then still prefer the biggest, most recognizable tracks in that slice.
        
        Examples (few-shot):
        
        Request: upbeat rap songs for workout
        Return: ["Lose Yourself","Till I Collapse","SICKO MODE","HUMBLE.","POWER","In Da Club","Stronger","DNA.","All of the Lights","Started From The Bottom"]
        
        Request: sad acoustic songs
        Return: ["Skinny Love","Hallelujah","The A Team","I Will Follow You into the Dark","Fast Car","Someone Like You","Heartbeats","The Scientist","Tears in Heaven","Photograph"]
        
        Now produce the seeds:
        
        Request: "{user_request}"
        Return:

        """

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                llm_output = result['response'].strip()

                # Extract JSON from response
                json_match = re.search(r'\[.*\]', llm_output, re.DOTALL)
                if json_match:
                    seed_songs = json.loads(json_match.group())
                    return seed_songs[:n_seeds]
        except Exception as e:
            print(f"Error getting seed songs from LLM: {e}")

        # Fallback: default workout rap songs
        if "rap" in user_request.lower() and ("workout" in user_request.lower() or "upbeat" in user_request.lower()):
            return ["Lose Yourself", "Till I Collapse", "Started From The Bottom", "Stronger", "In Da Club"]

        # Default fallback
        return ["Shape of You", "Blinding Lights", "Uptown Funk", "Can't Stop The Feeling", "Happy"]

    def create_playlist_from_description(self, user_request: str, final_playlist_size: int = 50) -> Dict:
        """
        Pipeline: Text → LLM Seed Songs → Your Enhanced Cosine Similarity → Language Filter
        """
        print(f"Creating playlist for: '{user_request}'")

        # Step 1: Get seed songs from LLM
        print("Step 1: Getting seed songs from Ollama...")
        seed_song_names = self.get_seed_songs_from_llm(user_request, n_seeds=10)
        print(f"Seed songs: {seed_song_names}")

        # Step 2: Determine preferred language based on request
        preferred_language = self._detect_language_preference(user_request)
        print(f"Detected language preference: {preferred_language}")

        # Step 3: Use your existing recommender with LANGUAGE FILTERING
        print("Step 2: Getting recommendations from your cosine similarity system with language filtering...")
        all_recommendations = []

        for i, seed_song_name in enumerate(seed_song_names):
            print(f"  Getting recommendations for seed {i + 1}: {seed_song_name}")
            try:
                # Use the enhanced language-aware recommendation method
                if hasattr(self.music_recommender, 'get_recommendations_with_language_filter'):
                    recommendations_df = self.music_recommender.get_recommendations_with_language_filter(
                        song_title=seed_song_name,
                        n_recommendations=25,  # Get fewer per seed but higher quality
                        preferred_language=preferred_language
                    )
                else:
                    # Fallback to regular method
                    recommendations_df = self.music_recommender.get_recommendations(
                        song_title=seed_song_name,
                        n_recommendations=25
                    )

                if not recommendations_df.empty:
                    # Apply additional quality filters
                    filtered_recommendations = self._apply_quality_filters(recommendations_df, user_request)

                    # Convert to list of dicts
                    for _, row in filtered_recommendations.iterrows():
                        song_dict = {
                            'name': row['name'],
                            'artists': row['artists'],
                            'track_id': row.get('track_id', ''),
                            'similarity_score': row.get('similarity_score', 0.0),
                            'seed_song': seed_song_name,
                            'language': row.get('language', 'unknown')
                        }
                        all_recommendations.append(song_dict)
                    print(f"    Found {len(filtered_recommendations)} quality recommendations")
                else:
                    print(f"    No recommendations found for {seed_song_name}")
            except Exception as e:
                print(f"    Error getting recommendations for {seed_song_name}: {e}")

        print(f"Total recommendations collected: {len(all_recommendations)}")

        # Step 4: IMPROVED duplicate removal
        unique_recommendations = self._remove_duplicates_improved(all_recommendations)
        print(f"After removing duplicates: {len(unique_recommendations)}")

        # Step 5: Final quality scoring and ranking (FIXED to stay in [0, 1] range)
        scored_recommendations = self._score_recommendations_fixed(unique_recommendations, user_request,
                                                                   preferred_language)

        # Step 6: Sort by total score and take top candidates
        final_songs = sorted(scored_recommendations, key=lambda x: x.get('total_score', x['similarity_score']),
                             reverse=True)[:final_playlist_size]
        print(f"Final playlist size: {len(final_songs)}")

        # Step 7: Format for output
        playlist_data = self.create_playlist_data(final_songs, user_request)

        return playlist_data

    def _detect_language_preference(self, user_request: str) -> str:
        """Detect the preferred language from user request."""
        request_lower = user_request.lower()

        # Check for explicit language mentions
        if any(word in request_lower for word in ['english', 'american', 'british']):
            return 'en'
        elif any(word in request_lower for word in ['spanish', 'latino', 'reggaeton']):
            return 'es'
        elif any(word in request_lower for word in ['french', 'francais']):
            return 'fr'
        elif any(word in request_lower for word in ['german', 'deutsch']):
            return 'de'

        # Default to English for most Western music genres
        western_genres = ['pop', 'rap', 'hip-hop', 'rock', 'country', 'r&b', 'soul', 'funk', 'disco', 'house', 'edm']
        if any(genre in request_lower for genre in western_genres):
            return 'en'

        return 'en'  # Default to English

    def _apply_quality_filters(self, recommendations_df: pd.DataFrame, user_request: str) -> pd.DataFrame:
        """Apply additional quality filters to remove obviously bad matches."""
        filtered = recommendations_df.copy()

        # Filter 1: Remove songs with very low similarity scores
        filtered = filtered[filtered.get('similarity_score', 0) >= 0.3]

        # Filter 2: For rap/hip-hop requests, try to filter out non-rap genres by artist names
        request_lower = user_request.lower()
        if 'rap' in request_lower or 'hip-hop' in request_lower or 'hip hop' in request_lower:
            # This is a simple heuristic - you could make this more sophisticated
            pass  # Keep all for now, let the similarity scores handle it

        # Filter 3: For workout requests, prefer higher energy songs
        if 'workout' in request_lower or 'gym' in request_lower or 'exercise' in request_lower:
            # Prefer songs with higher similarity scores (which should correlate with energy)
            filtered = filtered[filtered.get('similarity_score', 0) >= 0.4]

        return filtered

    def _score_recommendations_fixed(self, recommendations: List[Dict], user_request: str, preferred_language: str) -> \
    List[Dict]:
        """
        FIXED: Enhanced scoring system that keeps scores normalized between 0.0 and 1.0.
        """
        for rec in recommendations:
            base_score = rec['similarity_score']

            # Ensure base score is between 0 and 1
            base_score = max(0.0, min(1.0, base_score))

            # Use multiplicative bonuses that preserve the [0, 1] range
            language_bonus = 0.0
            if rec.get('language') == preferred_language:
                language_bonus = 0.05  # Small boost for preferred language
            elif rec.get('language') == 'unknown':  # Unknown language gets smaller bonus
                language_bonus = 0.02

            # Genre relevance bonus (simple keyword matching)
            genre_bonus = 0.0
            request_lower = user_request.lower()
            artist_name = str(rec.get('artists', '')).lower()

            # Rap/Hip-hop artist recognition
            if 'rap' in request_lower and any(artist in artist_name for artist in
                                              ['eminem', 'drake', 'kanye', 'jay-z', 'kendrick', 'cole', 'future',
                                               'travis', 'lil', 'young', '50 cent']):
                genre_bonus = 0.05  # Small boost for recognized rap artists

            # Calculate total score using weighted average to keep in [0, 1]
            # Weight the base score more heavily than bonuses
            total_score = (base_score * 0.85) + (language_bonus * 0.10) + (genre_bonus * 0.05)

            # Final normalization to ensure [0, 1] range
            total_score = max(0.0, min(1.0, total_score))

            rec['total_score'] = total_score
            rec['language_bonus'] = language_bonus
            rec['genre_bonus'] = genre_bonus

        return recommendations

    def _remove_duplicates_improved(self, recommendations: List[Dict]) -> List[Dict]:
        """Improved duplicate removal using multiple methods."""
        seen = {}

        for rec in recommendations:
            # Clean song name for better matching
            clean_name = self._clean_song_name(rec['name'])
            clean_artist = self._clean_artist_name(rec['artists'])

            # Create multiple keys to catch different duplicate patterns
            keys = [
                f"{clean_name}_{clean_artist}",  # Main key
                clean_name,  # Just song name (catches different versions)
            ]

            # Check all keys for duplicates
            is_duplicate = False
            best_key = keys[0]

            for key in keys:
                if key in seen:
                    # Found duplicate - keep the one with higher score
                    current_score = rec.get('total_score', rec['similarity_score'])
                    existing_score = seen[key].get('total_score', seen[key]['similarity_score'])

                    if current_score > existing_score:
                        # This version is better, remove the old one and use this
                        seen[key] = rec
                    is_duplicate = True
                    best_key = key
                    break

            if not is_duplicate:
                seen[best_key] = rec

        return list(seen.values())

    def _clean_song_name(self, name: str) -> str:
        """Clean song name to catch more duplicates."""
        if not isinstance(name, str):
            return str(name).lower()

        # Remove common additions that create duplicates
        name = name.lower()

        # Remove version indicators
        patterns = [
            r'\s*-\s*soundtrack version.*$',
            r'\s*-\s*from.*soundtrack.*$',
            r'\s*-\s*radio edit.*$',
            r'\s*-\s*album version.*$',
            r'\s*-\s*edited.*$',
            r'\s*\(.*version.*\).*$',
            r'\s*\(.*edit.*\).*$',
            r'\s*\(.*soundtrack.*\).*$',
            r'\s*\(remaster.*\).*$',
            r'\s*\(feat\..*\).*$',
            r'\s*feat\..*$',
        ]

        for pattern in patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)

        return name.strip()

    def _clean_artist_name(self, artists) -> str:
        """Clean artist name for better matching."""
        if not isinstance(artists, str):
            artists = str(artists)
        return artists.lower().strip()

    def create_playlist_data(self, songs: List[Dict], user_request: str) -> Dict:
        """Format for output."""
        playlist_name = f"AI Playlist: {user_request[:30]}..."

        track_uris = []
        for song in songs:
            if 'track_id' in song and song['track_id']:
                track_uris.append(f"spotify:track:{song['track_id']}")

        return {
            "name": playlist_name,
            "description": f"AI-generated playlist for: {user_request}",
            "track_uris": track_uris,
            "track_count": len(songs),
            "songs": [{"name": s['name'], "artist": s['artists'],
                       "score": round(s.get('total_score', s['similarity_score']), 3)} for s in songs]
        }