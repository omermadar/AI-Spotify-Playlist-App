import json
import requests
import re
from typing import List, Dict
import pandas as pd


class PlaylistCreator:
    def __init__(self, music_recommender, ollama_base_url="http://localhost:11434"):
        self.music_recommender = music_recommender
        self.ollama_base_url = ollama_base_url

    def call_ollama(self, prompt: str, model: str = "llama3.1:8b") -> str:
        """Helper method to call Ollama API."""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(f"{self.ollama_base_url}/api/generate", json=data, timeout=30)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Ollama API error: {response.status_code}")
                return ''
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return ''

    def get_seed_songs_from_llm(self, user_request: str, n_seeds: int = 10) -> List[Dict]:
        """
        Use Ollama to suggest seed songs (with artists) based on the user's request.
        Returns a list of dictionaries, e.g., [{"artist": "Drake", "title": "God's Plan"}].
        """
        prompt = f"""
        You are a hit-song seeding assistant. Your task is to generate a list of seed songs based on a user's request.

        **Instructions:**
        1.  **Output Format:** Return a single, clean JSON array of objects. Each object must have two keys: "artist" and "title".
        2.  **Content:** The array should contain exactly {n_seeds} distinct songs.
        3.  **Song Choice:**
            *   Prefer globally popular, mainstream original versions.
            *   Avoid remixes, live, or acoustic versions unless explicitly requested.
            *   Adhere strictly to the mood and genre of the request.
            *   Avoid songs released after 2020.

        **Examples:**

        Request: upbeat rap songs for a workout
        Return: [ {{"artist": "Eminem", "title": "Lose Yourself"}}, {{"artist": "Kanye West", "title": "Stronger"}} ]

        Request: mellow, sad acoustic songs
        Return: [ {{"artist": "Bon Iver", "title": "Skinny Love"}}, {{"artist": "Jeff Buckley", "title": "Hallelujah"}} ]

        Now, fulfill this request:

        Request: "{user_request}"

        Return:
        """

        try:
            response = self.call_ollama(prompt)
            if response:
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    potential_json = json_match.group(0)
                    try:
                        seed_songs = json.loads(potential_json)
                        # Basic validation
                        if isinstance(seed_songs, list) and all(isinstance(s, dict) and 'artist' in s and 'title' in s for s in seed_songs):
                            print(f"✅ Successfully parsed {len(seed_songs)} seed songs from LLM response.")
                            return seed_songs[:n_seeds]
                        else:
                            print("⚠️ LLM response was not in the expected format (list of artist/title dicts).")
                    except json.JSONDecodeError:
                        print("⚠️ LLM response contained a JSON-like structure that failed to parse.")

        except Exception as e:
            print(f"Error getting seed songs from LLM: {e}")

        # Fallback
        print("⚠️ Could not generate seed songs from LLM, using a default fallback.")
        return [ {"artist": "Drake", "title": "God's Plan"} ]


    def filter_recommendations_with_llm(self, songs_list: List[Dict], user_request: str, seed_song: str = None) -> List[
        Dict]:
        """
        Use Ollama to filter and score song recommendations based on how well they fit the original request.
        """
        if not songs_list:
            return []

        # Prepare song list for LLM
        song_descriptions = []
        for i, song in enumerate(songs_list):
            artist = song.get('artist', song.get('artists', 'Unknown'))
            song_descriptions.append(f"{i + 1}. {song['name']} by {artist}")

        songs_text = "\\n".join(song_descriptions[:15])  # Limit to avoid token issues

        seed_context = f" (similar to {seed_song})" if seed_song else ""

        prompt = f"""
        You are a critical music curator. Your task is to rate a list of songs based on how well they fit a user's playlist request.

        **User Request:** "{user_request}"
        **Seed Song:** "{seed_song}"

        **Instructions:**
        1.  **Rate each song** from 1 (bad fit) to 10 (perfect fit).
        2.  **Be critical.** A score of 5-6 is average. A 9 or 10 should be reserved for songs that are a near-perfect match.
        3.  **Prioritize:**
            *   **Genre and Artist Match:** Does the song match the genre and artists in the user's request? Penalize songs that are clearly from a different genre.
            *   **Mood Match:** Does the song's mood (e.g., mellow, energetic, sad) fit the request?
            *   **Seed Song Similarity:** Is the song a good recommendation based on the seed song?
        4.  **Output Format:** Respond ONLY with a clean JSON array of integer scores (e.g., [8, 6, 9, 4, 7]).

        **Songs to Rate:**
        {songs_text}

        **Return (JSON array of scores only):**
        """

        try:
            response = self.call_ollama(prompt)
            if response and response.strip().startswith('['):
                scores = json.loads(response.strip())

                # Add LLM scores to songs
                scored_songs = []
                for i, song in enumerate(songs_list[:len(scores)]):
                    song_copy = song.copy()
                    song_copy['llm_score'] = scores[i] if i < len(scores) else 5
                    # Combine with similarity score
                    similarity = song.get('similarity_score', 0.5)
                    song_copy['total_score'] = (similarity * 0.6) + (scores[i] / 10.0 * 0.4)
                    scored_songs.append(song_copy)

                # Sort by combined score
                return sorted(scored_songs, key=lambda x: x['total_score'], reverse=True)

        except Exception as e:
            print(f"⚠️  LLM filtering failed: {e}")

        # Fallback: return original list with similarity scores
        return songs_list

    def _normalize_song_fields(self, songs_list: List[Dict]) -> List[Dict]:
        """Normalize song field names to ensure consistency."""
        normalized = []

        for song in songs_list:
            normalized_song = song.copy()

            # Ensure 'artist' field exists
            if 'artist' not in normalized_song and 'artists' in normalized_song:
                artists = normalized_song['artists']
                if isinstance(artists, list):
                    normalized_song['artist'] = artists[0] if artists else 'Unknown'
                else:
                    normalized_song['artist'] = str(artists)
            elif 'artist' not in normalized_song:
                normalized_song['artist'] = 'Unknown'

            # Ensure required fields exist
            normalized_song.setdefault('name', 'Unknown Track')
            normalized_song.setdefault('similarity_score', 0.0)

            normalized.append(normalized_song)

        return normalized

    def create_playlist_from_description_enhanced(self, user_request: str, final_playlist_size: int = 50, use_llm_filter: bool = True):
        """
        Enhanced playlist creation with LLM filtering, seed-based diversity, and smart fallbacks.
        """

        try:
            print(f"🎤 Creating playlist for: '{user_request}' (LLM Filter: {use_llm_filter})")

            # Step 1: Get seed songs from LLM
            print("🌱 Getting seed songs...")
            seed_songs = self.get_seed_songs_from_llm(user_request, n_seeds=10)

            if not seed_songs:
                print("❌ No seed songs generated")
                return None

            print(f"✅ Generated {len(seed_songs)} seed songs")

            # Step 2: Collect recommendations from all seeds
            all_buckets = []
            for seed in seed_songs:
                seed_title = seed.get('title', '[Unknown Title]')
                print(f"🔍 Getting recommendations for seed: {seed_title} by {seed.get('artist', '[Unknown Artist]')}")

                # Get similarity-based recommendations (passing the whole seed object)
                recs = self.music_recommender.get_recommendations(seed, n_recommendations=75, user_request=user_request)

                if recs.empty:
                    print(f"❌ No recommendations found for '{seed}' - skipping")
                    continue

                # Convert to list of dicts and normalize field names
                recs_list = recs.to_dict('records')
                normalized_recs = self._normalize_song_fields(recs_list)

                # Step 4: LLM filters each bucket (optional)
                # FOR THIS TEST: LLM filtering is temporarily disabled to establish a baseline.
                print("🚫 LLM filtering is temporarily disabled for this baseline test.")
                filtered_recs = sorted(normalized_recs, key=lambda x: x.get('similarity_score', 0), reverse=True)

                # Add seed source to each recommendation for traceability
                for rec in filtered_recs:
                    rec['seed_source'] = seed

                # Take all good recommendations from the seed
                if filtered_recs:
                    all_buckets.append(filtered_recs)
                    print(f"✅ Selected {len(filtered_recs)} songs from '{seed}'")
                else:
                    print(f"⚠️  No songs passed filter for '{seed}'")

            if not all_buckets:
                print("❌ No recommendations from any seed songs")
                return None

            # --- New Assembly Logic: Best of the Best ---
            print("🎯 Assembling final playlist from the best songs across all seeds...")

            # 1. Flatten all buckets into a single list
            all_recommendations = [song for bucket in all_buckets for song in bucket]

            # 2. Determine the sorting key
            if use_llm_filter:
                sort_key = lambda x: x.get('total_score', 0)
            else:
                sort_key = lambda x: x.get('similarity_score', 0)
            
            all_recommendations.sort(key=sort_key, reverse=True)

            # 3. Build the final playlist with de-duplication
            final_songs = []
            seen_tracks = set()
            for song in all_recommendations:
                if len(final_songs) >= final_playlist_size:
                    break
                
                track_key = (song.get('name', ''), song.get('artist', ''))
                if track_key not in seen_tracks:
                    final_songs.append(song)
                    seen_tracks.add(track_key)

            # Step 5: Fill remaining slots (This is now a fallback, as the new logic should usually hit the target)
            if len(final_songs) < final_playlist_size:
                print(f"  [Fill] Playlist is still short. Running fallback fill...")

                # Collect all remaining songs
                all_remaining = []
                for bucket in all_buckets:
                    all_remaining.extend(bucket)

                # Remove duplicates and sort by score
                remaining_unique = []
                for song in all_remaining:
                    track_key = (song['name'], song.get('artist', ''))
                    if track_key not in seen_tracks:
                        remaining_unique.append(song)

                # Adjust sorting key based on whether LLM filter was used
                if use_llm_filter:
                    sort_key = lambda x: (x.get('total_score', 0), x.get('similarity_score', 0))
                else:
                    sort_key = lambda x: x.get('similarity_score', 0)
                
                remaining_sorted = sorted(remaining_unique, key=sort_key, reverse=True)

                # Add remaining songs
                for song in remaining_sorted:
                    if len(final_songs) >= final_playlist_size:
                        break
                    final_songs.append(song)

            # Step 6: Format output
            playlist_name = f"{user_request.title()} - AI Generated"
            description = f"AI-curated playlist for: {user_request}"

            result = {
                'name': playlist_name,
                'description': description,
                'songs': final_songs[:final_playlist_size],
                'total_songs': len(final_songs),
                'seed_songs': seed_songs,
                'successful_seeds': len(all_buckets)
            }

            print(f"🎉 Created playlist with {len(final_songs)} songs from {len(all_buckets)} seeds!")
            return result

        except Exception as e:
            print(f"❌ Error creating playlist: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Keep the original method name for backward compatibility
    def create_playlist_from_description(self, user_request: str, final_playlist_size: int = 50, use_llm_filter: bool = True):
        """Backward compatibility wrapper for the enhanced method."""
        return self.create_playlist_from_description_enhanced(user_request, final_playlist_size, use_llm_filter)

    def _detect_language_preference(self, user_request: str) -> str:
        """Detect the preferred language from user request."""
        request_lower = user_request.lower()

        # Check for explicit language mentions
        if any(word in request_lower for word in ['english', 'american', 'british']):
            return 'English'
        elif any(word in request_lower for word in ['spanish', 'latino', 'reggaeton']):
            return 'Spanish'
        elif any(word in request_lower for word in ['french', 'francais']):
            return 'French'
        elif any(word in request_lower for word in ['german', 'deutsch']):
            return 'German'
        elif any(word in request_lower for word in ['hebrew']):
            return 'Hebrew'

        # Default to English for most Western music genres
        western_genres = ['pop', 'rap', 'hip-hop', 'rock', 'country', 'r&b', 'soul', 'funk', 'disco', 'house', 'edm']
        if any(genre in request_lower for genre in western_genres):
            return 'English'

        return 'English'  # Default to English

    def _apply_quality_filters(self, recommendations_df: pd.DataFrame, user_request: str) -> pd.DataFrame:
        """Apply additional quality filters to remove obviously bad matches."""
        filtered = recommendations_df.copy()

        # Filter 1: Remove songs with very low similarity scores
        filtered = filtered[filtered.get('similarity_score', 0) >= 0.3]

        # Filter 2: For workout requests, prefer higher energy songs
        request_lower = user_request.lower()
        if 'workout' in request_lower or 'gym' in request_lower or 'exercise' in request_lower:
            # Prefer songs with higher similarity scores (which should correlate with energy)
            filtered = filtered[filtered.get('similarity_score', 0) >= 0.4]

        return filtered

    def save_playlist_locally(self, playlist_data: dict, filename: str = None):
        """Save playlist data to a local file."""
        if not filename:
            # Generate filename from playlist name
            safe_name = "".join(c for c in playlist_data['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name.replace(' ', '_')}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(playlist_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Playlist saved locally as: {filename}")
            return True

        except Exception as e:
            print(f"❌ Error saving playlist: {e}")
            return False