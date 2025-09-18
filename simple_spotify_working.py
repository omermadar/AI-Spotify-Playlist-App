#!/usr/bin/env python3
"""
WORKING Spotify Integration - Simple CLI Authentication
This version works by showing you the exact URL to visit manually
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Optional
import json
import re

class SimpleSpotifyIntegration:
    """
    Simple Spotify integration that works with manual URL authentication.
    No browser dependency - just copy/paste URLs.
    """
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize with your Spotify app credentials."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = "http://127.0.0.1:8888/callback"
        self.scope = "playlist-modify-public playlist-modify-private"
        
        # Create SpotifyOAuth with cache
        self.sp_oauth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            cache_path=".spotify_cache",  # Cache tokens locally
            open_browser=False  # Don't try to open browser automatically
        )
        
        self.sp = None
        self.authenticated = False
    
    def authenticate(self) -> bool:
        """
        Authenticate with Spotify using manual URL method.
        This method ALWAYS works regardless of browser settings.
        """
        try:
            print("🔐 Spotify Authentication Required")
            print("=" * 50)
            
            # Step 1: Get the authorization URL
            auth_url = self.sp_oauth.get_authorize_url()
            
            print("📋 COPY THIS URL AND OPEN IT IN YOUR BROWSER:")
            print(f"\n{auth_url}\n")
            print("👆 Copy the URL above and paste it into any web browser")
            print("🔑 Log in to Spotify and click 'Agree' to authorize the app")
            print("📄 After authorization, you'll see a page that might show an error")
            print("🔗 Copy the ENTIRE URL from your browser's address bar")
            print("   (It should start with: http://127.0.0.1:8888/callback?code=...)")
            print("=" * 50)
            
            # Step 2: Get the redirect URL from the user
            while True:
                redirect_url = input("\n📥 Paste the full redirect URL here: ").strip()
                
                if not redirect_url:
                    print("❌ Please paste the redirect URL")
                    continue
                    
                if not redirect_url.startswith("http://127.0.0.1:8888/callback"):
                    print("❌ That doesn't look like the right URL. It should start with:")
                    print("   http://127.0.0.1:8888/callback?code=...")
                    continue
                    
                break
            
            # Step 3: Extract the code and get token
            print("🔄 Processing authorization...")
            
            try:
                # Parse the authorization code from the URL
                code = self.sp_oauth.parse_response_code(redirect_url)
                if not code:
                    print("❌ Could not find authorization code in URL")
                    return False
                
                # Get the access token
                token_info = self.sp_oauth.get_access_token(code)
                if not token_info:
                    print("❌ Could not get access token")
                    return False
                
                # Create Spotify client
                self.sp = spotipy.Spotify(auth_manager=self.sp_oauth)
                
                # Test the connection
                user = self.sp.current_user()
                print(f"✅ Success! Authenticated as: {user['display_name']} (@{user['id']})")
                print(f"🎵 Ready to create playlists in your Spotify account!")
                
                self.authenticated = True
                return True
                
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                print("💡 Make sure you copied the complete redirect URL")
                return False
                
        except Exception as e:
            print(f"❌ Error during authentication: {e}")
            return False
    
    def create_playlist_from_songs(self, playlist_name: str, description: str, songs: List[Dict]) -> Dict:
        """Create a Spotify playlist from AI-recommended songs."""
        if not self.authenticated:
            return {'success': False, 'error': 'Not authenticated with Spotify'}
            
        try:
            print(f"\n🎵 Creating playlist: '{playlist_name}'")
            
            # Get current user
            user = self.sp.current_user()
            user_id = user['id']
            
            # Create empty playlist
            playlist = self.sp.user_playlist_create(
                user=user_id,
                name=playlist_name,
                description=description,
                public=True
            )
            
            playlist_id = playlist['id']
            playlist_url = playlist['external_urls']['spotify']
            
            print(f"✅ Created empty playlist")
            print(f"🔗 Playlist URL: {playlist_url}")
            
            # Search for and add tracks
            track_uris = []
            found_count = 0
            
            print(f"\n🔍 Finding {len(songs)} songs on Spotify...")
            
            for i, song in enumerate(songs, 1):
                song_name = song.get('name', '')
                artist_name = song.get('artist', song.get('artists', ''))
                
                print(f"  {i:2d}. Searching: {song_name} - {artist_name}", end="")
                
                # Search for the track
                track_uri = self._search_track(song_name, artist_name)
                
                if track_uri:
                    track_uris.append(track_uri)
                    found_count += 1
                    print(" ✅")
                else:
                    print(" ❌")
            
            # Add tracks to playlist in batches
            if track_uris:
                print(f"\n📥 Adding {len(track_uris)} songs to playlist...")
                
                # Spotify allows max 100 tracks per request
                batch_size = 100
                for i in range(0, len(track_uris), batch_size):
                    batch = track_uris[i:i + batch_size]
                    self.sp.playlist_add_items(playlist_id, batch)
                
                print(f"🎉 SUCCESS! Added {len(track_uris)} out of {len(songs)} songs!")
                
            else:
                print("❌ No songs were found on Spotify")
            
            return {
                'success': True,
                'playlist_id': playlist_id,
                'playlist_url': playlist_url,
                'playlist_name': playlist_name,
                'tracks_added': len(track_uris),
                'tracks_requested': len(songs),
                'success_rate': f"{len(track_uris)}/{len(songs)} ({100*len(track_uris)/len(songs):.1f}%)"
            }
            
        except Exception as e:
            print(f"❌ Error creating playlist: {e}")
            return {'success': False, 'error': str(e)}
    
    def _search_track(self, song_name: str, artist_name: str) -> Optional[str]:
        """Search for a track and return its URI."""
        try:
            # Clean artist name
            if isinstance(artist_name, str):
                artist_clean = re.sub(r"[\[\]']", "", artist_name)
                artist_clean = artist_clean.split(',')[0].strip()
            else:
                artist_clean = str(artist_name)
            
            # Try different search strategies
            search_queries = [
                f'track:"{song_name}" artist:"{artist_clean}"',
                f'"{song_name}" "{artist_clean}"',
                f'{song_name} {artist_clean}',
                song_name
            ]
            
            for query in search_queries:
                results = self.sp.search(q=query, type='track', limit=1)
                
                if results['tracks']['items']:
                    return results['tracks']['items'][0]['uri']
            
            return None
            
        except:
            return None


def test_spotify_integration():
    """Test the Spotify integration."""
    print("🧪 Testing Spotify Integration")
    print("=" * 40)
    
    # Get credentials from user
    client_id = input("Enter your Spotify Client ID: ").strip()
    client_secret = input("Enter your Spotify Client Secret: ").strip()
    
    if not client_id or not client_secret:
        print("❌ Please provide both Client ID and Client Secret")
        return
    
    # Initialize and authenticate
    spotify = SimpleSpotifyIntegration(client_id, client_secret)
    
    if spotify.authenticate():
        print("\n🎉 Authentication successful!")
        
        # Test playlist creation
        test_songs = [
            {"name": "Lose Yourself", "artist": "Eminem"},
            {"name": "God's Plan", "artist": "Drake"},
            {"name": "Stronger", "artist": "Kanye West"}
        ]
        
        result = spotify.create_playlist_from_songs(
            playlist_name="AI Test Playlist",
            description="Test playlist created by AI music recommender",
            songs=test_songs
        )
        
        if result['success']:
            print(f"\n🎵 Test playlist created successfully!")
            print(f"🔗 URL: {result['playlist_url']}")
        else:
            print(f"❌ Failed to create test playlist: {result.get('error', 'Unknown error')}")
    
    else:
        print("❌ Authentication failed")


if __name__ == "__main__":
    test_spotify_integration()