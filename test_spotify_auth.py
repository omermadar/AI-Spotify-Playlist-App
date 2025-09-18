#!/usr/bin/env python3
"""
QUICK TEST - Run this first to verify Spotify authentication works
"""

from simple_spotify_working import SimpleSpotifyIntegration

def main():
    print("🧪 SPOTIFY AUTHENTICATION TEST")
    print("="*50)
    print("This will test if we can connect to Spotify.")
    print("You'll need your Client ID and Secret from:")
    print("https://developer.spotify.com/dashboard")
    print()
    
    # Get credentials
    client_id = input("Enter your Spotify Client ID: ").strip()
    if not client_id:
        print("❌ Client ID required")
        return
        
    client_secret = input("Enter your Spotify Client Secret: ").strip()  
    if not client_secret:
        print("❌ Client Secret required")
        return
    
    # Test authentication
    spotify = SimpleSpotifyIntegration(client_id, client_secret)
    
    print("\n🔄 Starting authentication...")
    success = spotify.authenticate()
    
    if success:
        print("\n🎉 AUTHENTICATION SUCCESSFUL!")
        print("✅ You can now create playlists on Spotify")
        
        # Offer to create test playlist
        test = input("\nCreate a test playlist with 3 songs? (y/n): ").strip().lower()
        if test == 'y':
            test_songs = [
                {"name": "Blinding Lights", "artist": "The Weeknd"},
                {"name": "Watermelon Sugar", "artist": "Harry Styles"}, 
                {"name": "Levitating", "artist": "Dua Lipa"}
            ]
            
            result = spotify.create_playlist_from_songs(
                playlist_name="🤖 AI Test Playlist",
                description="Test playlist created by AI music system",
                songs=test_songs
            )
            
            if result['success']:
                print(f"\n🎵 Test playlist created!")
                print(f"🔗 Check your Spotify: {result['playlist_url']}")
            else:
                print(f"❌ Test failed: {result.get('error')}")
    else:
        print("\n❌ AUTHENTICATION FAILED")
        print("💡 Make sure you:")
        print("   - Copied the complete redirect URL")
        print("   - Have the right Client ID and Secret")
        print("   - Added http://127.0.0.1:8888/callback to your app settings")

if __name__ == "__main__":
    main()