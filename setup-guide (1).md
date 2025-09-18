# AI Music Recommendation System - Setup Guide

## 🎵 Complete Interactive Playlist Creator with Spotify Integration

This system creates personalized playlists from natural language descriptions using AI recommendations and can automatically create them on Spotify.

### ✅ What You Get:
- **AI-powered recommendations** using your cosine similarity + clustering system
- **Natural language input**: "upbeat rap for workout" → actual playlist
- **Spotify integration**: Creates real playlists in your Spotify account  
- **Interactive CLI**: Easy-to-use command line interface
- **Local saving**: Save playlists as JSON files
- **Smart filtering**: Language and genre-aware recommendations

---

## 🚀 Complete Setup Guide

### 1. System Requirements

**Operating System:**
- macOS (recommended)
- Windows 10/11
- Linux (Ubuntu/Debian)

**Python:**
- Python 3.8 or higher
- pip package manager

**External Services:**
- Ollama (for AI language processing)
- Spotify Developer Account (for playlist creation)

---

### 2. Install Ollama (Required for AI Processing)

**On macOS:**
```bash
# Install Ollama
brew install ollama

# Or download from: https://ollama.ai/download
```

**On Windows/Linux:**
```bash
# Download installer from: https://ollama.ai/download
# Follow installation instructions for your OS
```

**Setup Ollama:**
```bash
# Start Ollama service
ollama serve

# In a new terminal, install any model you want
ollama pull llama2 #example
# OR if you prefer a smaller model:
ollama pull phi

# Verify installation
ollama list
```

**Test Ollama:**
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello world",
  "stream": false
}'
```

---

### 3. Install Python Dependencies

**Create Virtual Environment (Recommended):**
```bash
# Create virtual environment
python3 -m venv music-ai-env

# Activate it
source music-ai-env/bin/activate  # macOS/Linux
# OR
music-ai-env\Scripts\activate     # Windows
```

**Install All Required Packages:**
```bash
# Core ML and Data Processing
pip install pandas==2.1.0
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.2

# Spotify Integration
pip install spotipy==2.22.1

# Language Detection
pip install langdetect==1.0.9
pip install polyglot
pip install pycld2

# Data Visualization (Optional)
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# HTTP Requests for Ollama
pip install requests==2.31.0

# Text Processing  
pip install nltk==3.8.1

# Progress Bars
pip install tqdm==4.66.1
```

**OR Install from Requirements File:**
```bash
# Save this as requirements.txt
echo "pandas>=2.1.0
numpy>=1.24.3
scikit-learn>=1.3.0
joblib>=1.3.2
spotipy>=2.22.1
langdetect>=1.0.9
polyglot
pycld2
matplotlib>=3.7.2
seaborn>=0.12.2
requests>=2.31.0
nltk>=3.8.1
tqdm>=4.66.1" > requirements.txt

# Install all at once
pip install -r requirements.txt
```

---

### 4. Spotify API Setup (Required for Playlist Creation)

1. **Go to Spotify Developer Dashboard:**
   - Visit: https://developer.spotify.com/dashboard
   - Log in with your Spotify account

2. **Create an App:**
   - Click **"Create an App"**
   - Fill in:
     - **App Name**: "AI Playlist Creator" (or your choice)
     - **Description**: "AI-powered playlist generator"
     - **Website**: http://localhost (or leave blank)

3. **Configure App Settings:**
   - Click **"Edit Settings"**
   - Add Redirect URI: **`http://127.0.0.1:8888/callback`**
   - Save settings

4. **Get Your Credentials:**
   - Copy your **Client ID** 
   - Copy your **Client Secret** (click "Show Client Secret")
   - **Keep these safe!** You'll enter them when running the app

---

### 5. Project File Structure

**Make sure you have these files in your project directory:**
```
📁 Music-Recommender-Project/
├── 📄 music_app_final.py               # Main application (FINAL VERSION)
├── 📄 tracks_added_languages.csv       # Your music dataset
├── 📄 DataCleaning.py                  # Data preprocessing
├── 📄 MusicRecommender.py              # AI recommendation engine  
├── 📄 PlaylistCreator.py               # Playlist generation logic
├── 📄 config.py                        # Configuration settings
├── 📄 requirements.txt                 # Python dependencies
├── 📄 setup-guide.md                   # This guide
├── 📁 models/                          # ML models directory
│   └── 📄 kmeans_k18.joblib           # (auto-created when first run)
├── 📁 data/                            # Optional: additional datasets
└── 📄 .spotify_cache                   # (auto-created after first auth)
```


### 6. Verify Installation

**Test Individual Components:**

1. **Test Python Environment:**
   ```bash
   python3 -c "import pandas, numpy, sklearn, spotipy, requests; print('✅ All packages installed')"
   ```

2. **Test Ollama:**
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "llama2",
     "prompt": "Test",
     "stream": false
   }'
   ```

3. **Test Dataset:**
   ```bash
   python3 -c "import pandas as pd; df = pd.read_csv('tracks_added_languages.csv'); print(f'✅ Dataset loaded: {len(df)} tracks')"
   ```

---

### 7. First Run

**Start the Application:**
```bash
# Make sure Ollama is running
ollama serve &

# Run the music app
python3 music_app_final.py
```

**What Happens on First Run:**
1. **System loads** your music dataset (~586K tracks)
2. **Data cleaning** and feature scaling 
3. **AI recommender initialization** with clustering
4. **Spotify credentials setup** (you'll be prompted)
5. **Interactive session starts**

---

## 🎮 How to Use

### Interactive Workflow:
1. **System loads** your music data and AI models
2. **Spotify authentication** (copy/paste URL method)
3. **Describe your playlist**: "energetic pop-rap for gym workouts"
4. **Choose playlist size**: 25, 50, or custom number
5. **Preview AI results** with similarity scores  
6. **Create playlist**:
   - On Spotify (automatic playlist creation)
   - Save locally (JSON file)  
   - Both options

### Example Session:
```
🎵 AI Music Recommendation System
==================================================
🚀 Starting AI Music Playlist Creator...

📊 Loading music dataset...
✅ Loaded 586672 tracks
🧹 Cleaning data...
✅ Cleaned data: 586601 tracks with 11 features
🤖 Initializing AI recommender...
Loading existing KMeans model from models/kmeans_k18.joblib
✅ System ready!

🔑 Spotify API Setup
Enter your Spotify Client ID: [your_client_id]
Enter your Spotify Client Secret: [your_client_secret]

🔐 Spotify Authentication Required
==================================================
📋 COPY THIS URL AND OPEN IT IN YOUR BROWSER:

https://accounts.spotify.com/authorize?client_id=...

👆 Copy the URL above and paste it into any web browser
🔑 Log in to Spotify and click 'Agree' to authorize the app
📥 Paste the full redirect URL here: [paste_redirect_url]

✅ Success! Authenticated as: YourName (@your_spotify_username)
✅ Ready to create playlists on Spotify!

🎤 Describe your ideal playlist!
🎯 What kind of playlist do you want? upbeat rap songs for workout

📏 How many songs? (default: 50): 30

🤖 Creating AI playlist for: 'upbeat rap songs for workout'
⏳ This may take a moment...

🎵 Generated Playlist Preview (30 songs total)
============================================================
   1. Lose Yourself
      by Eminem
      (similarity: 0.987)

   2. Till I Collapse
      by Eminem, Nate Dogg  
      (similarity: 0.934)

   3. Stronger
      by Kanye West
      (similarity: 0.912)
...

🎯 What would you like to do?
  1. Create on Spotify
  2. Save locally only
  3. Both
  4. Try again with different description
  5. Quit

Choice (1-5): 1

🎵 Creating playlist: 'Upbeat Rap Songs for Workout'
✅ Created empty playlist
🔗 Playlist URL: https://open.spotify.com/playlist/xxx

🔍 Finding 30 songs on Spotify...
   1. Lose Yourself - Eminem ✅
   2. Till I Collapse - Eminem, Nate Dogg ✅
   3. Stronger - Kanye West ✅
...

📥 Adding 28 songs to playlist...
🎉 SUCCESS! Added 28 out of 30 songs!
```

---

## 🎵 Spotify Integration Features

### Authentication Method:
- **Manual URL copy/paste** (works on all systems)
- **Automatic token caching** (no re-login needed)
- **Secure credential handling**

### Automatic Track Matching:
- **Smart search strategy** with fallbacks
- **Artist name cleaning** and formatting
- **Success rate reporting** (e.g., "28/30 tracks found")
- **Missing songs notification**

### Playlist Creation:
- **Real Spotify playlist creation**
- **Automatic track addition** (handles Spotify limits)
- **Direct playlist links**
- **Public playlist by default** (can be modified)

---

## 🔧 Configuration Options

### Core Settings in `music_app_final.py`:
```python
# AI Recommendation Settings
language_weight=0.2,    # Boost same-language songs  
artist_weight=0.1,      # Boost similar artists

# Default Settings
playlist_size = 50      # Can be changed interactively
max_songs = 100         # Maximum playlist size
min_songs = 10          # Minimum playlist size

# Ollama Settings  
ollama_url = "http://localhost:11434"
model_name = "llama2"   # or "phi" for smaller model
```

### Playlist Size Recommendations:
- **Quick playlist**: 15-25 songs
- **Standard playlist**: 30-50 songs  
- **Extended playlist**: 75-100 songs

---

## 🐛 Troubleshooting

### Common Issues and Solutions:

**"tracks_added_languages.csv not found"**
```bash
# Make sure the dataset is in your project directory
ls -la *.csv
# Should show: tracks_added_languages.csv
```

**Ollama Connection Failed**
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/version

# Install model if missing  
ollama pull llama2
```

**Spotify Authentication Issues**
- ✅ Use **exact redirect URI**: `http://127.0.0.1:8888/callback`  
- ✅ Copy **complete redirect URL** after authorization
- ✅ Paste **entire URL** starting with `http://127.0.0.1:8888/callback?code=...`
- ❌ Don't use `localhost` - use `127.0.0.1` 

**Python Package Conflicts**
```bash  
# Create clean environment
python3 -m venv fresh-music-env
source fresh-music-env/bin/activate
pip install -r requirements.txt
```

**No Songs Found on Spotify**
- Your dataset might need **artist name cleaning**
- Try with **popular/mainstream** requests first
- Check your **music dataset quality**

**Memory/Performance Issues**
```bash
# Monitor memory usage
python3 -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
# Recommends: 4+ GB RAM for full dataset
```

---

## 📊 System Architecture

```
User Input → Ollama LLM → AI Recommender → Spotify Integration
    ↓             ↓             ↓              ↓
"workout rap" → Seed Songs → Cosine Sim. → Real Playlist  
              → ["Eminem"]  → Similar     → Created!
```

### AI Pipeline Details:
1. **Natural Language Processing**: Ollama interprets user requests
2. **Seed Song Generation**: LLM suggests starting songs for recommendations  
3. **Cosine Similarity Matching**: Your trained model finds similar tracks (on the song's cluster for performance)
4. **Smart Filtering**: Language, genre, and quality filtering
5. **Deduplication**: Advanced duplicate removal
6. **Spotify Integration**: Automatic real playlist creation

---

## 🎉 Success Metrics

**A great AI playlist should have:**
- ✅ **High similarity scores** (0.8+ average)
- ✅ **Genre consistency** (rap songs for rap requests)  
- ✅ **Language consistency** (English for Western genres)
- ✅ **High Spotify match rate** (85%+ songs found)
- ✅ **No duplicates or near-duplicates**
- ✅ **Good variety within genre** (different artists/years)

---

## 🚀 Advanced Usage

### Custom Model Training:
```bash
# Retrain recommendation model with different parameters
python3 test_minibatch_elbow.py  # Find optimal clusters
# Modify k value in MusicRecommender.py
```

### Bulk Playlist Generation:
```python
# Create multiple playlists programmatically
requests = [
    "chill indie rock for studying",
    "energetic EDM for running", 
    "romantic pop songs for date night"
]
# Process each request...
```

### Dataset Expansion:
```bash
# Add new music data
python3 add_language_column.py  # Process new CSV
# Retrain models with expanded dataset
```

---

**🎵 Ready to create amazing AI playlists? Start with:**
```bash
ollama serve &
python3 music_app_final.py
```

**Enjoy your personalized AI-powered music experience!** 🎵