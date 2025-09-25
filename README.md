# AI-Powered Music Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated music recommendation engine that leverages a Large Language Model (LLM) and Cosine Similarity to generate personalized Spotify playlists from simple text descriptions.

This project demonstrates an end-to-end pipeline, from raw data processing and model training to final user interaction and API integration, showcasing a creative and effective approach to music discovery.

---

## 🌟 Core Features

*   **Natural Language Playlist Generation:** Describe the music you want (e.g., *"upbeat rap for a workout"*) and the system generates a playlist to match.
*   **LLM-Powered Seed Generation:** Uses a local LLM (like Llama 3 or Phi) to act as a music expert, generating a list of "seed songs" that form the basis for the recommendation.
*   **Cosine Similarity Recommender:** A highly efficient recommendation model that finds songs mathematically similar to the LLM-generated seeds from a database of over 580,000 tracks.
*   **Automatic Spotify Integration:** Authenticates with the Spotify API (OAuth 2.0) to automatically create and populate playlists in a user's account.
*   **Scalable Data Pipeline:** Processes and cleans a large song dataset, preparing it for the machine learning model.
*   **Interactive CLI:** A user-friendly command-line interface for a smooth user experience.

---

## ⚙️ How It Works (System Architecture)

The system follows a multi-stage pipeline to go from a user's text query to a Spotify playlist:

**User Input → Local LLM → Seed Songs → AI Recommender → Spotify API**

1.  **Natural Language Input:** The user provides a text prompt (e.g., "energetic pop-rap for gym workouts").
2.  **Seed Song Generation (LLM):** The prompt is sent to a local Ollama model (e.g., Llama 3, Phi). Through carefully crafted prompt engineering, the LLM acts as a music expert, recommending a list of real song names and artists (from tracks released **up to 2020**, to match the project's dataset) that fit the user's request.
3.  **Database Lookup:** The system takes the seed songs recommended by the LLM and locates their audio feature vectors within the project's 580k+ song database.
4.  **Cosine Similarity Search:** The audio features of the seed songs are used to find other songs in the database that are mathematically most similar. This creates a list of high-quality recommendations that align with the user's original intent.
5.  **Playlist Curation:** The top N matching songs are selected, filtered, and de-duplicated to create a final playlist.
6.  **Spotify Playlist Creation:** The system connects to the Spotify API, creates a new playlist, finds the corresponding tracks on Spotify, and adds them to the playlist.

---

## 🚀 Skills Showcase

This project demonstrates proficiency across the full spectrum of a modern data science and AI application:

### **Machine Learning & AI**
*   **Recommendation Engines:** Built a content-based filtering system using Cosine Similarity on 11-dimensional audio feature vectors to find songs with similar characteristics.
*   **Unsupervised Learning:** Implemented K-Means clustering to efficiently partition the song dataset, enabling faster and more targeted recommendations (used in an earlier version, with the foundation still in the code).
*   **Feature Engineering & Scaling:** Used `scikit-learn` to normalize song features, ensuring accurate similarity calculations.

### **Natural Language Processing (NLP)**
*   **LLM as a Creative Expert:** Instead of just classifying text, the LLM is used creatively as a "music expert." It interprets a user's abstract request and generates a concrete list of example songs, bridging the gap between user intent and the song database.
*   **Advanced Prompt Engineering:** The prompt includes constraints (e.g., songs up to 2020) to ensure the LLM's suggestions are compatible with the available dataset, demonstrating a sophisticated use of prompt design to control model output.

### **Data Science & Engineering**
*   **Large-Scale Data Processing:** Managed and cleaned a dataset of over 580,000 tracks using `pandas` and `numpy`.
*   **Data Cleaning:** Handled missing values and prepared raw data for the machine learning pipeline.

### **API Integration & Web Services**
*   **REST APIs:** Mastered interaction with the Spotify Web API.
*   **Authentication:** Implemented the OAuth 2.0 Authorization Code Flow to securely authenticate users and gain permission to create playlists on their behalf.
*   **Web Requests:** Used the `requests` library to communicate with the local Ollama API.

### **Software Engineering**
*   **Object-Oriented Programming (OOP):** Designed a modular system with distinct classes for `DataCleaner`, `MusicRecommender`, and `PlaylistCreator`, promoting code reusability and maintainability.
*   **CLI Design:** Built an intuitive and interactive command-line interface.
*   **Dependency Management:** Provided a `requirements.txt` file and a virtual environment setup for reproducible installations.

---

## 🛠️ Tech Stack

*   **Core Language:** Python
*   **Machine Learning:** Scikit-learn, NumPy
*   **Data Manipulation:** Pandas
*   **NLP:** Ollama (with Llama 3, Phi, and other models)
*   **API Interaction:** Spotipy, Requests
*   **Tooling:** Joblib, NLTK, Langdetect

---

## 🏁 Getting Started

### Prerequisites

*   Python 3.8+
*   Ollama installed and a model pulled (e.g., `ollama pull llama3`)
*   A Spotify Developer account and App credentials (Client ID & Secret)

### Installation & Usage

For detailed instructions on setup, dependencies, and first-time use, please refer to the complete **[setup-guide.md](setup-guide%20(1).md)**.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/omermadar/AI-Spotify-Playlist-App.git
    cd AI-Spotify-Playlist-App
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    # Make sure the Ollama service is running in the background
    ollama serve &

    # Start the main application
    python3 music_app_final.py
    ```

---

## 📄 License

This project is licensed under the MIT License.

---
*This README was generated with the assistance of Google's Gemini.*