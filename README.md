# 🎬 Movie Recommendation System

A machine learning-powered Movie Recommendation Web App that suggests similar movies based on the one you select. Built using `TfidfVectorizer`, `Cosine Similarity`, and `Streamlit`. Think of it as a mini Netflix clone — built from scratch!

---

## 🔍 Features

- Select any movie from a dropdown
- Get top 10 similar movie recommendations
- Genre info displayed in stylish cards
- Responsive dark-themed UI using Streamlit
- Fully content-based filtering approach

---

## 🧠 How It Works

1. **Data Preparation:**  
   Combined important fields like title, genres, keywords, cast, and crew into a single `tags` column.

2. **Text Vectorization:**  
   Applied `TfidfVectorizer` to convert text data into numerical vectors, giving less weight to common words.

3. **Similarity Calculation:**  
   Used **cosine similarity** to compute how close two movies are based on the TF-IDF vectors.

4. **Frontend:**  
   Streamlit was used to create a clean, interactive web interface with columns and card layouts.

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (TfidfVectorizer, Cosine Similarity)
- Streamlit
- Pickle (for model and data serialization)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/WebberTiwari/movie_recommender_ML.git

# Navigate to the project directory
cd movie_recommender_ML

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

📂 Folder Structure
movie-recommender-ml/
│
├── app.py                 # Main Streamlit app
├── movies.pkl             # Preprocessed movie data
├── model.pkl              # Cosine similarity matrix
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation


## 💡 What I Learned

- Gained hands-on experience with how real-world **recommender systems** work under the hood.
- Understood and applied **TF-IDF vectorization** and **cosine similarity** for content-based filtering.
- Practiced **feature engineering** using text data (title, genres, keywords, cast, crew).
- Learned how to build interactive ML-based web apps using **Streamlit**.
- Took the project from **backend ML logic to frontend UI**, covering the full ML workflow.

Priyanshu Tiwari
📍 3rd Year B.Tech (Electrical), NIT Silchar
