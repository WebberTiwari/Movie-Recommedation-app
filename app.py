import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
    .movie-card {
        background-color: #f9f9f9;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .movie-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .movie-link {
        font-size: 16px;
        color: #1f77b4;
        text-decoration: none;
    }
    .movie-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    df.fillna('', inplace=True)
    return df

# --- Train Model ---
@st.cache_resource
def train_model(data):
    data['combined_features'] = data['title'] + ' ' + data['genres'] + ' ' + data['keywords']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity, data

# --- Recommend Function ---
def recommend(movie_name, similarity, data):
    movie_name = movie_name.lower()
    indices = data[data['title'].str.lower() == movie_name].index

    if len(indices) == 0:
        return ["Movie not found."], []

    idx = indices[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = data['title'].iloc[movie_indices].tolist()
    recommended_homepages = data['homepage'].iloc[movie_indices].tolist()
    return recommended_titles, recommended_homepages

# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Find similar movies instantly! Enter a movie title below:")

movies = load_data()
similarity, movies_data = train_model(movies)

movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    recommendations, homepages = recommend(movie_name, similarity, movies_data)

    if recommendations[0] == "Movie not found.":
        st.warning("Movie not found. Please check the spelling.")
    else:
        st.subheader("✨ Recommended Movies:")
        fallback_url = "https://yourwebsite.com/movies"

        for i, (title, homepage) in enumerate(zip(recommendations, homepages), start=1):
            url = homepage.strip() if homepage.strip() else fallback_url
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{i}. {title}</div>
                    <a class="movie-link" href="{url}" target="_blank">Go to Movie Homepage</a>
                </div>
            """, unsafe_allow_html=True)
