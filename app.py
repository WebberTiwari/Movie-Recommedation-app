import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    df.fillna('', inplace=True)  # handle any missing values
    return df

# --- Train Model ---
@st.cache_resource
def train_model(data):
    # Combine relevant text columns — adjust these as needed!
    data['combined_features'] = data['title'] + ' ' + data['genres'] + ' ' + data['keywords']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])

    similarity = cosine_similarity(tfidf_matrix)
    return similarity, data

# --- Recommendation Logic ---
def recommend(movie_name, similarity, data):
    movie_name = movie_name.lower()
    indices = data[data['title'].str.lower() == movie_name].index

    if len(indices) == 0:
        return ["Movie not found."], []

    idx = indices[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar

    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = data['title'].iloc[movie_indices].tolist()
    recommended_links = data['link'].iloc[movie_indices].tolist()

    return recommended_titles, recommended_links

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Content-Based Movie Recommendation System")

movies = load_data()
similarity, movies_data = train_model(movies)

movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    recommendations, links = recommend(movie_name, similarity, movies_data)

    if recommendations[0] == "Movie not found.":
        st.warning("Movie not found. Please check the spelling.")
    else:
        st.write("### Recommended Movies:")
        for i, (title, link) in enumerate(zip(recommendations, links), start=1):
            if link.strip():  # only make clickable if link exists
                st.markdown(f"{i}. [{title}]({link})")
            else:
                st.write(f"{i}. {title}")
