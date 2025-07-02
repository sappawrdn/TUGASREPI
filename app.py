import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv("clustered_df.csv")
overview_tfidf = pickle.load(open("overview_tfidf.pkl", 'rb'))

def recommend_movies(title, df, num_recommendations=6):
    if title not in df['Series_Title'].values:
        return "Movie not found in dataset."

    cluster_label = df[df['Series_Title'] == title]['Cluster'].values[0]
    cluster_movies = df[df['Cluster'] == cluster_label]
    movie_vector = overview_tfidf[df[df['Series_Title'] == title].index[0]]
    similarities = cosine_similarity(movie_vector, overview_tfidf[cluster_movies.index]).flatten()
    similar_indices = similarities.argsort()[-(num_recommendations + 1):-1][::-1]
    recommendations = cluster_movies.iloc[similar_indices][['Series_Title', 'Overview', 'IMDB_Rating', 'Poster_Link']]
    return recommendations.reset_index(drop=True)

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender Vibes", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #ff4b4b;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 2em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff7878;
        color: white;
    }
    .movie-container {
        margin-bottom: 2rem;
    }
    .movie-poster {
        width: 100%;
        border-radius: 12px;
        object-fit: cover;
        display: block;
        transition: 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .movie-title {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5em;
        color: #ffffff;
        text-align: center;
    }
    .movie-rating {
        font-size: 0.9rem;
        color: #ffb703;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .movie-overview {
        font-size: 0.85rem;
        color: #cccccc;
        line-height: 1.3em;
        text-align: center;
        margin-top: 0.3em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)





st.markdown('<div class="main-title">🍿 Movie Recommendation System</div>', unsafe_allow_html=True)

movie_list = df['Series_Title'].sort_values().unique()
selected_movie = st.selectbox("🎥 Choose your favorite movie first:", movie_list)

if st.button("✨ Recommend me similar movies"):
    output = recommend_movies(selected_movie, df)

    if isinstance(output, str):
        st.error(output)
    else:
        for i in range(0, len(output), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(output):
                    row = output.iloc[i + j]
                    with cols[j]:
                        with cols[j]:
                            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                            st.markdown(f'<img src="{row["Poster_Link"]}" class="movie-poster"/>', unsafe_allow_html=True)
                            st.markdown(f"<div class='movie-title'>🎬 {row['Series_Title']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='movie-rating'>⭐ {row['IMDB_Rating']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='movie-overview'>{row['Overview'][:100]}...</div>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)


