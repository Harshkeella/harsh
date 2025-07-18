import streamlit as st
import pandas as pd
import pickle
import requests
import os
api_key = os.getenv('TMDB_API_KEY')

def fetch_posters(movie_id):
    try:
        response = requests.get(
            'https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(
                movie_id))
        data = response.json()

        # Check if poster_path exists and is not None
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    except Exception as e:
        st.error(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"


def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_movie_posters = []

        for i in movies_list:
            movie_idx = i[0]
            # Get the actual movie_id from the dataframe (assuming you have a movie_id column)
            # If you don't have movie_id column, you'll need to use the index or add movie_id data
            if 'movie_id' in movies.columns:
                movie_id = movies.iloc[movie_idx]['movie_id']
            else:
                movie_id = movie_idx  # Fallback to using index as movie_id

            recommended_movies.append(movies.iloc[movie_idx]['title'])
            recommended_movie_posters.append(fetch_posters(movie_id))

        return recommended_movies, recommended_movie_posters
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return [], []


# Load data with error handling
try:
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a Movie',
    movies['title'].values
)

if st.button('Recommend Movies'):
    names, posters = recommend(selected_movie_name)

    if names and posters:
        # Create 5 columns for 5 recommendations
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(names[0])
            st.image(posters[0], width=150)
        with col2:
            st.text(names[1])
            st.image(posters[1], width=150)
        with col3:
            st.text(names[2])
            st.image(posters[2], width=150)
        with col4:
            st.text(names[3])
            st.image(posters[3], width=150)
        with col5:
            st.text(names[4])
            st.image(posters[4], width=150)
    else:
        st.error("Could not generate recommendations. Please try again.")