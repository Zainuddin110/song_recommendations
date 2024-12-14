import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from urllib.parse import urlparse, parse_qs

# Load the song dataset
file_path = 'Hindi_Songs_With_Tags_And_Links_.xlsx'  # Replace with your dataset file path
songs_df = pd.read_excel(file_path)

# Fill NaN values with empty strings to prevent errors in text processing
songs_df['Genre'] = songs_df['Genre'].fillna('')
songs_df['Artist'] = songs_df['Artist'].fillna('')
songs_df['Language'] = songs_df['Language'].fillna('')

# Combine relevant features into a single column for comparison
songs_df['combined_features'] = (
    songs_df['Genre'] + " " +
    songs_df['Artist'] + " " +
    songs_df['Language']
)

# Vectorize the text data (convert text to numerical form)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(songs_df['combined_features'])

# Compute cosine similarity between songs
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get song recommendations
def get_recommendations(query, cosine_sim=cosine_sim):
    # Fill NaN values with empty strings
    songs_df.fillna('', inplace=True)

    # Create a mask for matching the query across relevant columns
    mask = (
        songs_df['Genre'].str.contains(query, case=False, na=False) |
        songs_df['Artist'].str.contains(query, case=False, na=False) |
        songs_df['Language'].str.contains(query, case=False, na=False) |
        songs_df['Song Name'].str.contains(query, case=False, na=False)
    )

    if mask.any():
        indices = songs_df[mask].index
        sim_scores = []

        # Calculate similarity scores for matching rows
        for idx in indices:
            sim_scores += list(enumerate(cosine_sim[idx]))

        # Prioritize exact matches by boosting their scores
        exact_match_mask = songs_df['SongName'].str.contains(query, case=False, na=False)
        exact_match_indices = songs_df[exact_match_mask].index

        # Add exact matches with a high score (e.g., 1.5) to prioritize them
        for idx in exact_match_indices:
            sim_scores.append((idx, 1.5))  # Boost exact matches

        # Remove duplicates and sort by similarity score
        sim_scores = sorted(list(set(sim_scores)), key=lambda x: x[1], reverse=True)[:50]
        song_indices = [i[0] for i in sim_scores]

        top_results = songs_df.iloc[song_indices]
        return top_results[['SongNameAndCode', 'Genre', 'Artist', 'Language', 'Song Link']]
    else:
        return "Sorry, no song found matching that query. Please try another."

# Streamlit App UI and Logic
def song_recommendation_system():
    st.title('Song Recommendation System')

    # Extract query parameters from the URL
    query_params = parse_qs(urlparse(st.experimental_get_url()).query)
    user_input = query_params.get('query', [''])[0]  # Default to an empty string if no query parameter

    # Debug: Print query to check if it's captured
    st.write("Debug: Captured query:", user_input)

    if user_input:
        # Display the query being processed
        st.write(f"Looking for recommendations related to: **{user_input}**")

        # Get recommendations based on the query
        recommendations = get_recommendations(user_input)

        # Display recommendations
        display_recommendations(recommendations)
    else:
        st.write("No query provided. Please enter a search term in the search bar.")

# Function to display the recommendations in a table with clickable links
def display_recommendations(recommendations):
    if isinstance(recommendations, pd.DataFrame):
        entries_per_page = st.selectbox('Select number of entries to display:', options=[10, 25, 50], index=0)
        limited_recommendations = recommendations.head(entries_per_page)
        # Format the 'Song Link' column to make links clickable
        if 'Song Link' in recommendations.columns:
            recommendations['Song Link'] = recommendations['Song Link'].apply(
                lambda link: f'<a href="{link}" target="_blank">Listen Now</a>' if pd.notnull(link) else ''
            )

        # Display the data frame with clickable links
        st.write(recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write(recommendations)

# Run the song recommendation system
if __name__ == "__main__":
    song_recommendation_system()
