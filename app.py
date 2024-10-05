import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the CSV file (Update file path if necessary)
file_path = 'Hindi_Songs_With_Tags_And_Links_.xlsx'
songs_df = pd.read_excel(file_path)

# Fill NaN values with empty strings to prevent errors in text processing
songs_df['Genre'] = songs_df['Genre'].fillna('')
songs_df['Singer Name'] = songs_df['Singer Name'].fillna('')
songs_df['Type'] = songs_df['Type'].fillna('')
songs_df['Tags'] = songs_df['Tags'].fillna('')

# Combine relevant features into a single column for comparison, including Tags
songs_df['combined_features'] = songs_df['Genre'] + " " + songs_df['Singer Name'] + " " + songs_df['Type'] + " " + songs_df['Tags']

# Vectorize the text data (convert text to numerical form)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(songs_df['combined_features'])

# Compute cosine similarity between songs
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get song recommendations based on song name, artist, genre, or tags
def get_recommendations(query, cosine_sim=cosine_sim):
    mask = (songs_df['Song Name'].str.contains(query, case=False, na=False) |
            songs_df['Singer Name'].str.contains(query, case=False, na=False) |
            songs_df['Genre'].str.contains(query, case=False, na=False) |
            songs_df['Type'].str.contains(query, case=False, na=False) |
            songs_df['Tags'].str.contains(query, case=False, na=False))
    
    if mask.any():
        indices = songs_df[mask].index
        sim_scores = []
        for idx in indices:
            sim_scores += list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(list(set(sim_scores)), key=lambda x: x[1], reverse=True)[:25]
        song_indices = [i[0] for i in sim_scores]
        return songs_df[['Song Name', 'Singer Name', 'Type', 'Genre', 'Tags', 'Link']].iloc[song_indices]
    else:
        return "Sorry, no song, artist, genre, or tags found matching that query."

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f3f4f6; /* Tailwind gray-100 */
            color: #374151; /* Tailwind gray-800 */
        }
        .stButton>button {
            background-color: #3b82f6; /* Tailwind blue-500 */
            color: white;
        }
        .stButton>button:hover {
            background-color: #2563eb; /* Tailwind blue-600 */
        }
        .stTextInput>div>input {
            border-color: #d1d5db; /* Tailwind gray-300 */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit Chatbot Interface
def chatbot():
    st.title('Song Recommendation')

    user_input = st.text_input("Enter a song name, genre, artist, type, or tags:")

    if user_input:
        st.write(f"Looking for recommendations related to: **{user_input}**")
        recommendations = get_recommendations(user_input)

        if isinstance(recommendations, pd.DataFrame):
            for index, row in recommendations.iterrows():
                st.write(f"**Song Name**: {row['Song Name']}")
                st.write(f"**Singer Name**: {row['Singer Name']}")
                st.write(f"**Type**: {row['Type']}")
                st.write(f"**Genre**: {row['Genre']}")
                st.write(f"[Listen here]({row['Link']})")
                st.write("---")
        else:
            st.write(recommendations)

if __name__ == "__main__":
    chatbot()
