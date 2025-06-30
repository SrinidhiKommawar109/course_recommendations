import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
# Add a background image using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("bg.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset with error handling
try:
    df = pd.read_csv('udemy_course_data.csv')
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("Dataset file 'udemy_course_data.csv' not found. Please check the file path.")
    st.stop()

# Data preprocessing
if 'published_time' in df.columns:
    df['published_time'] = df['published_time'].fillna(df['published_time'].mode()[0])
else:
    st.warning("Column 'published_time' not found in dataset.")

if 'course_title' in df.columns:
    df['course_title'] = df['course_title'].fillna('')
else:
    st.warning("Column 'course_title' not found in dataset.")

# Convert content_duration to numeric format with verification
if 'content_duration' in df.columns:
    def extract_numeric_duration(duration):
        if pd.notnull(duration):
            match = re.search(r'\d+', str(duration))
            return float(match.group()) if match else None
        return None

    df['numeric_duration'] = df['content_duration'].apply(extract_numeric_duration)
    df = df.dropna(subset=['numeric_duration'])
else:
    st.warning("Column 'content_duration' not found in dataset.")
    st.stop()

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['course_title'])

# Recommendation function with error handling
def get_recommendations_for_word(word, max_duration, is_paid=None, level=None):
    if not word:
        return "Please enter a keyword to search for courses."
    
    word_vector = tfidf.transform([word])
    cosine_sim = linear_kernel(word_vector, tfidf_matrix).flatten()
    
    similarity_threshold = 0.1
    sim_scores_indices = [i for i in cosine_sim.argsort()[::-1] if cosine_sim[i] > similarity_threshold]

    top_courses = df.iloc[sim_scores_indices]

    # Apply max duration filter
    top_courses = top_courses[top_courses['numeric_duration'] <= max_duration]
    
    # Apply is_paid filter
    if is_paid is not None and 'is_paid' in df.columns:
        top_courses = top_courses[top_courses['is_paid'] == is_paid]
    elif 'is_paid' not in df.columns:
        st.warning("Column 'is_paid' not found in dataset. Filtering by course type will be skipped.")
    
    # Apply level filter
    if level is not None and 'level' in df.columns:
        top_courses = top_courses[top_courses['level'].str.lower() == level.lower()]
    elif 'level' not in df.columns:
        st.warning("Column 'level' not found in dataset. Level filtering will be skipped.")

    if top_courses.empty:
        return "No courses found with the specified criteria."
    
    # Return the top 10 courses with the link included
    return top_courses[['course_title', 'content_duration', 'is_paid', 'level', 'url']].head(10)

# Streamlit UI
st.title("Course Recommendation System")

# User inputs
word = st.text_input("Enter a keyword (e.g., 'Web development'):")
daily_hours = st.number_input("How many hours can you spend per day?", min_value=1, value=2)
num_days = st.number_input("How many days do you have?", min_value=1, value=5)

# Calculate the maximum duration in hours
max_duration = daily_hours * num_days

is_paid = st.selectbox("Course Type:", ["Both", "Paid", "Free"])
level = st.selectbox("Course Level:", ["All Levels", "Beginner Level", "Intermediate Level", "Expert Level"])

# Convert user input for is_paid to a boolean if necessary
is_paid = None if is_paid == "Both" else (is_paid == "Paid")
level = None if level == "All Levels" else level

# Recommend button
if st.button("Get Recommendations"):
    recommendations = get_recommendations_for_word(word, max_duration, is_paid, level)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("Top Recommended Courses:")
        
        # Format the output as a DataFrame for a cleaner table view
        recommendations['url'] = recommendations['url'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')
        st.write(recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)
