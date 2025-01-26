
# Import required libraries for project
import streamlit as st # For user interface
import joblib # Loading the save model and vectorizer
import string # Text preprocessing (punctuation removal)
import nltk # Stopwords content
from nltk.corpus import stopwords
import os # File path manipulation
import subprocess
import sys

# Define directory path
proj_dir = '/Users/nicolassarmiento/Desktop/IMDB PROJECT'

# Load the model and vectorizer
model_path = os.path.join(proj_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(proj_dir, 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function for data preprocessing
def preprocess_text(text):

    text = text.lower() # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation from text
    words = text.split() # Tokenize into individual words
    words = [word for word in words if word not in stop_words] # Remove stop words
    return ' '.join(words) # Rebuild in a single string

# Function for running web app
def run_app():

    # Create streamlit web app interface
    st.title('Movie Review Sentiment Analysis')
    st.subheader('Enter a movie review to determine its sentiment')

    # User sentiment input
    user_input = st.text_area('Movie Review', '')

    if st.button('Analyze Sentiment'):
        if user_input:
            processed_input = preprocess_text(user_input)
            transformed_input = vectorizer.transform([processed_input])
            prediction = model.predict(transformed_input)[0]
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            st.write(f'**Sentiment:** {sentiment}') # Display prediction result
        else:
            st.write('Please enter a movie review!') # Display re-enter prompt

# Make script run directly
if __name__ == '__main__':
    if os.getenv('RUN_AS_STREAMLIT', 'false') == 'false':

        script_path = os.path.abspath(__file__)
        os.environ['RUN_AS_STREAMLIT'] = 'true'
        subprocess.run(['streamlit', 'run', script_path])
    else:
        run_app()