
# Import required libraries for project
import numpy as np # Numerical computations
import pandas as pd # Loading and manipulating datasets
from sklearn.model_selection import train_test_split # Split data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer # Convert text into numerical features
from sklearn.linear_model import LogisticRegression # Machine learning model for classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Model evaluation (confusion matrix)
import nltk # Toolkit for text processing
from nltk.corpus import stopwords # Common words for removal
import string # Removing punctuation from strings
import joblib # Saving and loading model and vectorizer
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Styling confusion matrix visuals
import os # File path manipulation

# Define directory path
proj_dir = '/Users/nicolassarmiento/Desktop/IMDB PROJECT'

# Define dataset file path
dataset_path = '/Users/nicolassarmiento/Desktop/IMDB PROJECT/IMDB Dataset.csv'

# Load the required datasets to work with
df = pd.read_csv(dataset_path)

'''
The dataset contains two (2) columns, one with the text review and another with
the sentiment (positive or negative)
'''

# Preprocess the data
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) # Look only English stop words

# Function for data preprocessing
def preprocess_text(text):

    text = text.lower() # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation from text
    words = text.split() # Tokenize into individual words
    words = [word for word in words if word not in stop_words] # Remove stop words
    return ' '.join(words) # Rebuild in a single string

# Apply preprocessing function to the 'review' column
df['review'] = df['review'].apply(preprocess_text)

# Assign labels for model training
X = df['review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0) # Convert positive to 1 and negative to 0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # Testing size takes 20% of the dataset, training takes the remaining
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train) # Transform training data
X_test_tfidf = vectorizer.transform(X_test) # Transform test data using the same vocabulary

# Train the logistic regression model
model = LogisticRegression() # Logistic regression initialization
model.fit(X_train_tfidf, y_train) # Train model on training data

# Predictions based on the test data
y_pred = model.predict(X_test_tfidf) # Predict the sentiment based on test data

print('Accuracy:', accuracy_score(y_test, y_pred)) # Accuracy score calculated
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model and vectorizer in project directory
model_path = os.path.join(proj_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(proj_dir, 'tfidf_vectorizer.pkl') 

# Save both model and vectorizer to files in the directory
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved at: {model_path}")
print(f"Vectorizer saved at: {vectorizer_path}")