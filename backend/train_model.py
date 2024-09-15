import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Use the correct column names from your CSV
data = data[['Category', 'Message']]  # Adjusted based on your CSV
data.columns = ['label', 'text']  # Rename for consistency

# Convert labels to binary (spam: 1, ham: 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model for later use
joblib.dump(pipeline, 'spam_classifier_model.pkl')

print("Model trained and saved as spam_classifier_model.pkl")
