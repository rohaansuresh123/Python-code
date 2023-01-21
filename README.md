# Python-code
News classification using NLP
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the categories
categories = ['business', 'entertainment', 'politics', 'sport', 'technology']

# Define the training data
train_data = [
    {'text': 'Business deal signed between companies A and B', 'category': 'business'},
    {'text': 'New movie to be released next month', 'category': 'entertainment'},
    {'text': 'Political campaign ends in a scandal', 'category': 'politics'},
    {'text': 'Team wins championship game', 'category': 'sport'},
    {'text': 'Latest technology in smartphones', 'category': 'technology'},
]

# Define the test data
test_data = [
    {'text': 'Business merger announced', 'category': 'business'},
    {'text': 'Celebrity spotted at restaurant', 'category': 'entertainment'},
    {'text': 'Politician accused of corruption', 'category': 'politics'},
    {'text': 'Team loses final game', 'category': 'sport'},
    {'text': 'New app to revolutionize the industry', 'category': 'technology'},
]

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LinearSVC()),
])

# Train the model
train_texts = [item['text'] for item in train_data]
train_categories = [item['category'] for item in train_data]
pipeline.fit(train_texts, train_categories)

# Test the model
test_texts = [item['text'] for item in test_data]
test_categories = [item['category'] for item in test_data]
predictions = pipeline.predict(test_texts)

# Print the results
for prediction, actual in zip(predictions, test_categories):
    print(f'Predicted: {prediction}, Actual: {actual}')

This is a basic python code for News classification using Natural language processing using python.
