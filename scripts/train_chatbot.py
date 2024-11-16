# scripts/train_chatbot.py

import json
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
import pickle
import os
import contractions
import re
from utils import spacy_tokenizer  # Import the tokenizer

# Determine the base directory
script_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Define paths
intents_file = os.path.join(base_dir, 'data', 'intents.json')
model_dir = os.path.join(base_dir, 'models')
model_file = os.path.join(model_dir, 'chatbot_model.pkl')

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[\W_]', ' ', text)  # Remove punctuation and replace with space
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

def train_chatbot_model():
    # Load intents
    try:
        with open(intents_file, 'r', encoding='utf-8') as f:
            intents = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{intents_file}' not found. Please run 'process_data.py' first.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{intents_file}' contains invalid JSON.")
        return

    # Prepare training data
    tags = []
    patterns = []

    for intent in intents:
        for pattern in intent['patterns']:
            cleaned_pattern = preprocess_text(pattern)
            patterns.append(cleaned_pattern)
            tags.append(intent['tag'])

    # Check number of unique tags
    unique_tags = set(tags)
    if len(unique_tags) < 2:
        print("Error: Need at least two distinct intents to train the model.")
        return

    # Extend stop words and convert to list
    extended_stop_words = list(ENGLISH_STOP_WORDS.union({'ca', "n't"}))

    # Vectorize patterns using SpaCy tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        stop_words=extended_stop_words,
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )

    X = vectorizer.fit_transform(patterns)
    y = tags

    # Train classifier
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(X, y)

    # Ensure the models directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained model and vectorizer
    with open(model_file, 'wb') as f:
        pickle.dump((vectorizer, clf, intents), f)

    print("Model training completed successfully.")

if __name__ == '__main__':
    train_chatbot_model()
