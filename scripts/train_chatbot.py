import json
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
from utils import preprocess_text, spacy_tokenizer  # Import utility functions

# Determine the base directory
script_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Define paths
intents_file = os.path.join(base_dir, 'data', 'intents.json')
model_dir = os.path.join(base_dir, 'models')
model_file = os.path.join(model_dir, 'chatbot_model.pkl')

# Load SpaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    raise RuntimeError("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")

def train_chatbot_model():
    """
    Train a chatbot model using intents from the intents.json file.
    """
    # Load intents
    try:
        with open(intents_file, 'r', encoding='utf-8') as f:
            intents = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{intents_file}' not found. Please ensure the intents file is available.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding failed for '{intents_file}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading intents: {e}")
        return

    # Prepare training data
    tags = []
    patterns = []

    for intent in intents:
        for pattern in intent['patterns']:
            if not pattern.strip():
                print(f"Skipping empty pattern in intent: {intent['tag']}")
                continue  # Skip empty patterns
            try:
                cleaned_pattern = preprocess_text(pattern)
                if cleaned_pattern:  # Only add non-empty cleaned patterns
                    patterns.append(cleaned_pattern)
                    tags.append(intent['tag'])
                else:
                    print(f"Skipping pattern after cleaning (empty result): '{pattern}'")
            except Exception as e:
                print(f"Error processing pattern '{pattern}' in intent '{intent['tag']}': {e}")

    if not patterns:
        print("Error: No valid patterns found. Ensure your intents file contains valid patterns.")
        return

    # Check number of unique tags
    unique_tags = set(tags)
    if len(unique_tags) < 2:
        print("Error: Need at least two distinct intents to train the model.")
        return

    # Vectorize patterns using TfidfVectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        stop_words="english",  # Remove common English stopwords
        max_features=5000,  # Use a larger feature set for better accuracy
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )

    try:
        X = vectorizer.fit_transform(patterns)
        y = tags
    except ValueError as e:
        print(f"Error during vectorization: {e}")
        return

    # Split the data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression classifier
    clf = LogisticRegression(random_state=0, max_iter=1000, solver='liblinear')
    clf.fit(X_train, y_train)

    # Validate the model
    y_pred = clf.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Ensure the models directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained model and vectorizer
    try:
        with open(model_file, 'wb') as f:
            pickle.dump((vectorizer, clf, intents), f)
        print("\nModel training completed successfully. Model saved to:", model_file)
    except Exception as e:
        print(f"Error saving the model: {e}")

if __name__ == '__main__':
    print("Training Chatbot Model...")
    train_chatbot_model()
    print("Training Completed.")
