# scripts/process_data.py

import spacy
import json
import re
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Determine the base directory (one level up from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(base_dir, 'data')

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

def clean_sentence(sentence):
    """Clean and preprocess a sentence."""
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove extra whitespace
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    sentence = sentence.lower()  # Convert to lowercase
    sentence = sentence.strip()  # Remove leading/trailing whitespace
    return sentence

def generate_response(cluster_sentences):
    """Generate a response based on the most common words in the cluster."""
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(cluster_sentences)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_words = [word for word, freq in words_freq[:5]]
    if top_words:
        response = f"Our {', '.join(top_words)} related programs offer excellent opportunities for students."
    else:
        response = "We offer a wide range of programs to suit your educational needs."
    return response

def process_scraped_data():
    """Process scraped data to generate intents automatically."""
    try:
        with open(os.path.join(data_dir, 'scraped_data.txt'), 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print("Error: 'scraped_data.txt' not found. Please run 'scrape_website.py' first.")
        return

    # Tokenize into sentences using SpaCy
    doc = nlp(data)
    sentences = [sent.text for sent in doc.sents]

    # Clean sentences
    sentences = [clean_sentence(sentence) for sentence in sentences]

    # Remove duplicates and very short sentences
    sentences = list(set(sentences))
    sentences = [s for s in sentences if len(s.split()) > 3]

    print(f"Total unique sentences after cleaning: {len(sentences)}")

    # Vectorize sentences using TF-IDF
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=1000)
    X = vectorizer.fit_transform(sentences)

    # Determine number of clusters (you can implement Elbow Method here)
    num_clusters = 5  # Adjust based on your data

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Assign sentences to clusters
    clustered_sentences = {}
    for idx, label in enumerate(labels):
        if label not in clustered_sentences:
            clustered_sentences[label] = []
        clustered_sentences[label].append(sentences[idx])

    # Create intents based on clusters
    intents = []
    for label, cluster_sentences in clustered_sentences.items():
        response = generate_response(cluster_sentences)
        intent = {
            "tag": f"intent_{label+1}",
            "patterns": cluster_sentences,
            "responses": [
                response,
                f"For more information on our {', '.join(response.split()[1:3])}, please visit our website."
            ]
        }
        intents.append(intent)

    # Save intents to a JSON file in the base data directory
    intents_file = os.path.join(data_dir, 'intents.json')
    with open(intents_file, 'w', encoding='utf-8') as f:
        json.dump(intents, f, indent=4)

    print(f"Data processing completed. Intents saved to {intents_file}")

if __name__ == '__main__':
    process_scraped_data()
