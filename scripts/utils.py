# scripts/utils.py
import spacy

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]
