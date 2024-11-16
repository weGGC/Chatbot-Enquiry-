# scripts/utils.py

import spacy

# Load SpaCy English model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    raise RuntimeError("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm' to install.")

def spacy_tokenizer(text):
    """
    Tokenizes and preprocesses text using SpaCy.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        List[str]: Lemmatized tokens without stop words or punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    # Process the text using SpaCy
    doc = nlp(text)
    # Return lemmatized tokens excluding stop words and punctuation
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def preprocess_text(text):
    """
    Additional preprocessing for text, such as lowercasing and cleaning.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    # Lowercase and clean text
    text = text.lower()
    text = " ".join(spacy_tokenizer(text))
    return text

if __name__ == "__main__":
    # Example usage for testing
    sample_text = "Hello! How can I assist you today?"
    print("Original Text:", sample_text)
    print("Tokenized Text:", spacy_tokenizer(sample_text))
    print("Preprocessed Text:", preprocess_text(sample_text))
