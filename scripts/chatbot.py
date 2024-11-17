# scripts/chatbot.py

import spacy
import random
import pickle
import os

# Load SpaCy English model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    raise RuntimeError("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")

def spacy_tokenizer(text):
    """
    Tokenizes and preprocesses input text using SpaCy.

    Args:
        text (str): Input text to tokenize.

    Returns:
        List[str]: Tokenized text.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

# Determine the base directory
script_dir = os.path.dirname(__file__)
model_file = os.path.join(script_dir, '..', 'models', 'chatbot_model.pkl')

# Load the trained model and data
def load_model():
    """
    Loads the trained chatbot model.

    Returns:
        tuple: Vectorizer, classifier, and intents data.
    """
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            if len(model_data) != 3:
                raise ValueError("Model file is corrupted or incomplete.")
            return model_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{model_file}' not found. Please run 'train_chatbot.py' first.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {e}")

vectorizer, clf, intents = load_model()

def chatbot_response(user_input):
    """
    Generates a response based on user input.

    Args:
        user_input (str): User's message.

    Returns:
        str: Chatbot's response.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return "I didn't catch that. Could you please rephrase?"

    try:
        cleaned_input = ' '.join(spacy_tokenizer(user_input))
        user_input_processed = vectorizer.transform([cleaned_input])
        predicted_tag = clf.predict(user_input_processed)[0]
        predicted_probabilities = clf.predict_proba(user_input_processed)
        max_proba = max(predicted_probabilities[0])

        # Debugging: Print the confidence and predicted tag
        print(f"Predicted Tag: {predicted_tag}, Confidence: {max_proba}")

        # Set a confidence threshold
        CONFIDENCE_THRESHOLD = 0.03
        if max_proba < CONFIDENCE_THRESHOLD:
            return "I'm not sure I understand. Could you rephrase or ask about another topic?"

        for intent in intents:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])

    except Exception as e:
        print(f"Error during response generation: {e}")
        return "An error occurred while processing your message. Please try again later."

    # Failsafe response
    return "I'm sorry, I couldn't process that. Can you try again?"

if __name__ == '__main__':
    print("Chatbot is running. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("Chatbot: Goodbye! Have a great day!")
                break
            response = chatbot_response(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
