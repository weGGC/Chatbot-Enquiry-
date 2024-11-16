# scripts/chatbot.py

import spacy
import random
import pickle
import os

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Determine the base directory
script_dir = os.path.dirname(__file__)
model_file = os.path.join(script_dir, '..', 'models', 'chatbot_model.pkl')

# Load the trained model and data
try:
    with open(model_file, 'rb') as f:
        vectorizer, clf, intents = pickle.load(f)
except FileNotFoundError:
    print(f"Error: '{model_file}' not found. Please run 'train_chatbot.py' first.")
    exit()

def chatbot_response(user_input):
    user_input_processed = vectorizer.transform([user_input])
    predicted_tag = clf.predict(user_input_processed)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response

    return "I'm sorry, I didn't understand that. Could you please rephrase?"

if __name__ == '__main__':
    print("Chatbot is running. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
