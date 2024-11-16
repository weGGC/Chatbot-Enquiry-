import streamlit as st
import random
import pickle
import os
from utils import spacy_tokenizer  # Import the tokenizer

# Determine the base directory
script_dir = os.path.dirname(__file__)
model_file = os.path.join(script_dir, '..', 'models', 'chatbot_model.pkl')

# Load the trained model and data
def load_model():
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: '{model_file}' not found. Please run 'train_chatbot.py' first.")
        st.stop()

vectorizer, clf, intents = load_model()

# Function to generate chatbot response
def chatbot_response(user_input):
    cleaned_input = ' '.join(spacy_tokenizer(user_input))  # Use the imported tokenizer
    user_input_processed = vectorizer.transform([cleaned_input])
    predicted_probabilities = clf.predict_proba(user_input_processed)
    predicted_tag = clf.predict(user_input_processed)[0]
    max_proba = max(predicted_probabilities[0])

    # Set a confidence threshold
    CONFIDENCE_THRESHOLD = 0.2  # Adjust as needed

    if max_proba < CONFIDENCE_THRESHOLD:
        # Provide a default response
        return "I'm sorry, I didn't quite understand that. Could you please rephrase or ask about admissions, courses, tuition, contact information, or campus life?"

    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response

    # Failsafe default response
    return "I'm sorry, I didn't understand that. Could you please rephrase?"


def main():
    st.set_page_config(page_title="College Enquiry Chatbot", page_icon="🤖")
    st.title("College Enquiry Chatbot")
    st.write("Welcome! How can I assist you today?")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("You:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        response = chatbot_response(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Chatbot", response))

    # Display conversation history
    for sender, message in st.session_state.history:
        st.markdown(f"**{sender}:** {message}")

    # Auto-scroll to the bottom to show the latest message
    st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)
    st.markdown("<script>document.getElementById('chat-end').scrollIntoView();</script>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
