import streamlit as st
import random
import pickle
import os
from utils import spacy_tokenizer
from streamlit_chat import message  # Import the message component

# Set page configuration at the very top
st.set_page_config(page_title="College Enquiry Chatbot", page_icon=":robot_face:", layout="centered")

# Determine the base directory
script_dir = os.path.dirname(__file__)
model_file = os.path.join(script_dir, '..', 'models', 'chatbot_model.pkl')

# Define load_model() function but do not call it yet
@st.cache_resource
def load_model():
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: '{model_file}' not found. Please run 'train_chatbot.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        st.stop()

# Function to generate chatbot response
def chatbot_response(user_input, vectorizer, clf, intents):
    cleaned_input = ' '.join(spacy_tokenizer(user_input))
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
    # Now call load_model() inside main()
    vectorizer, clf, intents = load_model()

    # Apply custom CSS
    st.markdown("""
        <style>
        /* Container for the chat history */
        .chat-container {
            max-height: 400px; /* Adjust the height as needed */
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #f9f9f9;
        }
        /* Chat message bubbles */
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 10px;
            text-align: left;
            width: fit-content;
            max-width: 70%;
        }
        .bot-message {
            background-color: #E4E6EB;
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 10px;
            text-align: left;
            width: fit-content;
            max-width: 70%;
        }
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        .chat-container::-webkit-scrollbar-thumb {
            background-color: #c1c1c1;
            border-radius: 4px;
        }
        /* Input box styling */
        .stTextInput > div > div > input {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("College Enquiry Chatbot")
    st.write("Welcome! How can I assist you today?")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Display conversation history in a container
    chat_container = st.container()

    # Use a form to handle user input
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Type your message here...", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        response = chatbot_response(user_input, vectorizer, clf, intents)
        st.session_state.history.append({"message": user_input, "is_user": True})
        st.session_state.history.append({"message": response, "is_user": False})

    # Display conversation history with unique keys inside the chat container
    with chat_container:
        # Create a scrollable chat area
        st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)

        for idx, chat in enumerate(st.session_state.history):
            if chat["is_user"]:
                message(chat["message"], is_user=True, key=f'user_{idx}')
            else:
                message(chat["message"], is_user=False, key=f'bot_{idx}')

        st.markdown("</div>", unsafe_allow_html=True)

        # Add a script to auto-scroll to the bottom
        st.markdown("""
            <script>
            const chatContainer = window.parent.document.getElementById('chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
            </script>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
