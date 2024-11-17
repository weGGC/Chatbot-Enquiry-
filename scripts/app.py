import streamlit as st
import random
import pickle
import os
from utils import spacy_tokenizer
from streamlit_chat import message

# Set page configuration
st.set_page_config(
    page_title="🎓 SGI Chatbot",
    page_icon="👨‍🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# File paths and configuration
script_dir = os.path.dirname(__file__)
model_file = os.path.join(script_dir, '..', 'models', 'chatbot_model.pkl')
CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence for intent matching

# Load the trained model and intents
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            if len(model_data) != 3:
                raise ValueError("Model file is corrupted or incomplete.")
            return model_data
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and available.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Generate chatbot responses
def chatbot_response(user_input, vectorizer, clf, intents):
    if not user_input.strip():
        return "Please enter a valid message."

    try:
        cleaned_input = ' '.join(spacy_tokenizer(user_input))
        user_input_processed = vectorizer.transform([cleaned_input])
        predicted_probabilities = clf.predict_proba(user_input_processed)
        predicted_tag = clf.predict(user_input_processed)[0]
        max_proba = max(predicted_probabilities[0])

        if max_proba < CONFIDENCE_THRESHOLD:
            return "I'm sorry, I didn't quite understand that. Could you please rephrase?"

        for intent in intents:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])
    except Exception as e:
        return f"An error occurred: {e}"

    return "I'm sorry, I didn't understand that. Could you please rephrase?"

# Initialize chat history and message counter in session state
def initialize_chat():
    if 'initialized' not in st.session_state:
        st.session_state.history = []
        st.session_state.message_counter = 0  # Initialize message counter
        # Initial chatbot greeting
        initial_message = {
            "message": "Hello! I'm your College Enquiry Chatbot. How can I assist you today?",
            "is_user": False,
            "key": f"message_{st.session_state.message_counter}"
        }
        st.session_state.history.append(initial_message)
        st.session_state.message_counter += 1
        st.session_state.initialized = True  # Mark initialization as done

# Render chat messages with unique keys
def render_chat():
    for chat in st.session_state.history:
        if chat["is_user"]:
            message(chat["message"], is_user=True, key=chat["key"])
        else:
            message(chat["message"], is_user=False, key=chat["key"])

# Main Streamlit app
def main():
    # Load model components
    vectorizer, clf, intents = load_model()

    # Custom CSS for chat UI
    st.markdown("""
        <style>
        /* Chat container styling */
        .chat-container {
            max-height: 600px; 
            overflow-y: auto;
            padding: 20px;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease-in-out;
        }
        .chat-container:hover {
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
        }
        /* User message styling */
        .user-message {
            background-color: #007ACC;
            color: white;
            padding: 15px;
            border-radius: 25px;
            margin-bottom: 10px;
            text-align: left;
            width: fit-content;
            max-width: 75%;
            align-self: flex-end;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
            position: relative;
        }
        .user-message:after {
            content: "👤";
            position: absolute;
            top: -30px;
            right: -10px;
            font-size: 25px;
            animation: bounce 1s infinite;
        }
        .user-message:hover {
            transform: scale(1.05);
        }
        /* Bot message styling */
        .bot-message {
            background-color: #f0f0f0;
            color: #333;
            padding: 15px;
            border-radius: 25px;
            margin-bottom: 10px;
            text-align: left;
            width: fit-content;
            max-width: 75%;
            align-self: flex-start;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
            position: relative;
        }
        .bot-message:after {
            content: "🤖";
            position: absolute;
            top: -30px;
            left: -10px;
            font-size: 25px;
            animation: shake 1s infinite;
        }
        .bot-message:hover {
            transform: scale(1.05);
        }
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        .chat-container::-webkit-scrollbar-thumb {
            background-color: #cccccc;
            border-radius: 10px;
        }
        /* Input box styling */

        .stTextInput>div>div>input:focus {
            border-color: #007ACC;
        }
        /* Send button styling */
        .stButton>button {
            background-color: #007ACC;
            color: white;
            padding: 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
            transition: background-color 0.3s ease, transform 0.2s;
        }
        .stButton>button:hover {
            background-color: #005f99;
            transform: scale(1.05);
        }
        /* Keyframes for animations */
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0); 
            }
            40% {
                transform: translateY(-10px);
            }
            60% {
                transform: translateY(-5px);
            }
        }
        @keyframes shake {
            0%, 100% {
                transform: rotate(0deg);
            }
            25% {
                transform: rotate(2deg);
            }
            50% {
                transform: rotate(-2deg);
            }
            75% {
                transform: rotate(2deg);
            }
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🎓 SGI Chatbot")
    st.write("Welcome! I'm here to assist you with any questions you might have about our college.")

    # Initialize chat history
    initialize_chat()

    # Create a placeholder for the chat messages
    chat_placeholder = st.container()

    # Input form for user message below the chat
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here...", key="user_input")
        submit_button = st.form_submit_button(label="Send")

    # Handle user input
    if submit_button and user_input.strip():
        # Append the user's message to history with a unique key
        user_message = {
            "message": user_input,
            "is_user": True,
            "key": f"message_{st.session_state.message_counter}"
        }
        st.session_state.history.append(user_message)
        st.session_state.message_counter += 1

        # Generate the chatbot's response
        response = chatbot_response(user_input, vectorizer, clf, intents)

        # Append the chatbot's response to history with a unique key
        bot_message = {
            "message": response,
            "is_user": False,
            "key": f"message_{st.session_state.message_counter}"
        }
        st.session_state.history.append(bot_message)
        st.session_state.message_counter += 1

    # Render chat history
    with chat_placeholder:
        render_chat()

    # Auto-scroll to the bottom of the chat container
    st.markdown("""
        <script>
        const chatContainer = window.parent.document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """, unsafe_allow_html=True)

    # Optional: Debugging Aid (Remove or Comment Out in Production)
    # st.write("Chat History:", st.session_state.history)

if __name__ == "__main__":
    main()
