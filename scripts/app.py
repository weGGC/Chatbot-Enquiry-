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
feedback_file = os.path.join(script_dir, 'feedback.txt')  # File to save feedback
CONFIDENCE_THRESHOLD = 0.03  # Minimum confidence for intent matching

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

# Save feedback to a file
def save_feedback(feedback_text):
    with open(feedback_file, 'a') as f:
        f.write(feedback_text + "\n")

# Load the three most recent feedback entries
def load_recent_feedback():
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_list = f.readlines()
        # Return the three most recent feedbacks
        return feedback_list[-3:]
    return []

# Main Streamlit app
def main():
    # Load model components
    vectorizer, clf, intents = load_model()

    # Add navigation bar
    st.sidebar.title("Navigation")
    nav_option = st.sidebar.radio("Go to", ("Home", "Chatbot", "Feedback", "About Us"))

    # Handle navigation
    if nav_option == "Home":
        st.title("Welcome to SGI College Enquiry System")
        st.write("This platform is designed to assist students with any queries they may have about the college.")
        st.write("For more information about our college, visit [SGI Official Website](https://www.sgi.ac.in)")

    elif nav_option == "Chatbot":
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

    elif nav_option == "Feedback":
        st.title("Feedback")
        st.write("We value your feedback. Please share your thoughts about your experience.")

        # Feedback form
        with st.form(key="feedback_form"):
            feedback_text = st.text_area("Your feedback", key="feedback_input")
            submit_feedback = st.form_submit_button(label="Submit Feedback")

        if submit_feedback and feedback_text.strip():
            save_feedback(feedback_text)
            st.success("Thank you for your feedback!")

        # Display the three most recent feedbacks
        st.subheader("Recent Feedback")
        recent_feedback = load_recent_feedback()
        if recent_feedback:
            for fb in recent_feedback:
                st.write(f"- {fb.strip()}")
        else:
            st.write("No feedback available yet.")

    elif nav_option == "About Us":
        st.title("About Us")
        st.write("SGI is a reputed educational institution dedicated to providing quality education to students.")
        st.write("We offer a variety of courses in different fields to help students excel in their career paths.")

if __name__ == "__main__":
    main()
