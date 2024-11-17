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
        initial_sidebar_state="expanded"
    )

    # File paths and configuration
    script_dir = os.path.dirname(__file__)
    model_file = os.path.join(script_dir, '..', 'models', 'chatbot_model.pkl')
    feedback_file = os.path.join(script_dir, 'feedback.txt')  # File to save feedback
    notice_file = os.path.join(script_dir, 'notice.txt')  # File to save notices
    CONFIDENCE_THRESHOLD = 0.024  # Minimum confidence for intent matching

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
        """
        Generate a chatbot response based on user input.
        """
        if not user_input.strip():
            return "Please enter a valid message."

        # Predefined responses for specific queries
        predefined_responses = {
            "who are you": "I'm a robot, your virtual assistant here to help with college-related queries.",
            "what are you": "I'm a robot, designed to provide you with information and assistance about SGI College.",
            "notice": "You can view the latest notices on our notice board here: [SGI Notice Board](https://your-notice-board-link.com)",
            "notice board": "You can view the latest notices on our notice board here: [SGI Notice Board](https://your-notice-board-link.com)"
        }

        # Check for predefined responses
        user_input_cleaned = user_input.strip().lower()
        if user_input_cleaned in predefined_responses:
            return predefined_responses[user_input_cleaned]

        try:
            # Process input and predict intent
            cleaned_input = ' '.join(spacy_tokenizer(user_input))
            user_input_processed = vectorizer.transform([cleaned_input])
            predicted_probabilities = clf.predict_proba(user_input_processed)
            predicted_tag = clf.predict(user_input_processed)[0]
            max_proba = max(predicted_probabilities[0])

            # Handle low-confidence predictions
            if max_proba < CONFIDENCE_THRESHOLD:
                return "I'm sorry, I didn't quite understand that. Could you please rephrase?"

            # Return a response based on the predicted intent
            for intent in intents:
                if intent['tag'] == predicted_tag:
                    return random.choice(intent['responses'])
        except Exception as e:
            return "I'm sorry, something went wrong. Could you please rephrase your query?"

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

    # Save notice to a file
    def save_notice(notice_text):
        with open(notice_file, 'w') as f:
            f.write(notice_text)

    # Load notice
    def load_notice():
        if os.path.exists(notice_file):
            with open(notice_file, 'r') as f:
                return f.read()
        return "No notices available."

    # Utility: Apply CSS globally
    def apply_css():
        st.markdown(
            """
            <style>
            .stButton > button {
                width: 100%;
                margin: 5px 0;
                background-color: #1e1e1e;
                color: #ffffff;
                border: none;
                padding: 10px;
                text-align: left;
                font-size: 16px;
                transition: background-color 0.3s, transform 0.2s;
            }
            .stButton > button:hover {
                background-color: #444;
            }
            .stButton > button:focus {
                background-color: #0a58ca !important;
                color: white !important;
                outline: none !important;
                box-shadow: none !important;
            }
            .stButton > button:active {
                transform: scale(0.98);
            }
            .st-sidebar {
                width: 240px;
            }
            .chatbot-container {
                max-width: 800px;
                margin: 0 auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Render different sections
    def render_home():
        st.title("Welcome to SGI College Enquiry System")
        st.write("This platform is designed to assist students with any queries they may have about the college.")
        st.write("For more information about our college, visit [SGI Official Website](https://www.sgi.ac.in)")

    def render_feedback():
        st.title("Feedback")
        st.write("We value your feedback. Please share your thoughts about your experience.")

        with st.form(key="feedback_form"):
            feedback_text = st.text_input("Your feedback", key="feedback_input")  # Changed to text_input for one line input
            submit_feedback = st.form_submit_button(label="Submit Feedback")

        if submit_feedback and feedback_text.strip():
            save_feedback(feedback_text)
            st.success("Thank you for your feedback!")

        st.subheader("Recent Feedback")
        recent_feedback = load_recent_feedback()
        if recent_feedback:
            for fb in recent_feedback:
                st.write(f"- {fb.strip()}")
        else:
            st.write("No feedback available yet.")

    def render_about():
        st.title("About Us")
        st.write("SGI is a reputed educational institution dedicated to providing quality education to students.")
        st.write("We offer a variety of courses in different fields to help students excel in their career paths.")

    def render_notice():
        st.title("Notice Board")
        user_type = st.radio("Select User Type", ("User", "Admin"), key="user_type")

        if user_type == "Admin":
            admin_password = st.text_input("Enter Admin Password", type="password", key="admin_password")
            if admin_password == "admin123":  # Example password, replace with a secure authentication method
                new_notice = st.text_area("Set Notice", key="notice_input")
                if st.button("Submit Notice", key="submit_notice"):
                    save_notice(new_notice)
                    st.success("Notice updated successfully!")
            else:
                st.warning("Incorrect password. Please try again.")
        else:
            notice = load_notice()
            st.write("### Current Notice")
            st.write(notice)

    # Main Streamlit app
    def main():
        apply_css()
        vectorizer, clf, intents = load_model()

        with st.sidebar:
            st.title("🎓 SGI College Enquiry")
            st.markdown("---")
            if st.button("Home", key="home_button"):
                st.session_state.nav_option = "Home"
            if st.button("Chatbot", key="chatbot_button"):
                st.session_state.nav_option = "Chatbot"
            if st.button("Feedback", key="feedback_button"):
                st.session_state.nav_option = "Feedback"
            if st.button("About Us", key="about_button"):
                st.session_state.nav_option = "About Us"
            if st.button("Notice Board", key="notice_button"):
                st.session_state.nav_option = "Notice Board"

        if 'nav_option' not in st.session_state:
            st.session_state.nav_option = "Chatbot"

        nav_option = st.session_state.nav_option

        if nav_option == "Home":
            render_home()
        elif nav_option == "Chatbot":
            # Retain original Chatbot Section
            left_col, center_col, right_col = st.columns([1, 2, 1])
            with center_col:
                st.title("🎓 SGI Chatbot")
                st.write("Welcome! I'm here to assist you with any questions you might have about our college.")
                initialize_chat()
                chat_placeholder = st.container()
                with st.form(key="chat_form", clear_on_submit=True):
                    user_input = st.text_input("Type your message here...", key="user_input")
                    submit_button = st.form_submit_button(label="Send")
                if submit_button and user_input.strip():
                    user_message = {
                        "message": user_input,
                        "is_user": True,
                        "key": f"message_{st.session_state.message_counter}"
                    }
                    st.session_state.history.append(user_message)
                    st.session_state.message_counter += 1
                    response = chatbot_response(user_input, vectorizer, clf, intents)
                    bot_message = {
                        "message": response,
                        "is_user": False,
                        "key": f"message_{st.session_state.message_counter}"
                    }
                    st.session_state.history.append(bot_message)
                    st.session_state.message_counter += 1
                with chat_placeholder:
                    render_chat()
        elif nav_option == "Feedback":
            render_feedback()
        elif nav_option == "About Us":
            render_about()
        elif nav_option == "Notice Board":
            render_notice()

    if __name__ == "__main__":
        main()
