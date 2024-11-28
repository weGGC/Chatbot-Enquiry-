import streamlit as st
import random
import pickle
import os
from utils import spacy_tokenizer
from streamlit_chat import message

# Set page configuration
st.set_page_config(
    page_title="🎓 ASAC Chatbot",
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

def render_home():
    # Centered and bold title
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>Welcome to ASAC College Enquiry System</h1><br><br>", unsafe_allow_html=True)

    # Centered heading for "Explore Our College Services"
    st.markdown("<h2 style='text-align: center;'>Explore Our College Services</h2><br><br>", unsafe_allow_html=True)

    # First Section: Content on the left and Image on the right
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("""
            <div style='padding-right: 30px;'>
                <p style='font-size: 18px;'><br><br><br><br>SGI College offers a wide variety of educational programs that cater to the needs of diverse student groups. Whether you're looking for undergraduate, postgraduate, or diploma programs, SGI College is here to guide you every step of the way. We pride ourselves on our strong academic reputation and commitment to student success.</p>
            </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.image("img/college_building.jpg", use_column_width=True, caption="SGI College Campus")

    # Second Section: Image on the left and Content on the right
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.image("img/student_life.jpg", use_column_width=True, caption="Campus Life")

    with right_col:
        st.markdown("""
            <div style='padding-left: 30px;'>
                <p style='font-size: 18px;'><br><br><br><br>At SGI College, we believe in fostering a vibrant student community. Our campus is not just a place for academic learning but also for personal growth. We offer various student clubs, sports activities, and events that encourage students to engage in extracurricular activities, creating a well-rounded college experience.</p>
            </div>
        """, unsafe_allow_html=True)

    # "For more info" section positioned at the bottom-right
    st.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; font-size: 18px; background-color: black; padding: 10px 20px; border-radius: 5px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
            For more information about our programs and student services, visit <a href="https://sgipolytechnic.in/#/" target="_blank" style="color: #007bff; text-decoration: none;">SGI Official Website</a>
        </div>
    """, unsafe_allow_html=True)

    
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
    # Centered, bold, and large title
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>About Us</h1>", unsafe_allow_html=True)
    
    # Centered description about the institution
    st.markdown("<h3 style='text-align: center; font-size: 18px;'>SGI is a reputed educational institution dedicated to providing quality education to students.</h3><br>", unsafe_allow_html=True)

    # Upload image using st.image() at the top of the page
    st.image("img/background.jpg", use_column_width=True)
    
    # Path to the background image for div sections
    background_image_path = "img/background.jpg"
    
    # Section for "Why Use a Chatbot for Learning?"
    st.markdown(
        f"""
        <div style="background-image: url('{background_image_path}'); background-size: cover; background-position: center center; padding: 20px; border-radius: 8px; color: white; text-align: center;">
            <h2>Why Use a Chatbot for Learning?</h2>
            <p>A chatbot can be a helpful resource for memorization tasks. By asking or responding to a set of questions, students can learn through repetition as well as accompanying explanations. The chatbot will not tire as students use it repeatedly, and is available as a practice partner at any time of day or night.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Section for "We Provide the Answers for Each Student"
    st.markdown(
        f"""
        <div style="background-image: url('{background_image_path}'); background-size: cover; background-position: center center; padding: 20px; border-radius: 8px; color: white; text-align: center;">
            <h2>We Provide the Answers for Each Student</h2>
            <p>Some great placeholder content for the first featurette here. Imagine some exciting prose here. At the most basic level, a chatbot is a computer program that simulates and processes human conversation (either written or spoken), allowing humans to interact with digital devices as if they were communicating with a real person.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )




def render_notice():
    # Center the "Notice Board" title
    st.markdown("<h1 style='text-align: center;'>Notice Board</h1><br><br>", unsafe_allow_html=True)
    
    # Dropdown to select User or Admin
    user_type = st.selectbox("Select User Type", ["User", "Admin"], key="user_type")

    if user_type == "Admin":
        # Admin section to enter password and update notice
        admin_password = st.text_input("Enter Admin Password", type="password", key="admin_password")
        if admin_password == "admin123":  # Example password, replace with a secure authentication method
            new_notice = st.text_area("Set Notice", key="notice_input", height=150)
            authority = st.selectbox("Approved By", ["HOD", "Principal", "Director"], key="authority_select")

            if st.button("Submit Notice", key="submit_notice"):
                notice_text = f"{new_notice}\n\nApproved by: {authority}"
                save_notice(notice_text)
                st.success("Notice updated successfully!")

        else:
            st.warning("Incorrect password. Please try again.")

    else:
        # Display current notice for User
        notice = load_notice()
        st.markdown("<h3 style='text-align: center;'>Current Notice</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="text-align: center; font-size: 40px; padding: 20px; border-radius: 8px; ;">
                {notice}
            </div>
        """, unsafe_allow_html=True)

        


# Main Streamlit app
def main():
    apply_css()
    vectorizer, clf, intents = load_model()

    with st.sidebar:
        st.title("🎓 SGI College Enquiry")
        st.markdown("---")
        if st.button("Home", key="home_button"):
            st.session_state.nav_option = "Home"
        if st.button("About Us", key="about_button"):
            st.session_state.nav_option = "About Us"
        if st.button("Chatbot", key="chatbot_button"):
            st.session_state.nav_option = "Chatbot"
        if st.button("Notice Board", key="notice_button"):
            st.session_state.nav_option = "Notice Board"
        if st.button("Feedback", key="feedback_button"):
            st.session_state.nav_option = "Feedback"


    if 'nav_option' not in st.session_state:
        st.session_state.nav_option = "Chatbot"

    nav_option = st.session_state.nav_option

    if nav_option == "Home":
        render_home()
    elif nav_option == "Chatbot":
        # Retain original Chatbot Section
        left_col, center_col, right_col = st.columns([1, 2, 1])
        with center_col:
            st.title("🎓 ASAC Chatbot")
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
