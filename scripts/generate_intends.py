# scripts/generate_intents.py

import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Error: GROQ_API_KEY not found. Please set it in the environment or .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)

# Determine the base directory (one level up from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(base_dir, 'data')

# Load scraped data
def load_scraped_data():
    try:
        with open(os.path.join(data_dir, 'scraped_data.txt'), 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("Error: 'scraped_data.txt' not found. Please run 'scrape_website.py' first.")
        return None

# Generate intents using Groq API
def generate_intents(scraped_data):
    prompt = (
        "Based on the following text, generate a list of intents for a college inquiry chatbot in valid JSON format. "
        "Each intent should include a 'tag', a list of 'patterns' representing different ways users might ask about the topic, "
        "and 'responses' providing suitable answers. The output must be a JSON array with intents objects and nothing else.\n\n"
        "Text:\n" + scraped_data
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"  # Replace with the Groq model you want to use
    )

    raw_output = response.choices[0].message.content.strip()
    
    # Print the raw response to help understand what is returned
    print("Raw Response from Groq API:")
    print(raw_output)

    return raw_output

# Sanitize and parse the output
def sanitize_and_parse_intents(intents_text):
    try:
        # Extract the JSON content from the raw text
        start_idx = intents_text.find("[")
        end_idx = intents_text.rfind("]") + 1
        if start_idx != -1 and end_idx != -1:
            intents_json = intents_text[start_idx:end_idx]
        else:
            raise ValueError("Unable to locate JSON content in the generated output.")

        # Parse JSON
        intents = json.loads(intents_json)
        return intents
    except json.JSONDecodeError:
        print("Error: Failed to parse the generated intents as JSON.")
        # Save the raw output to a file for further inspection
        with open(os.path.join(data_dir, 'intents_raw_output.txt'), 'w', encoding='utf-8') as f:
            f.write(intents_text)
        print("The raw output has been saved to 'intents_raw_output.txt' for inspection.")
        return None

# Save intents to a JSON file
def save_intents(intents):
    if intents is None:
        print("Error: No intents to save.")
        return

    intents_file = os.path.join(data_dir, 'intents.json')
    with open(intents_file, 'w', encoding='utf-8') as f:
        json.dump(intents, f, indent=4)

    print(f"Intents saved to {intents_file}")

if __name__ == '__main__':
    scraped_data = load_scraped_data()
    if scraped_data:
        intents_text = generate_intents(scraped_data)
        intents = sanitize_and_parse_intents(intents_text)
        save_intents(intents)
