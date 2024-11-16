# scripts/scrape_website.py

import os
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service  # New import
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Determine the base directory (one level up from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(base_dir, 'data')

# Path to ChromeDriver
chromedriver_path = r"D:\GGC\projects\ONKAR PROJECT NIKITA\project attemp 2\chromedriver-win64\chromedriver.exe"

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    return text

def scrape_college_website():
    # Configure Selenium to use headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument("--window-size=1920,1080")  # Set window size to load all elements

    # Initialize the Chrome driver using Service
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # List of URLs to scrape
    urls = [
        'https://sgipolytechnic.in/',
        'https://sgipolytechnic.in/#/department/cse',
        'https://sgipolytechnic.in/#/admissions',
        'https://sgipolytechnic.in/#/about',
        'https://sgipolytechnic.in/#/',
        'https://sgipolytechnic.in/#/about/logo',
        'https://sgipolytechnic.in/#/about/sgp',
        'https://sgipolytechnic.in/#/about/philosophy',
        'https://sgipolytechnic.in/#/about/vision',
        'https://sgipolytechnic.in/#/about/quality',
        'https://sgipolytechnic.in/#/about/Affiliation',
        'https://sgipolytechnic.in/#/about/differentiator',
        'https://sgipolytechnic.in/#/about/achievements',
        'https://sgipolytechnic.in/#/about/socialresponsiblities',
        'https://sgipolytechnic.in/#/department/fy',
        'https://sgipolytechnic.in/#/department/fy/hoddesk',
        'https://sgipolytechnic.in/#/department/fy/teaching-staff',
        'https://sgipolytechnic.in/#/department/fy/technical-staff',
        'https://sgipolytechnic.in/#/department/fy/activities',
        'https://sgipolytechnic.in/#/department/fy/labs',
        'https://sgipolytechnic.in/#/department/cse',
        'https://sgipolytechnic.in/#/department/cse/hoddesk',
        'https://sgipolytechnic.in/#/department/cse/vision-mission',
        'https://sgipolytechnic.in/#/department/cse/outcomes',
        'https://sgipolytechnic.in/#/department/cse/teaching-staff',
        'https://sgipolytechnic.in/#/department/cse/labs',
        'https://sgipolytechnic.in/#/department/cse/achivements',
        'https://sgipolytechnic.in/#/department/cse/activities',
        'https://sgipolytechnic.in/#/department/cse/academic-calender',
        'https://sgipolytechnic.in/#/department/cse/publications'
        


    ]

    all_texts = []

    try:
        for url in urls:
            print(f"Scraping URL: {url}")
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript to load content; adjust as needed

            # Scroll to the bottom to ensure all dynamic content is loaded
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for additional content to load

            # Example: Click on specific tabs or buttons if needed
            # Uncomment and modify the following lines based on the website's structure
            # try:
            #     admissions_tab = WebDriverWait(driver, 10).until(
            #         EC.element_to_be_clickable((By.XPATH, "//a[@href='#/admissions']"))
            #     )
            #     admissions_tab.click()
            #     time.sleep(3)  # Wait for content to load
            # except Exception as e:
            #     print(f"Error clicking Admissions tab: {e}")

            # Get the page source and parse with BeautifulSoup
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Extract text from specific sections
            texts = []

            # Example: Extract all paragraphs, headers, list items, and spans
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'a']):
                text = clean_text(element.get_text())
                if text:
                    all_texts.append(text)

        # Optional: Implement infinite scroll if the website uses it
        # infinite_scroll(driver)

    except Exception as e:
        print(f"An error occurred while scraping: {e}")

    finally:
        driver.quit()

    # Remove duplicates
    all_texts = list(set(all_texts))

    # Optional: Further filter texts if needed
    # For example, remove very short sentences or irrelevant content
    all_texts = [text for text in all_texts if len(text.split()) > 3]

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Save the extracted texts to a file in the base data directory
    scraped_data_file = os.path.join(data_dir, 'scraped_data.txt')
    with open(scraped_data_file, 'w', encoding='utf-8') as f:
        for line in all_texts:
            f.write(line + '\n')

    print(f"Scraping completed. Data saved to {scraped_data_file}")

if __name__ == '__main__':
    scrape_college_website()
