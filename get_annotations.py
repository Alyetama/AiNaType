import os
import requests
import sys
from datetime import datetime
import random
import string
import time

def get_random_string(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def load_env():
    with open('.env') as f:
        env_vars = dict(line.strip().split('=', 1) for line in f if '=' in line)
    os.environ.update(env_vars)

def fetch_and_save(url, headers, output_file, retries=3, delay=5):
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return
        else:
            print(f"Error: Received status code {response.status_code} on attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(delay)
    print("Failed to fetch the data after multiple attempts.")
    sys.exit(1)

def main():
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    random_string = get_random_string()

    load_env()

    label_studio_url = os.getenv('LABEL_STUDIO_URL')
    api_key = os.getenv('API_KEY')

    if not label_studio_url or not api_key:
        print("Error: LABEL_STUDIO_URL or API_KEY not set in the environment variables.")
        sys.exit(1)

    headers = {
        "Authorization": f"Token {api_key}"
    }

    export_url_min = f"{label_studio_url}/api/projects/5/export?exportType=JSON_MIN"
    export_url_full = f"{label_studio_url}/api/projects/5/export?exportType=JSON"

    output_file_min = f"project-5-at-{current_date}-{random_string}.json"
    output_file_full = f"project-5-full-at-{current_date}-{random_string}.json"

    fetch_and_save(export_url_min, headers, output_file_min)
    fetch_and_save(export_url_full, headers, output_file_full)

    print(output_file_min)
    print(output_file_full)

if __name__ == "__main__":
    main()

