import requests

BASE_URL = "http://127.0.0.1:8000"

def health_check():
    response = requests.get(BASE_URL)
    return response.json()