import requests
import base64
import os

# 1. Paste the Google Web App URL you just obtained here
URL = "https://script.google.com/macros/s/AKfycbz5YfcuYeJTLIBB8VcjwwMMjA4C-ePWRryI0rtVGFwiaxRHXM93c4zuUrrVP6oHvYyTYw/exec"

# 2. The path to the file on your server
FILE_PATH = "/usr/local/app/algo_3kings_fund/trades.csv"

def upload():
    # Check if the file exists before attempting upload
    if not os.path.exists(FILE_PATH):
        print("Error: File not found!")
        return

    with open(FILE_PATH, "rb") as f:
        # Convert file content to Base64 format for transmission
        encoded_string = base64.b64encode(f.read()).decode('utf-8')

    # Prepare the payload for the Google Apps Script
    data = {
        "fileData": encoded_string,
        "fileName": "trades_from_server.csv", # The name that will appear in Google Drive
        "mimeType": "text/csv"
    }

    print("Uploading to Google Drive...")
    try:
        # Send the POST request to your Google Script URL
        response = requests.post(URL, data=data)
        print(f"Result: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    upload()