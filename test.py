import urllib.request
import json
import time

try:
    with urllib.request.urlopen('http://localhost:5000') as response:
        print(response.read().decode())
except Exception as e:
    print(f"Error: {e}")
