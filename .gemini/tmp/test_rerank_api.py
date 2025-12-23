import requests, json

url = "http://127.0.0.1:8000/api/rerank"
payload = {
  "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
  "query": "how to fix a leaky shower",
  "documents": [
    {"text": "Shower cartridge replacement: turn off water, remove handle, pull cartridge with pliers…", "original_index": 0},
    {"text": "How to fix a leaky faucet in the kitchen.", "original_index": 1},
    {"text": "Spark plug replacement for a 2012 Civic: remove coil packs, use socket, torque to spec…", "original_index": 2}
  ]
}
try:
    r = requests.post(url, json=payload, timeout=600)
    print("status:", r.status_code)
    print(r.text)
except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: {e}")
except requests.exceptions.Timeout as e:
    print(f"Timeout Error: {e}")
except requests.exceptions.RequestException as e:
    print(f"An unexpected error occurred: {e}")