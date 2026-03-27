# test_api.py
import requests
import json

url = "http://127.0.0.1:8000/predict"

test_data = {
    "temperature": 25.0,
    "humidity": 60.0,
    "wind_speed": 2.5,
    "general_diffuse_flows": 100.0,
    "year": 2024,
    "month": 6,
    "day": 15,
    "hour": 14
}

print("📤 Sending prediction request...")
print(json.dumps(test_data, indent=2))

response = requests.post(url, json=test_data)

if response.status_code == 200:
    print("\n✅ Prediction successful!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)