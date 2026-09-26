import requests

URL = "http://127.0.0.1:5001/"

data = [
    {
        "bedrooms": 3,
        "bathrooms": 1,
        "sqft_living": 1180,
        "sqft_lot": 5650,
        "floors": 1,
        "waterfront": 0,
        "condition": 3,
    }
]

if __name__ == "__main__":
    response = requests.post(URL, json=data, timeout=10)
    print(response.json())
