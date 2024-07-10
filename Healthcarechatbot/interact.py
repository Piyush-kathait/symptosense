import requests
import json

url = 'http://127.0.0.1:5000/chat'
message = input("Enter your message: ")

data = {'message': message}

response = requests.post(url, json=data)
print(response.json())
