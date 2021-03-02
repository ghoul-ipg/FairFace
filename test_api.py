import requests
content = open('test/race_Latino.jpg','rb')
response= requests.post('http://127.0.0.1:5000/fairface',files={'image': content})
print(response.status_code)
print(response.content)