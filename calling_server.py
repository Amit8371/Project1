import requests

# r = requests.get('http://127.0.0.1:5000/help')0
r = requests.post('http://127.0.0.1:5000/get_iris_predictions', json={"mydata": '4'})
r.status_code
r.text