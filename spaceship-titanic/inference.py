
import requests

ambassador_base_url = "http://locallhost:8775"
url = f"{ambassador_base_url}/predict"
headers = {'Content-Type': 'application/json'}

data = [["5012_01","Earth",True,"G/811/P","55 Cancri e",24,False, 0,0, 0, 0, 0,"Benny Joynewtonks"],
        ['0013_01', 'Earth', True, 'G/3/S', 'TRAPPIST-1e', 27.0, False, 0.0,0.0, 0.0, 0.0, 0.0, 'Nelly Carsoning']]

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result)



# CLI post command
# curl -i \
# --request POST \
# --header "Content-Type: application/json" \
# --data '[["5012_01","Earth",true,"G/811/P","55 Cancri e",24,false, 0,0, 0, 0, 0,"Benny Joynewtonks"]]' \
# 116.47.188.227:32350/predict
# localhost:8775/predict


# monitoring grafana
# id:admin
# password: prom-operator