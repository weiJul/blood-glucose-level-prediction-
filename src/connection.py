import requests
import json

# API Call
class Api_Connection:
    def __init__(self, email, password):
        self.email = email
        self.password = password
        self.api_endpoint = 'https://api-de.libreview.io'
        self.headers = {'accept-encoding': 'gzip',
                        'cache-control': 'no-cache',
                        'connection': 'Keep-Alive',
                        'content-type': 'application/json',
                        'product': 'llu.android',
                        'version': '4.7.0'
                        }
        self.patientId = None

    def get_connection(self):
        loginData = {"email": self.email, "password": self.password}
        r = requests.post(url=self.api_endpoint + "/llu/auth/login", headers=self.headers, json=loginData)
        data = r.json()
        return data, data['status']
    def setToken(self):
        data, status = self.get_connection()
        if status == 0:
            JWT_token = data['data']['authTicket']['token']
            extra_header_info = {'authorization': 'Bearer ' + JWT_token}
            self.headers.update(extra_header_info)
        else:
            print('quit')
            quit()
    def setPatientId(self):
        r = requests.get(url = self.api_endpoint + "/llu/connections", headers=self.headers)
        data = r.json()
        self.patientId = data['data'][0]['patientId']

    def getData(self):
        self.setToken()
        self.setPatientId()
        r = requests.get(url = self.api_endpoint + "/llu/connections/" + self.patientId + "/graph", headers=self.headers)
        return r.json()

    def saveData(self, file_name):
        json_data = self.getData()
        json_formatted_str = json.dumps(json_data, indent=2)
        print(json_formatted_str)
        file_path = 'json/'+file_name+'.json'
        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)