
# In case you want to submit with livy from batch this might be a start

import json
import requests
from pprint import pprint

url = 'https://spark-datascience.azurehdinsight.net/livy/batches'
auth = ('<clusterssh-login>', '<cluster-ssh-login-pasword>')


body = { "file" : "wasb:///repos/spark_hello_world.py" }


post = requests.post(url,
                     auth=auth,
                     headers = {'Content-Type': 'application/json',
                                'X-Requested-By': 'admin'},
                     # data = json.dumps({'kind':'pyspark'}),
                     json = body)

print(post.status_code)
pprint(post.json())

_log = lambda: requests.get(url+'/%s/log'%post.json()['id'], auth=auth).json()
_status = lambda: requests.get(url+'/%s'%post.json()['id'], auth=auth).json()
_active = lambda: requests.get(url, auth=auth).json()

def log():
    for i in _log()['log']: print(i)

def status():
    pprint(_status())
