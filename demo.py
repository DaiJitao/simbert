
import requests
import json
import base64


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            print(l)
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D

if __name__ == '__main__1':
    data = 'data_sample.json'
    print(load_data(data))

if __name__ == '__main__2':
    restfulUrl = 'http://nlpaip.autohome.com.cn/textcorrect/v2/text_correct'

    text = '瑞虎七'
    data = {"text": text}
    public_key = 'autosearch-enginetzxKPMLt' # KEY:
    private_key = 'rTNa6ffdYRkn' # SECRET:

    authorization = base64.b64encode(bytes(public_key + ':' + private_key, encoding="utf-8")).decode('utf-8')
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Connection": "keep-alive",
               'Authorization': 'Basic ' + authorization}
    try:
        resp = requests.post(restfulUrl, headers=headers, data=data, verify=False)
        result = json.loads(resp.text)
        print(result)
    except Exception as e:
        error_message = str(e)
        print(error_message)



    authorization = base64.b64encode(bytes(public_key + ':' + private_key, encoding="utf-8")).decode('utf-8')
    headers = {'Authorization': 'Basic ' + authorization}
    url = "http://nlpaip.autohome.com.cn/textcorrect/v2/text_correct?text=我开着哈佛汽车和五菱红光在哈佛大学转悠了一圈"

def consin_distance(vec1, vec2):
    if len(vec1) == len(vec2):
        p1 = []
        p2_1 = []
        p2_2 = []
        for x,y in zip(vec1, vec2):
            p1.append(x * y)
            p2_1.append(x ** 2)
            p2_2.append(y ** 2)
        return sum(p1) / ((sum(p2_1) ** .5) * (sum(p2_2) ** .5))

import numpy as np

def softmax(v):
    if v and len(v) > 0:
        res = []
        for i in v:
            res.append(np.exp(i))

        sum_ = np.sum(res)
        return [t/sum_ for t in res]

def crossentry(v1,v2):
    v1 = softmax(v1)
    v2 = softmax(v2)
    t_sum = 0
    for x,y in zip(v1, v2):
        t_sum += (x * np.log(y))

    return -t_sum


if __name__ == '__main__':
    v1 = [0,8,1,1]
    v2 = [9,4,0,2]
    print(softmax(v1))
    print(crossentry(v1,v2))