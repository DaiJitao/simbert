#encoding=utf-8
from get_bert_vec import get_vec_by_query
from flask import Flask
from flask import request
import json
import platform
import traceback

os_p = platform.system().lower()
if os_p == 'linux':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    print('===============> platform {}'.format(os_p))
elif os_p == 'windows':
    print('===============> platform {}'.format(os_p))

app = Flask(__name__)


class InnerResponse(object):
    def __init__(self, text):
        self.text = text

    def Success(self, vec):
        res = {}
        res['status'] = '0'
        res['msg'] = 'success'
        res['result'] = vec
        res['text'] = self.text
        datajson = json.dumps(res)
        return datajson

    def Error(self):
        '''
        输入非法
        :return:
        '''
        res = {}
        res['status'] = '400'
        res['msg'] = 'Illegal input text:{}'.format(self.text)
        res['result'] = []
        res['text'] = self.text
        datajson = json.dumps(res)
        return datajson

    def Failed(self):
        """
        输入合法，但是内部处理错误
        :return:
        """
        res = {}
        res['status'] = '500'
        res['msg'] = 'inertal error'
        res['result'] = []
        res['text'] = self.text
        datajson = json.dumps(res)
        return datajson


@app.route('/vectors/simbert', methods=['GET', "POST"])
def get_simbert_vector():
    text = request.args.get('text')
    print('---->{}'.format(text))
    response = InnerResponse(text=text)
    if text.strip() == '' or text == None:
        return response.Error()

    try:
        vec = get_vec_by_query(text)
    except Exception as e:
        traceback.print_exc()
        return response.Failed()

    if vec == None or len(vec) == 0:
        return response.Failed()
    else:
        return response.Success(vec=vec)


if __name__ == '__main__':
    vec = get_vec_by_query('测试')
    print('----->', vec)
    app.run(port=8899, debug=False)
