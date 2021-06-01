# encoding=utf8
import sys

import numpy as np
import requests
from urllib.parse import unquote, quote
from sklearn.metrics import accuracy_score, auc, roc_auc_score

def pred_by_http(url, query, title):
    query, title = quote(query, 'utf-8'), quote(title, 'utf-8')
    url = url.format(query, title)
    print(url)
    response = requests.get(url)
    result = response.json()
    label = result['label']
    sim = result['similarity']
    return sim, label


def traverse_file(in_path, out_path):
    large_model_url = 'http://qt-sim.large.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    middle_model_url = 'http://qt-sim.midd.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    in_open = open(in_path, 'r', encoding='utf-8')
    out_open = open(out_path, 'w', encoding='utf-8')
    in_lines = in_open.readlines()
    num = 0
    same_num = 0
    for index, in_line in enumerate(in_lines):
        line_list = in_line.strip().split("\t")
        if not len(line_list) == 3:
            continue
        query = line_list[0]
        title = line_list[1]
        label = line_list[2]
        try:
            sim, pred_label = pred_by_http(middle_model_url, query, title)
        except Exception as e:
            print('ERROR:', e)
            continue
        num += 1
        if (label in ('1', '2') and pred_label == '1') or (label == pred_label == '0'):
            same_num += 1
        out_line = query + '\t' + title + '\t' + label + '\t' + pred_label + '\t' + str(sim)
        out_open.write(out_line + '\n')
        if index == 100:
            break
    print('accuracy on test data -> ', same_num / num)
    in_open.close()
    out_open.close()
    print('ok predict file:' + out_path)

def predict_data(inf, outf, url, islabel=False, deli='\t'):
    outfp = open(outf, 'w', encoding='utf-8')
    with open(inf, 'r', encoding='utf-8') as fp:
        for index, line in enumerate(fp):
            arr = line.strip().split(deli)
            if len(arr) == 3:
                query , title, true_label = arr[0], arr[1], arr[2]
                sim, label = pred_by_http(url, query, title)
                if islabel:
                    if float(sim) > 0.5:
                        label = 1
                    else:
                        label = 0

                temp = '{}\t{}\t{}\t{}\t{}\n'.format(query, title, true_label, label, sim)
                outfp.write(temp)
                if index % 50 == 0:
                    outfp.flush()
    outfp.close()
    print('ok saved in {}'.format(outf))

name = 'all_test_data'
inf = 'data/{}.txt'.format(name)
print('inf {}'.format(inf))
if __name__ == '__main__':
    sentence_url='http://qtsim.twincosin.h1.relu3.midd.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    outf = 'data/{}_predict_jingdong_midd.txt'.format(name)
    predict_data(inf, outf, sentence_url, islabel=False)

    # mlp_midd_url = 'http://qtsim.mlp.midd.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    # outf = 'data/{}_predict_mlp_midd.txt'.format(name)
    # predict_data(inf, outf, mlp_midd_url)

    # midd_url = 'http://qt-sim.midd.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    # outf = 'data/{}_predict_midd.txt'.format(name)
    # predict_data(inf, outf, midd_url)

    # large_url = 'http://qt-sim.large.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    # outf = 'data/{}_predict_large.txt'.format(name)
    # predict_data(inf, outf, large_url)

    # all_url = 'http://qt-sim.all.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    # outf = 'data/{}_predict_all_large.txt'.format(name)
    # predict_data(inf, outf, all_url)

if __name__ == '__main__0':
    # mlp_midd_url = 'http://qtsim.mlp.midd.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    # outf = 'data/test_data_label0_1k_predict_mlp_midd.txt'
    # predict_data(inf, outf, mlp_midd_url)

    midd_url = 'http://qt-sim.midd.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    outf = 'data/test_data_label0_1k_predict_midd.txt'
    predict_data(inf, outf, midd_url)

    large_url = 'http://qt-sim.large.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'
    outf = 'data/test_data_label0_1k_predict_large.txt'
    predict_data(inf, outf, large_url)

    all_url = 'http://qt-sim.all.service.diana.corpautohome.com/qt/sim?cls=1&query={}&title={}'


if __name__ == '__main__1':
    query = '伊思坦纳'
    title = '吉利嘉际首台红色实车现身，质感不输宋MAX，空间胜别克GL6'
    res = pred_by_http(query, title)
    print(res)

if __name__ == '__main__1':
    in_path = r'D:\upload\ctr.v0.0.2\qt_similarity\test_data.txt'
    out_path = r'.\pred_test_data.txt'
    traverse_file(in_path, out_path)
