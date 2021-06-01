# encoding=utf8
import sys
import pandas as pd
import re

def containsIllegal(query):
    if "^_^^_^" in query or 'var_dump' in query:
        return True
    pattern = r'^[\d]{1,}$'
    res = re.findall(pattern, query)
    pattern = r"&amp;#\d{0,}"
    res.extend(re.findall(pattern, query))
    if len(res) > 0:
        return True
    pattern = r"，{3,}"
    query = re.sub(pattern, '', query).strip()
    if len(query) < 4:
        return True

    return False

def getLongQuery(inf,outf, pdict, dealLine=200):
    resfp = open(outf, encoding='utf-8', mode='w')
    with open(inf, encoding='utf-8', mode='r') as fp:
        for index, line in enumerate(fp):
            if '�' not in line:
                arr = line.strip().split('')
                query = arr[0].strip()
                num = arr[1].strip().lower()
                if containsIllegal(query):
                    continue

                if len(query) >= 5 and len(query) < 31 and '\\n' not in num and num != '' and 'https:' not in query:
                    num = int(num)
                    if num <= dealLine:
                        res = pdict.get(query)
                        if res == None:
                            temp = "{}\n".format(query)
                            resfp.write(temp)

                if index % 10000 == 0:
                    resfp.flush()

    resfp.close()
    print('ok saved in {}'.format(outf))

def getDict(inf):
    res = {}
    with open(inf, mode='r', encoding='utf-8') as fp:
        for index, line in enumerate(fp):
            key = line.strip()
            res[key] = index + 1
    return res

def convertExel(inf, outf):
    df = pd.read_csv(inf, header=None, delimiter='\t')
    df.to_excel('{}.xls'.format(outf), index=False)
    print('ok saved in {}.xls'.format(outf))

if __name__ == '__main__':
    name = 'all_test_data_predict_epoch40'
    inf = r'data/{}.txt'.format(name)
    outf = r'data/{}'.format(name)
    inf = r'D:\qichezhijia\workspace\pycharm\tensorflow_worksapce\tfVersion_1.15.2\simbert\qt_sim\predict\twin_midd_test_data\predict_all_test_data.txt'
    outf = r'D:\qichezhijia\workspace\pycharm\tensorflow_worksapce\tfVersion_1.15.2\simbert\qt_sim\predict\twin_midd_test_data\predict_all_test_data'

    convertExel(inf, outf)

if __name__ == '__main__1':
    dictf = r'D:\upload\ctr.v0.0.2\qt_similarity\random_query.2020-10-13'
    pdict = getDict(dictf)
    inf = r'D:\upload\ctr.v0.0.2\qt_similarity\000000_0'
    outf = 'long_query_0.txt'
    getLongQuery(inf, outf, pdict)