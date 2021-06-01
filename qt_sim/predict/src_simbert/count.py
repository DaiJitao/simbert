# encoding=utf8
import sys
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.metrics import f1_score

def count_by_http(isAcc=False, isF1=True, isAUC=True, inf = r'pred_test_data.txt'):
    '''
    统计acc AUC
    :param inf:
    :return:
    '''
    y_true = []
    y_pred_sims = []
    y_pred_labels = []
    with open(inf, 'r', encoding='utf-8') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            if len(arr) > 3:
                label = arr[2]
                y_true.append(int(label))
                y_pred_sims.append(float(arr[4]))
                y_pred_labels.append(int(arr[3]))

    if isAcc:
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_labels)
        print('acc:', acc)

    if isAUC:
        auc_score = roc_auc_score(y_true=y_true, y_score=y_pred_sims)
        print('auc:', auc_score)

    if isF1:
        sc = f1_score(y_true=y_true, y_pred=y_pred_labels)
        print('F1-score:', sc)

def count(isAcc=False, isF1=True, isAUC=True, inf = r'pred_test_data.txt'):
    '''
    统计acc AUC
    :param inf:
    :return:
    '''
    y_true = []
    y_pred_sims = []
    y_pred_labels = []
    with open(inf, 'r', encoding='utf-8') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            if len(arr) > 3:
                label = arr[2]
                y_true.append(int(label))
                y_pred_sims.append(float(arr[3]))
                sim = float(arr[3])
                if sim > 0.5:
                    y_pred_labels.append(1)
                else:
                    y_pred_labels.append(0)

    if isAcc:
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_labels)
        print('acc:', acc)

    if isAUC:
        auc_score = roc_auc_score(y_true=y_true, y_score=y_pred_sims)
        print('auc:', auc_score)

    if isF1:
        sc = f1_score(y_true=y_true, y_pred=y_pred_labels)
        print('F1-score:', sc)

    # print('auc:', roc_auc_score(y_true, y_pred_labels))
    # f1 = f1_score(y_true, y_pred_labels)
    # print('F1-socre:', f1)

if __name__ == '__main__':
    # name = 'template_test_data_predict_epoch9'
    # inf = 'data/{}.txt'.format(name)
    # print('----------->{} auc:')
    # count(inf=inf)
    # name = 'template_test_data_predict_epoch18'
    # inf = 'data/{}.txt'.format(name)
    # print('----------->{}:'.format(name))
    # count(inf=inf)
    # name = 'template_test_data_predict_epoch20'
    # inf = 'data/{}.txt'.format(name)
    # print('----------->{}:'.format(name))
    # count(inf=inf)
    # name = 'template_test_data_predict_epoch30'
    # inf = 'data/{}.txt'.format(name)
    # print('----------->{}:'.format(name))
    # count(inf=inf)
    # name = 'template_test_data_predict_epoch38'
    # inf = 'data/{}.txt'.format(name)
    # print('----------->{}:'.format(name))
    # count(inf=inf)
    name = 'twin_predict_all_test_data'
    inf = 'data/{}.txt'.format(name)
    print('----------->{}:'.format(name))
    count(isAcc=True, inf=inf)


    name= 'all_test_data_predict_epoch72'
    inf = 'data/{}.txt'.format(name)
    print('----------->{}:'.format(name))
    count(isAcc=True, inf=inf)

    name = 'all_test_data_predict_weights_90'
    inf = 'data/{}.txt'.format(name)
    print('----------->{}:'.format(name))
    count(isAcc=True, inf=inf)

    name = 'all_test_data_predict_epoch38'
    inf = 'data/{}.txt'.format(name)
    print('----------->{}:'.format(name))
    count(isAcc=True, inf=inf)

    name = 'all_test_data_predict_epoch40'
    inf = 'data/{}.txt'.format(name)
    print('----------->{}:'.format(name))
    count(isAcc=True, inf=inf)

    name = 'all_test_data_predict_epoch48'
    inf = 'data/{}.txt'.format(name)
    print('----------->{}:'.format(name))
    count(isAcc=True, inf=inf)

if __name__ == '__main__0':
    inf = 'data/test_data_label0_1k_predict_mlp_midd.txt'
    print('----------->mlp middle model:')
    count(inf)
    print('----------->middle model:')
    inf = 'data/test_data_label0_1k_predict_midd'
    count(inf)
    print('--------->large model:')
    inf = 'data/test_data_label0_1k_predict_large'
    count(inf)

if __name__ == '__main__1':
    mlp_midd_inf = r'data/predict_combine_from_mlp_midd.txt'
    print('----------->mlp middle model:')
    count(mlp_midd_inf)
    print('----------->middle model:')
    midd_inf = r'data/predict_middl.txt'
    count(midd_inf)
    print('--------->large model:')
    large_inf = 'data/predict_large.txt'
    count(large_inf)
    print('----->all large:')
    inf = 'data/predict_all_large.txt'
    count(inf)