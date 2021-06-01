# encoding=utf8
import sys


def validate(inf, outf, plusf):
    plusfp, outfp = open(plusf, 'w', encoding='utf-8'), open(outf, 'w', encoding='utf-8')
    with open(inf, 'r', encoding='utf-8') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            if len(arr) == 3:
                label = arr[2].strip()
                if label in ['0', '1', '2']:
                    outfp.write(line)
                else:
                    outfp.write(line)
    print('ok saved in {}, \n{}'.format(outf, plusf))

def validate(inf, outf, plusf):
    plusfp, outfp = open(plusf, 'w', encoding='utf-8'), open(outf, 'w', encoding='utf-8')
    with open(inf, 'r', encoding='utf-8') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            if len(arr) == 3:
                label = arr[0].strip()
                if label in ['0', '1', '2']:
                    q, t = arr[0], arr[1]
                    outfp.write('{}\t{}\t{}\n}'.format(q, t ,label))
                else:
                    outfp.write(line)
    print('ok saved in {}, \n{}'.format(outf, plusf))

inf='cleaned_train_data_plus.txt'
outf='ok_data.txt'
plusf='ok_data_plus.txt'
validate(inf, outf, plusf)