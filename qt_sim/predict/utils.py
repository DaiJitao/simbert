# encoding=utf8
import sys


def convert_data(inf, outf):
    if outf != None:
        outfp = open(outf, mode='w', encoding='utf-8')
    with open(inf, 'r', encoding='utf-8') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            if arr == 3:
                q,t, label = arr[0], arr[1], arr[2]
                outfp.write('{}\t{}\n'.format(q, t))

if __name__ == '__main__':
    inf = r'D:\upload\ctr.v0.0.2\qt_similarity\combine.dat'
    convert_data(inf, outf=None)