
import json

inf='all_data.txt_bak'
outf='all_data_pos.json'
outfp = open(outf,'w',encoding='utf-8')
res = {}
with open(inf,'r',encoding='utf-8') as fp:
	for line in fp:
		arr = line.strip().split('\t')
		key =arr[0]
		t = arr[1]
		num = arr[2]
		if num != '0':
			if key not in res:
				res[key] = [t]
			else:
				res[key].append(t)

		data = {}
		for key, ts in res.items():
			data[key] = ts
			tmp = json.dumps(data)
			outfp.wirte('{}\n'.format(tmp))
			outfp.flush()
			data = {}

outfp.close()
print('ok saved in {}'.format(outf))
